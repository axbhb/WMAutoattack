from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from agent.llm import (
    AttackerAgent,
    AttackSearchState,
    AuditorAgent,
    build_reflection_note,
    normalized_reward_drop,
    pareto_front,
    scalarized_utility,
)
from agent.memory import ExperienceMemoryStore
from agent.schema import AttackSearchSpace, ProbeRepresentation, SearchConfig, TaskProfile, TaskSpec, TrialConfig, TrialResult
from autoattack.runtime import DreamerV3SearchExecutor


class DebateSearchController:
    def __init__(
        self,
        executor: DreamerV3SearchExecutor,
        attacker_agent: AttackerAgent,
        auditor_agent: AuditorAgent,
        search_config: SearchConfig,
    ) -> None:
        self.executor = executor
        self.attacker_agent = attacker_agent
        self.auditor_agent = auditor_agent
        self.search_config = search_config
        self.output_dir = search_config.ensure_output_dir()
        self.transcript_path = self.output_dir / "debate_transcript.jsonl"
        self.memory_store = ExperienceMemoryStore(search_config.experience_store_path)

    def run(self, tasks: Iterable[TaskSpec], search_spaces: Dict[str, AttackSearchSpace]) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "search_mode": self.search_config.search_mode,
            "initialization_mode": self.search_config.initialization_mode,
            "agent_backend": self.search_config.agent_backend,
            "agent_model": self.search_config.agent_model,
            "experience_store_path": self.search_config.experience_store_path,
            "experience_retrieval_mode": self.search_config.experience_retrieval_mode,
            "tasks": [],
        }
        for task in tasks:
            task_summary = self._run_task(task, search_spaces)
            summary["tasks"].append(task_summary)

        summary_path = self.output_dir / "search_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path = self.output_dir / "search_summary.txt"
        txt_path.write_text(self._to_text(summary), encoding="utf-8")
        return summary

    def _run_task(self, task: TaskSpec, search_spaces: Dict[str, AttackSearchSpace]) -> Dict[str, object]:
        baseline_config = TrialConfig(
            task_name=task.name,
            checkpoint_path=task.checkpoint_path,
            attack_name="baseline",
            epsilon=0.0,
            steps=0,
        )
        baseline_result = self.executor.run_trial(
            baseline_config,
            stage="confirm",
            num_episodes=self.search_config.confirm_episodes,
            persist_artifacts=True,
        )
        task_profile = self.executor.describe_task(task, baseline_result)
        self._write_transcript({"event": "baseline", "task": task.name, "result": baseline_result.to_dict()})

        task_summary: Dict[str, object] = {
            "task": task.name,
            "checkpoint_path": task.checkpoint_path,
            "task_profile": task_profile.to_dict(),
            "baseline": baseline_result.to_dict(),
            "attacks": {},
        }

        for attack_name, search_space in search_spaces.items():
            state = AttackSearchState(
                task=task,
                search_space=search_space,
                baseline_result=baseline_result,
                runtime_budget_seconds=self.search_config.runtime_budget_seconds,
                initialization_mode=self.search_config.initialization_mode,
                seed=self.search_config.seed,
                task_profile=task_profile,
                prior_experiences=self.memory_store.retrieve(
                    task_profile,
                    attack_name,
                    limit=self.search_config.experience_retrieval_limit,
                    mode=self.search_config.experience_retrieval_mode,
                    query_probe=task_profile.probe_representation,
                    latent_projection=self.search_config.experience_latent_projection,
                    latent_dim=self.search_config.experience_latent_dim,
                    hybrid_weight=self.search_config.experience_hybrid_weight,
                ),
            )

            while len(state.confirmed_results) < self.search_config.max_trials_per_attack:
                proposals = self.attacker_agent.propose(state, self.search_config.proposal_batch_size)
                if len(proposals) == 0:
                    break
                self._write_transcript(
                    {
                        "event": "proposal_batch",
                        "task": task.name,
                        "attack": attack_name,
                        "configs": [proposal.to_dict() for proposal in proposals],
                    }
                )
                scout_results: List[TrialResult] = []
                for proposal in proposals:
                    scout_result = self.executor.run_trial(
                        proposal,
                        stage="scout",
                        num_episodes=self.search_config.scout_episodes,
                        persist_artifacts=False,
                    )
                    scout_audit = self.auditor_agent.audit(state, scout_result)
                    state.record_result(scout_result)
                    state.record_audit(scout_audit)
                    state.record_reflection(build_reflection_note(state.baseline_result, scout_result, scout_audit))
                    self._refresh_prior_experiences(state, scout_result)
                    scout_results.append(scout_result)
                    self._write_transcript(
                        {
                            "event": "scout_result",
                            "task": task.name,
                            "attack": attack_name,
                            "result": scout_result.to_dict(),
                            "audit": scout_audit.to_dict(),
                        }
                    )

                confirm_candidates = self._select_confirm_candidates(state, scout_results)
                for candidate in confirm_candidates:
                    if candidate.config.key() in state.confirmed_results:
                        continue
                    confirm_result = self.executor.run_trial(
                        candidate.config,
                        stage="confirm",
                        num_episodes=self.search_config.confirm_episodes,
                        persist_artifacts=True,
                    )
                    confirm_audit = self.auditor_agent.audit(state, confirm_result)
                    state.record_result(confirm_result)
                    state.record_audit(confirm_audit)
                    state.record_reflection(build_reflection_note(state.baseline_result, confirm_result, confirm_audit))
                    self._refresh_prior_experiences(state, confirm_result)
                    self._write_transcript(
                        {
                            "event": "confirm_result",
                            "task": task.name,
                            "attack": attack_name,
                            "result": confirm_result.to_dict(),
                            "audit": confirm_audit.to_dict(),
                        }
                    )
                    if len(state.confirmed_results) >= self.search_config.max_trials_per_attack:
                        break

            attack_summary = self._summarize_state(state)
            task_summary["attacks"][attack_name] = attack_summary
            self._persist_experience(task_profile, attack_name, attack_summary, baseline_result)
        return task_summary

    def _select_confirm_candidates(
        self,
        state: AttackSearchState,
        scout_results: List[TrialResult],
    ) -> List[TrialResult]:
        ranked = sorted(
            scout_results,
            key=lambda trial: scalarized_utility(state.baseline_result, trial),
            reverse=True,
        )
        return ranked[: self.search_config.confirm_top_k]

    def _summarize_state(self, state: AttackSearchState) -> Dict[str, object]:
        confirmed = list(state.confirmed_results.values())
        scout = list(state.scout_results.values())
        best = state.best_result()
        front = pareto_front(state.baseline_result, confirmed or scout)
        return {
            "best_trial": None if best is None else best.to_dict(),
            "pareto_front": [trial.to_dict() for trial in front],
            "confirmed_trials": [trial.to_dict() for trial in confirmed],
            "scout_trials": [trial.to_dict() for trial in scout],
            "audits": [audit.to_dict() for audit in state.audits.values()],
            "reflections": [reflection.to_dict() for reflection in state.reflections],
            "retrieved_experiences": [item.to_dict() for item in state.prior_experiences],
        }

    def _persist_experience(
        self,
        task_profile: TaskProfile,
        attack_name: str,
        attack_summary: Dict[str, object],
        baseline_result: TrialResult,
    ) -> None:
        best = attack_summary.get("best_trial")
        if best is None:
            return
        result_summary = {
            "mean_reward": best["mean_reward"],
            "std_reward": best["std_reward"],
            "median_reward": best["median_reward"],
            "elapsed_seconds": best["elapsed_seconds"],
            "telemetry": best.get("telemetry", {}),
        }
        entry = self.memory_store.build_entry(
            task_profile=task_profile,
            attack_name=attack_name,
            best_config=dict(best["config"]),
            result_summary=result_summary,
            utility=normalized_reward_drop_obj({"mean_reward": baseline_result.mean_reward}, best),
            source_run_dir=str(self.output_dir),
            notes=["auto_saved_from_reflexion_search"],
            probe_representation=ProbeRepresentation.from_dict(best.get("probe_representation")),
        )
        self.memory_store.append(entry)

    def _refresh_prior_experiences(self, state: AttackSearchState, result: TrialResult) -> None:
        if self.search_config.experience_retrieval_mode not in ("latent", "hybrid"):
            return
        query_probe = result.probe_representation or (None if state.task_profile is None else state.task_profile.probe_representation)
        if query_probe is None:
            return
        state.prior_experiences = self.memory_store.retrieve(
            state.task_profile or TaskProfile(task_name=state.task.name, checkpoint_path=state.task.checkpoint_path),
            state.search_space.attack_name,
            limit=self.search_config.experience_retrieval_limit,
            mode=self.search_config.experience_retrieval_mode,
            query_probe=query_probe,
            latent_projection=self.search_config.experience_latent_projection,
            latent_dim=self.search_config.experience_latent_dim,
            hybrid_weight=self.search_config.experience_hybrid_weight,
        )

    def _write_transcript(self, payload: Dict[str, object]) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _to_text(self, summary: Dict[str, object]) -> str:
        lines: List[str] = []
        lines.append("DreamerV3 AutoAttack Multi-Agent Search Summary")
        lines.append("Search mode: {}".format(summary.get("search_mode", "unknown")))
        lines.append("Initialization mode: {}".format(summary.get("initialization_mode", "task_conditioned")))
        lines.append(
            "Agent backend: {} ({})".format(
                summary.get("agent_backend", "unknown"),
                summary.get("agent_model", "unknown"),
            )
        )
        lines.append("Experience store: {}".format(summary.get("experience_store_path", "unknown")))
        lines.append("Experience retrieval mode: {}".format(summary.get("experience_retrieval_mode", "structured")))
        lines.append("")
        for task_summary in summary["tasks"]:
            lines.append(f"Task: {task_summary['task']}")
            baseline = task_summary["baseline"]
            lines.append(
                "  Baseline: "
                f"mean={baseline['mean_reward']:.4f}, std={baseline['std_reward']:.4f}, "
                f"time={baseline['elapsed_seconds']:.2f}s"
            )
            for attack_name, attack_summary in task_summary["attacks"].items():
                best = attack_summary["best_trial"]
                if best is None:
                    lines.append(f"  {attack_name}: no confirmed trial")
                    continue
                best_config = best["config"]
                drop = normalized_reward_drop_obj(baseline, best)
                lines.append(
                    f"  {attack_name}: best eps={best_config['epsilon']}, steps={best_config['steps']}, "
                    f"allocation={best_config['allocation']['mode']}, reward_drop={drop:.4f}, "
                    f"mean={best['mean_reward']:.4f}, time={best['elapsed_seconds']:.2f}s"
                )
            lines.append("")
        return "\n".join(lines)


def normalized_reward_drop_obj(baseline: Dict[str, object], trial: Dict[str, object]) -> float:
    scale = abs(float(baseline["mean_reward"])) + 1.0
    return (float(baseline["mean_reward"]) - float(trial["mean_reward"])) / scale
