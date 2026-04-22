from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping

from agent.memory import ExperienceMemoryStore
from agent.schema import SearchConfig, TaskProfile, TaskSpec, TrialConfig, TrialResult

if TYPE_CHECKING:
    from agent.orchestration import DebateSearchController
    from autoattack.runtime import DreamerV3SearchExecutor


@dataclass(frozen=True)
class ComparisonMethod:
    name: str
    retrieval_mode: str
    description: str


def build_parser() -> argparse.ArgumentParser:
    from agent.run_search import build_parser as build_search_parser

    parser = build_search_parser()
    parser.description = "Fair comparison between a Claudini-style LLM baseline and the latent-RAG method."
    parser.add_argument(
        "--baseline-method-name",
        type=str,
        default="claudini_style",
        help="Label for the generic LLM self-evolution baseline.",
    )
    parser.add_argument(
        "--ours-method-name",
        type=str,
        default="ours",
        help="Label for the latent-RAG + self-evolution method.",
    )
    parser.add_argument(
        "--baseline-retrieval-mode",
        choices=("none", "structured", "latent", "hybrid"),
        default="none",
        help="Experience retrieval backend for the Claudini-style baseline.",
    )
    parser.add_argument(
        "--ours-retrieval-mode",
        choices=("none", "structured", "latent", "hybrid"),
        default="latent",
        help="Experience retrieval backend for the proposed method.",
    )
    parser.add_argument(
        "--shared-baseline-retrieval-mode",
        choices=("none", "latent", "hybrid"),
        default="latent",
        help="Probe collection mode used only for the shared clean baseline/task profile pass.",
    )
    parser.add_argument(
        "--allow-same-checkpoint-memory",
        action="store_true",
        help="Do not filter entries that come from the exact target checkpoint when seeding method-specific memory stores.",
    )
    return parser


def _default_output_dir(raw_output_dir: str) -> str:
    if raw_output_dir:
        return raw_output_dir
    return str(Path("logs") / "attack_search_compare" / datetime.now().strftime("%Y%m%d_%H%M%S"))


def _build_search_config(
    args: argparse.Namespace,
    *,
    output_dir: str,
    experience_store_path: str,
    experience_retrieval_mode: str,
) -> SearchConfig:
    return SearchConfig(
        output_dir=output_dir,
        initialization_mode=args.initialization_mode,
        scout_episodes=args.scout_episodes,
        confirm_episodes=args.confirm_episodes,
        proposal_batch_size=args.proposal_batch_size,
        confirm_top_k=args.confirm_top_k,
        max_trials_per_attack=args.max_trials_per_attack,
        runtime_budget_seconds=args.runtime_budget_seconds,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        agent_backend=args.agent_backend,
        agent_model=args.agent_model,
        agent_api_key_env=args.agent_api_key_env,
        agent_base_url=args.agent_base_url,
        agent_max_candidates=args.agent_max_candidates,
        agent_context_limit=args.agent_context_limit,
        agent_max_new_tokens=args.agent_max_new_tokens,
        agent_temperature=args.agent_temperature,
        agent_cache_dir=args.agent_cache_dir,
        experience_store_path=experience_store_path,
        experience_retrieval_limit=args.experience_retrieval_limit,
        experience_retrieval_mode=experience_retrieval_mode,
        experience_latent_projection=args.experience_latent_projection,
        experience_latent_dim=args.experience_latent_dim,
        experience_hybrid_weight=args.experience_hybrid_weight,
        experience_probe_max_steps=args.experience_probe_max_steps,
    )


def _enabled_search_spaces(args: argparse.Namespace) -> Dict[str, object]:
    from agent.run_search import default_search_spaces

    spaces = default_search_spaces()
    enabled_attacks = {attack.strip() for attack in args.attacks.split(",") if attack.strip()}
    spaces = {name: space for name, space in spaces.items() if name in enabled_attacks}
    if len(spaces) == 0:
        raise ValueError("No valid attack search spaces remain after filtering --attacks.")
    return spaces


def _seed_memory_store(
    source_path: str,
    destination_path: str,
    *,
    excluded_checkpoints: Iterable[str],
) -> Dict[str, object]:
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    source_store = ExperienceMemoryStore(source_path)
    retained_entries = []
    excluded_lookup = {str(Path(checkpoint)) for checkpoint in excluded_checkpoints}
    skipped = 0
    for entry in source_store.entries():
        checkpoint = str(Path(entry.task_profile.checkpoint_path))
        if checkpoint in excluded_lookup:
            skipped += 1
            continue
        retained_entries.append(entry)

    destination_store = ExperienceMemoryStore(str(destination))
    destination_store.extend(retained_entries)
    return {
        "source_path": str(Path(source_path)),
        "destination_path": str(destination),
        "seeded_entries": len(retained_entries),
        "skipped_entries": skipped,
    }


def _compute_shared_task_context(
    tasks: Iterable[TaskSpec],
    search_config: SearchConfig,
) -> Dict[str, Dict[str, object]]:
    from autoattack.runtime import DreamerV3SearchExecutor

    executor = DreamerV3SearchExecutor(search_config)
    shared_context: Dict[str, Dict[str, object]] = {}
    for task in tasks:
        baseline_config = TrialConfig(
            task_name=task.name,
            checkpoint_path=task.checkpoint_path,
            attack_name="baseline",
            epsilon=0.0,
            steps=0,
            seed=search_config.seed,
        )
        baseline_result = executor.run_trial(
            baseline_config,
            stage="confirm",
            num_episodes=search_config.confirm_episodes,
            persist_artifacts=True,
        )
        task_profile = executor.describe_task(task, baseline_result)
        shared_context[task.checkpoint_path] = {
            "baseline_result": baseline_result,
            "task_profile": task_profile,
        }
    return shared_context


def _build_method_summary(
    controller: "DebateSearchController",
    search_config: SearchConfig,
    task_summaries: List[Dict[str, object]],
    method: ComparisonMethod,
    memory_seed: Mapping[str, object],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "comparison_role": method.description,
        "search_mode": search_config.search_mode,
        "initialization_mode": search_config.initialization_mode,
        "agent_backend": search_config.agent_backend,
        "agent_model": search_config.agent_model,
        "experience_store_path": search_config.experience_store_path,
        "experience_retrieval_mode": search_config.experience_retrieval_mode,
        "memory_seed": dict(memory_seed),
        "tasks": task_summaries,
    }
    summary_path = Path(search_config.output_dir) / "search_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path = Path(search_config.output_dir) / "search_summary.txt"
    txt_path.write_text(controller._to_text(summary), encoding="utf-8")
    return summary


def _best_trial_row(
    baseline: Dict[str, object],
    attack_summary: Mapping[str, object],
) -> Dict[str, object]:
    best_trial = attack_summary.get("best_trial")
    if best_trial is None:
        return {"best_trial": None, "reward_drop": None}
    return {
        "best_trial": best_trial,
        "reward_drop": _normalized_reward_drop_obj(baseline, best_trial),
    }


def _build_comparison_summary(
    *,
    root_output_dir: Path,
    tasks: Iterable[TaskSpec],
    search_spaces: Mapping[str, object],
    shared_context: Mapping[str, Dict[str, object]],
    method_summaries: Mapping[str, Dict[str, object]],
    methods: Iterable[ComparisonMethod],
) -> Dict[str, object]:
    task_lookup: Dict[str, Dict[str, Dict[str, object]]] = {}
    for method_name, summary in method_summaries.items():
        per_task: Dict[str, Dict[str, object]] = {}
        for task_summary in summary["tasks"]:
            per_task[str(task_summary["checkpoint_path"])] = task_summary
        task_lookup[method_name] = per_task

    comparison_tasks: List[Dict[str, object]] = []
    for task in tasks:
        checkpoint_key = str(task.checkpoint_path)
        baseline_result = shared_context[checkpoint_key]["baseline_result"]
        baseline_dict = baseline_result.to_dict()
        attacks: Dict[str, object] = {}
        for attack_name in search_spaces:
            methods_for_attack: Dict[str, object] = {}
            winner = None
            winner_drop = None
            for method in methods:
                task_summary = task_lookup[method.name][checkpoint_key]
                attack_summary = task_summary["attacks"].get(attack_name, {})
                row = _best_trial_row(baseline_dict, attack_summary)
                methods_for_attack[method.name] = row
                reward_drop = row["reward_drop"]
                if reward_drop is None:
                    continue
                if winner_drop is None or reward_drop > winner_drop:
                    winner_drop = reward_drop
                    winner = method.name
            attacks[attack_name] = {
                "winner": winner,
                "winner_reward_drop": winner_drop,
                "methods": methods_for_attack,
            }
        comparison_tasks.append(
            {
                "task": task.name,
                "checkpoint_path": task.checkpoint_path,
                "baseline": baseline_dict,
                "attacks": attacks,
            }
        )

    summary = {
        "comparison_type": "claudini_style_vs_latent_rag",
        "output_dir": str(root_output_dir),
        "methods": [
            {
                "name": method.name,
                "description": method.description,
                "experience_retrieval_mode": method.retrieval_mode,
                "summary_path": str(root_output_dir / method.name / "search_summary.json"),
            }
            for method in methods
        ],
        "tasks": comparison_tasks,
    }
    summary_path = root_output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path = root_output_dir / "comparison_summary.txt"
    text_path.write_text(_comparison_to_text(summary), encoding="utf-8")
    return summary


def _comparison_to_text(summary: Mapping[str, object]) -> str:
    lines: List[str] = []
    lines.append("DreamerV3 LLM Attack Search Comparison")
    lines.append("Output dir: {}".format(summary.get("output_dir", "")))
    lines.append("")
    for method in summary.get("methods", []):
        lines.append(
            "Method: {} | retrieval={} | summary={}".format(
                method.get("name", ""),
                method.get("experience_retrieval_mode", ""),
                method.get("summary_path", ""),
            )
        )
    lines.append("")
    for task in summary.get("tasks", []):
        baseline = task["baseline"]
        lines.append("Task: {}".format(task["task"]))
        lines.append(
            "  Shared baseline: mean={:.4f}, std={:.4f}, time={:.2f}s".format(
                float(baseline["mean_reward"]),
                float(baseline["std_reward"]),
                float(baseline["elapsed_seconds"]),
            )
        )
        for attack_name, attack_row in task["attacks"].items():
            winner = attack_row.get("winner")
            winner_drop = attack_row.get("winner_reward_drop")
            lines.append(
                "  {}: winner={} reward_drop={}".format(
                    attack_name,
                    winner or "none",
                    "n/a" if winner_drop is None else "{:.4f}".format(float(winner_drop)),
                )
            )
            for method_name, method_row in attack_row["methods"].items():
                best_trial = method_row.get("best_trial")
                reward_drop = method_row.get("reward_drop")
                if best_trial is None:
                    lines.append("    {}: no confirmed trial".format(method_name))
                    continue
                config = best_trial["config"]
                lines.append(
                    "    {}: eps={}, steps={}, allocation={}, reward_drop={:.4f}, mean={:.4f}, time={:.2f}s".format(
                        method_name,
                        config["epsilon"],
                        config["steps"],
                        config["allocation"]["mode"],
                        float(reward_drop),
                        float(best_trial["mean_reward"]),
                        float(best_trial["elapsed_seconds"]),
                    )
                )
        lines.append("")
    return "\n".join(lines)


def _normalized_reward_drop_obj(baseline: Dict[str, object], trial: Dict[str, object]) -> float:
    scale = abs(float(baseline["mean_reward"])) + 1.0
    return (float(baseline["mean_reward"]) - float(trial["mean_reward"])) / scale


def main() -> None:
    from agent.orchestration import DebateSearchController
    from agent.run_search import build_agents, discover_tasks
    from autoattack.runtime import DreamerV3SearchExecutor

    parser = build_parser()
    args = parser.parse_args()

    root_output_dir = Path(_default_output_dir(args.output_dir)).resolve()
    root_output_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(args)
    search_spaces = _enabled_search_spaces(args)

    methods = [
        ComparisonMethod(
            name=args.baseline_method_name,
            retrieval_mode=args.baseline_retrieval_mode,
            description="Generic LLM self-evolution baseline inspired by Claudini without task-specific latent retrieval.",
        ),
        ComparisonMethod(
            name=args.ours_method_name,
            retrieval_mode=args.ours_retrieval_mode,
            description="Latent-RAG + self-evolution method under the same search space and budget.",
        ),
    ]

    shared_config = _build_search_config(
        args,
        output_dir=str(root_output_dir / "_shared_baseline"),
        experience_store_path=str(root_output_dir / "_shared_baseline" / "experience_store.jsonl"),
        experience_retrieval_mode=args.shared_baseline_retrieval_mode,
    )
    shared_context = _compute_shared_task_context(tasks, shared_config)

    excluded_checkpoints = () if args.allow_same_checkpoint_memory else [task.checkpoint_path for task in tasks]
    method_summaries: Dict[str, Dict[str, object]] = {}
    for method in methods:
        method_output_dir = root_output_dir / method.name
        method_memory_path = method_output_dir / "experience_store.jsonl"
        memory_seed = _seed_memory_store(
            args.experience_store_path,
            str(method_memory_path),
            excluded_checkpoints=excluded_checkpoints,
        )
        method_config = _build_search_config(
            args,
            output_dir=str(method_output_dir),
            experience_store_path=str(method_memory_path),
            experience_retrieval_mode=method.retrieval_mode,
        )
        executor = DreamerV3SearchExecutor(method_config)
        attacker_agent, auditor_agent = build_agents(method_config)
        controller = DebateSearchController(
            executor=executor,
            attacker_agent=attacker_agent,
            auditor_agent=auditor_agent,
            search_config=method_config,
        )
        task_summaries = []
        for task in tasks:
            shared = shared_context[task.checkpoint_path]
            task_summaries.append(
                controller.run_task(
                    task,
                    search_spaces,
                    baseline_result=shared["baseline_result"],
                    task_profile=shared["task_profile"],
                )
            )
        method_summaries[method.name] = _build_method_summary(
            controller,
            method_config,
            task_summaries,
            method,
            memory_seed,
        )

    _build_comparison_summary(
        root_output_dir=root_output_dir,
        tasks=tasks,
        search_spaces=search_spaces,
        shared_context=shared_context,
        method_summaries=method_summaries,
        methods=methods,
    )


if __name__ == "__main__":
    main()
