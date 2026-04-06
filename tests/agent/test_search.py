import json
from pathlib import Path

from agent.llm import (
    AttackSearchState,
    HeuristicAttackerAgent,
    HeuristicAuditorAgent,
    OpenAIAttackerAgent,
    OpenAIAuditorAgent,
    build_reflection_note,
    normalized_reward_drop,
)
from agent.memory import ExperienceMemoryStore, tokenize_task_name
from agent.schema import (
    AttackSearchSpace,
    ProbeRepresentation,
    TaskProfile,
    StepAllocationConfig,
    TaskSpec,
    TrialConfig,
    TrialResult,
)


def _result(task: str, attack: str, epsilon: float, steps: int, mean_reward: float, elapsed: float, flip_rate: float):
    return TrialResult(
        config=TrialConfig(
            task_name=task,
            checkpoint_path="/tmp/ckpt.ckpt",
            attack_name=attack,
            epsilon=epsilon,
            steps=steps,
            allocation=StepAllocationConfig(mode="fixed"),
        ),
        stage="confirm",
        num_episodes=3,
        mean_reward=mean_reward,
        std_reward=1.0,
        median_reward=mean_reward,
        min_reward=mean_reward,
        max_reward=mean_reward,
        elapsed_seconds=elapsed,
        returns=[mean_reward] * 3,
        telemetry={
            "decisions": 10,
            "flips": int(flip_rate * 10),
            "flip_rate": flip_rate,
            "clean_margin_mean": 1.2,
            "adv_margin_mean": 0.4,
            "allocated_steps_mean": steps,
            "allocated_epsilon_mean": epsilon,
        },
        artifact_dir="/tmp",
        probe_representation=ProbeRepresentation(
            source_stage="confirm",
            teacher_vector=(mean_reward / 10.0, elapsed / 10.0, flip_rate, epsilon, steps),
            feature_stats={"extras": {"flip_rate": flip_rate}},
            num_samples=4,
        ),
    )


class _FakeResponse:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class _FakeResponsesEndpoint:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._payloads:
            raise RuntimeError("No fake payload left for this test.")
        return _FakeResponse(self._payloads.pop(0))


class _FakeClient:
    def __init__(self, payloads):
        self.responses = _FakeResponsesEndpoint(payloads)


def test_reward_drop_positive_when_attack_hurts_performance():
    baseline = _result("pong", "baseline", 0.0, 0, 20.0, 1.0, 0.0)
    attack = _result("pong", "apgd_ce", 8.0, 8, -20.0, 10.0, 0.8)
    assert normalized_reward_drop(baseline, attack) > 1.5


def test_attacker_agent_proposes_unseen_candidates():
    baseline = _result("pong", "baseline", 0.0, 0, 20.0, 1.0, 0.0)
    state = AttackSearchState(
        task=TaskSpec(name="pong", checkpoint_path="/tmp/ckpt.ckpt"),
        search_space=AttackSearchSpace(
            attack_name="apgd_ce",
            epsilons=(4, 8),
            step_candidates=(4, 8),
            allocation_modes=("fixed",),
        ),
        baseline_result=baseline,
        runtime_budget_seconds=60.0,
    )
    agent = HeuristicAttackerAgent()
    first_batch = agent.propose(state, batch_size=2)
    second_batch = agent.propose(state, batch_size=2)
    first_keys = {cfg.key() for cfg in first_batch}
    second_keys = {cfg.key() for cfg in second_batch}
    assert len(first_batch) == 2
    assert first_keys.isdisjoint(second_keys)


def test_auditor_flags_runtime_and_no_flip_failures():
    baseline = _result("pong", "baseline", 0.0, 0, 20.0, 1.0, 0.0)
    state = AttackSearchState(
        task=TaskSpec(name="pong", checkpoint_path="/tmp/ckpt.ckpt"),
        search_space=AttackSearchSpace(
            attack_name="square",
            epsilons=(8,),
            step_candidates=(60,),
            allocation_modes=("fixed", "margin_linear"),
        ),
        baseline_result=baseline,
        runtime_budget_seconds=30.0,
    )
    weak_attack = _result("pong", "square", 8.0, 60, 19.5, 100.0, 0.0)
    audit = HeuristicAuditorAgent().audit(state, weak_attack)
    assert "runtime_over_budget" in audit.failure_tags
    assert "no_flip" in audit.failure_tags
    assert audit.recommendations["steps_bias"] < 0
    assert audit.strategy.search_action in ("expand", "shrink", "refine")


def test_reflexion_memory_drives_adaptive_candidates_beyond_base_grid():
    baseline = _result("pong", "baseline", 0.0, 0, 20.0, 1.0, 0.0)
    state = AttackSearchState(
        task=TaskSpec(name="pong", checkpoint_path="/tmp/ckpt.ckpt"),
        search_space=AttackSearchSpace(
            attack_name="apgd_ce",
            epsilons=(4, 8),
            step_candidates=(4, 8),
            allocation_modes=("fixed",),
        ),
        baseline_result=baseline,
        runtime_budget_seconds=60.0,
    )
    weak_attack = _result("pong", "apgd_ce", 8.0, 8, 19.5, 20.0, 0.0)
    audit = HeuristicAuditorAgent().audit(state, weak_attack)
    state.record_result(weak_attack)
    state.record_audit(audit)
    state.record_reflection(build_reflection_note(state.baseline_result, weak_attack, audit))
    proposals = HeuristicAttackerAgent().propose(state, batch_size=4)
    assert len(proposals) > 0
    assert any(proposal.epsilon > 8.0 or proposal.steps > 8 for proposal in proposals)


def test_experience_memory_retrieves_similar_task(tmp_path):
    store = ExperienceMemoryStore(str(tmp_path / "experience.jsonl"))
    profile = TaskProfile(
        task_name="PongNoFrameskip-v4",
        checkpoint_path="/tmp/pong.ckpt",
        env_id="PongNoFrameskip-v4",
        task_tokens=tuple(tokenize_task_name("PongNoFrameskip-v4")),
        baseline_clean_margin=2.0,
    )
    entry = store.build_entry(
        task_profile=profile,
        attack_name="apgd_ce",
        best_config={"epsilon": 6.0, "steps": 8, "allocation": {"mode": "margin_linear", "min_steps": 4}},
        result_summary={"mean_reward": -21.0},
        utility=1.8,
        source_run_dir="/tmp/run",
    )
    store.append(entry)
    retrieved = store.retrieve(profile, "apgd_ce", limit=2)
    assert len(retrieved) == 1
    assert retrieved[0].entry.best_config["epsilon"] == 6.0


def test_experience_memory_latent_retrieval_uses_probe_vectors(tmp_path):
    store = ExperienceMemoryStore(str(tmp_path / "experience_latent.jsonl"))
    profile = TaskProfile(
        task_name="PongNoFrameskip-v4",
        checkpoint_path="/tmp/pong.ckpt",
        env_id="PongNoFrameskip-v4",
        task_tokens=tuple(tokenize_task_name("PongNoFrameskip-v4")),
        baseline_clean_margin=2.0,
        probe_representation=ProbeRepresentation(
            source_stage="baseline",
            teacher_vector=(1.0, 0.5, -0.5, 0.2),
            feature_stats={"encoder": {"clean_mean_mean": 1.0}},
            num_samples=8,
        ),
    )
    close_entry = store.build_entry(
        task_profile=profile,
        attack_name="apgd_ce",
        best_config={"epsilon": 8.0, "steps": 10, "allocation": {"mode": "fixed"}},
        result_summary={"mean_reward": -21.0},
        utility=1.5,
        source_run_dir="/tmp/run_close",
    )
    far_profile = TaskProfile(
        task_name="AlienNoFrameskip-v4",
        checkpoint_path="/tmp/alien.ckpt",
        env_id="AlienNoFrameskip-v4",
        task_tokens=tuple(tokenize_task_name("AlienNoFrameskip-v4")),
        baseline_clean_margin=0.5,
        probe_representation=ProbeRepresentation(
            source_stage="baseline",
            teacher_vector=(-3.0, 1.0, 2.0, -4.0),
            feature_stats={"encoder": {"clean_mean_mean": -3.0}},
            num_samples=8,
        ),
    )
    far_entry = store.build_entry(
        task_profile=far_profile,
        attack_name="apgd_ce",
        best_config={"epsilon": 12.0, "steps": 12, "allocation": {"mode": "margin_linear", "min_steps": 6}},
        result_summary={"mean_reward": -10.0},
        utility=0.8,
        source_run_dir="/tmp/run_far",
    )
    store.extend([close_entry, far_entry])
    retrieved = store.retrieve(profile, "apgd_ce", limit=2, mode="latent", latent_projection="pca", latent_dim=2)
    assert len(retrieved) == 2
    assert retrieved[0].entry.best_config["epsilon"] == 8.0


def test_prior_experience_guides_initial_proposals(tmp_path):
    store = ExperienceMemoryStore(str(tmp_path / "experience.jsonl"))
    profile = TaskProfile(
        task_name="PongNoFrameskip-v4",
        checkpoint_path="/tmp/pong.ckpt",
        env_id="PongNoFrameskip-v4",
        task_tokens=tuple(tokenize_task_name("PongNoFrameskip-v4")),
        baseline_clean_margin=2.4,
    )
    store.append(
        store.build_entry(
            task_profile=profile,
            attack_name="apgd_ce",
            best_config={
                "epsilon": 10.0,
                "steps": 10,
                "restarts": 1,
                "rho": 0.75,
                "seed": 0,
                "allocation": {"mode": "margin_linear", "min_steps": 5},
            },
            result_summary={"mean_reward": -21.0},
            utility=1.9,
            source_run_dir="/tmp/run",
        )
    )
    baseline = _result("PongNoFrameskip-v4", "baseline", 0.0, 0, 20.0, 1.0, 0.0)
    state = AttackSearchState(
        task=TaskSpec(name="PongNoFrameskip-v4", checkpoint_path="/tmp/ckpt.ckpt"),
        search_space=AttackSearchSpace(
            attack_name="apgd_ce",
            epsilons=(4, 8),
            step_candidates=(4, 8),
            allocation_modes=("fixed", "margin_linear"),
        ),
        baseline_result=baseline,
        runtime_budget_seconds=60.0,
        task_profile=profile,
        prior_experiences=store.retrieve(profile, "apgd_ce", limit=2),
    )
    proposals = HeuristicAttackerAgent().propose(state, batch_size=4)
    assert len(proposals) > 0
    assert any(proposal.epsilon >= 10.0 or proposal.steps >= 10 for proposal in proposals)


def test_openai_attacker_agent_uses_structured_candidate_selection():
    baseline = _result("pong", "baseline", 0.0, 0, 20.0, 1.0, 0.0)
    state = AttackSearchState(
        task=TaskSpec(name="pong", checkpoint_path="/tmp/ckpt.ckpt"),
        search_space=AttackSearchSpace(
            attack_name="apgd_ce",
            epsilons=(4, 8),
            step_candidates=(4, 8),
            allocation_modes=("fixed",),
        ),
        baseline_result=baseline,
        runtime_budget_seconds=60.0,
    )
    client = _FakeClient(
        [
            {
                "candidate_ids": ["candidate_01"],
                "summary": "Pick the stronger midpoint candidate first.",
                "per_candidate_notes": [
                    {"candidate_id": "candidate_01", "note": "Best reward-drop/runtime tradeoff in the shortlist."}
                ],
            }
        ]
    )
    agent = OpenAIAttackerAgent(client=client, model="gpt-5-mini", max_candidates=6, context_limit=4)
    proposals = agent.propose(state, batch_size=1)
    assert len(proposals) == 1
    assert proposals[0].attack_name == "apgd_ce"
    assert proposals[0].epsilon in (4.0, 8.0)
    assert client.responses.calls[0]["model"] == "gpt-5-mini"


def test_openai_auditor_agent_parses_structured_audit():
    baseline = _result("pong", "baseline", 0.0, 0, 20.0, 1.0, 0.0)
    state = AttackSearchState(
        task=TaskSpec(name="pong", checkpoint_path="/tmp/ckpt.ckpt"),
        search_space=AttackSearchSpace(
            attack_name="apgd_dlr",
            epsilons=(8,),
            step_candidates=(8,),
            allocation_modes=("fixed",),
        ),
        baseline_result=baseline,
        runtime_budget_seconds=60.0,
    )
    weak_attack = _result("pong", "apgd_dlr", 8.0, 8, 18.5, 75.0, 0.05)
    client = _FakeClient(
        [
            {
                "failure_tags": ["runtime_over_budget", "no_flip"],
                "summary": "The trial is too slow and barely changes the action.",
                "root_cause": "insufficient_action_change",
                "recommendations": {
                    "epsilon_bias": 0.75,
                    "steps_bias": -0.5,
                    "prefer_allocation": "margin_linear",
                    "avoid_allocation": None,
                },
                "strategy": {
                    "search_action": "scout",
                    "epsilon_action": "local",
                    "steps_action": "decrease",
                    "allocation_action": "use_margin_linear",
                    "target_epsilon": 8.0,
                    "target_steps": 6,
                    "confidence": 0.85,
                },
            }
        ]
    )
    audit = OpenAIAuditorAgent(client=client, model="gpt-5-mini", context_limit=4).audit(state, weak_attack)
    assert "runtime_over_budget" in audit.failure_tags
    assert audit.recommendations["prefer_allocation"] == "margin_linear"
    assert audit.recommendations["steps_bias"] < 0
    assert audit.strategy.target_steps == 6
    assert audit.strategy.search_action == "refine"
    assert audit.strategy.steps_action == "down"
    assert audit.strategy.allocation_action == "prefer_margin_linear"
