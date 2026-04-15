from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ProbeRepresentation:
    version: str = "v1"
    source_stage: str = "baseline"
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    teacher_vector: Sequence[float] = field(default_factory=tuple)
    compression: str = "summary_stats"
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["teacher_vector"] = list(self.teacher_vector)
        return data

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["ProbeRepresentation"]:
        if not payload:
            return None
        feature_stats = {
            str(key): {str(inner_key): float(inner_value) for inner_key, inner_value in dict(values).items()}
            for key, values in dict(payload.get("feature_stats", {})).items()
        }
        teacher_vector = tuple(float(value) for value in payload.get("teacher_vector", ()) or ())
        return cls(
            version=str(payload.get("version", "v1")),
            source_stage=str(payload.get("source_stage", "baseline")),
            feature_stats=feature_stats,
            teacher_vector=teacher_vector,
            compression=str(payload.get("compression", "summary_stats")),
            num_samples=int(payload.get("num_samples", 0)),
        )


@dataclass(frozen=True)
class StepAllocationConfig:
    mode: str = "fixed"
    min_steps: Optional[int] = None
    margin_low: float = 0.1
    margin_high: float = 1.5
    epsilon_scale_low: float = 1.0
    epsilon_scale_high: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrialConfig:
    task_name: str
    checkpoint_path: str
    attack_name: str
    epsilon: float
    steps: int
    restarts: int = 1
    rho: float = 0.75
    seed: int = 0
    allocation: StepAllocationConfig = field(default_factory=StepAllocationConfig)

    @property
    def is_baseline(self) -> bool:
        return self.attack_name == "baseline"

    def key(self) -> str:
        alloc = self.allocation
        return (
            f"{self.task_name}|{self.attack_name}|eps={self.epsilon}|steps={self.steps}|"
            f"restarts={self.restarts}|rho={self.rho}|seed={self.seed}|"
            f"alloc={alloc.mode}|min={alloc.min_steps}|ml={alloc.margin_low}|mh={alloc.margin_high}|"
            f"esl={alloc.epsilon_scale_low}|esh={alloc.epsilon_scale_high}"
        )

    def short_name(self) -> str:
        if self.is_baseline:
            return "baseline"
        alloc = self.allocation.mode
        alloc_suffix = "" if alloc == "fixed" else f"_{alloc}"
        min_steps = "" if self.allocation.min_steps is None else f"_min{self.allocation.min_steps}"
        return f"{self.attack_name}_e{self.epsilon:g}_s{self.steps}{alloc_suffix}{min_steps}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["allocation"] = self.allocation.to_dict()
        return data


@dataclass
class TrialTelemetry:
    decisions: int = 0
    flips: int = 0
    clean_margin_sum: float = 0.0
    adv_margin_sum: float = 0.0
    allocated_steps_sum: float = 0.0
    allocated_epsilon_sum: float = 0.0

    def update(self, clean_margin: float, adv_margin: float, flipped: bool, steps: int, epsilon: float) -> None:
        self.decisions += 1
        self.flips += int(flipped)
        self.clean_margin_sum += clean_margin
        self.adv_margin_sum += adv_margin
        self.allocated_steps_sum += steps
        self.allocated_epsilon_sum += epsilon

    def to_dict(self) -> Dict[str, Any]:
        decisions = max(self.decisions, 1)
        return {
            "decisions": self.decisions,
            "flips": self.flips,
            "flip_rate": self.flips / decisions,
            "clean_margin_mean": self.clean_margin_sum / decisions,
            "adv_margin_mean": self.adv_margin_sum / decisions,
            "allocated_steps_mean": self.allocated_steps_sum / decisions,
            "allocated_epsilon_mean": self.allocated_epsilon_sum / decisions,
        }


@dataclass
class TrialResult:
    config: TrialConfig
    stage: str
    num_episodes: int
    mean_reward: float
    std_reward: float
    median_reward: float
    min_reward: float
    max_reward: float
    elapsed_seconds: float
    returns: List[float]
    telemetry: Dict[str, Any]
    artifact_dir: str
    notes: List[str] = field(default_factory=list)
    probe_representation: Optional[ProbeRepresentation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "stage": self.stage,
            "num_episodes": self.num_episodes,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "median_reward": self.median_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "elapsed_seconds": self.elapsed_seconds,
            "returns": self.returns,
            "telemetry": self.telemetry,
            "artifact_dir": self.artifact_dir,
            "notes": self.notes,
            "probe_representation": (
                None if self.probe_representation is None else self.probe_representation.to_dict()
            ),
        }


@dataclass
class ReflectionStrategy:
    search_action: str = "keep"
    epsilon_action: str = "keep"
    steps_action: str = "keep"
    allocation_action: str = "keep"
    target_epsilon: Optional[float] = None
    target_steps: Optional[int] = None
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    trial_key: str
    failure_tags: List[str]
    summary: str
    root_cause: str
    recommendations: Dict[str, Any]
    strategy: ReflectionStrategy = field(default_factory=ReflectionStrategy)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["strategy"] = self.strategy.to_dict()
        return data


@dataclass
class ReflectionNote:
    trial_key: str
    attack_name: str
    stage: str
    summary: str
    root_cause: str
    failure_tags: List[str]
    reward_drop: float
    flip_rate: float
    clean_margin_mean: float
    adv_margin_mean: float
    elapsed_seconds: float
    strategy: ReflectionStrategy

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["strategy"] = self.strategy.to_dict()
        return data


@dataclass(frozen=True)
class TaskProfile:
    task_name: str
    checkpoint_path: str
    algo_name: str = "dreamer_v3"
    env_id: str = ""
    run_name: str = ""
    action_type: str = "discrete"
    cnn_keys: Sequence[str] = field(default_factory=tuple)
    baseline_mean_reward: Optional[float] = None
    baseline_std_reward: Optional[float] = None
    baseline_clean_margin: Optional[float] = None
    task_tokens: Sequence[str] = field(default_factory=tuple)
    probe_representation: Optional[ProbeRepresentation] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["probe_representation"] = (
            None if self.probe_representation is None else self.probe_representation.to_dict()
        )
        return data

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaskProfile":
        return cls(
            task_name=str(payload.get("task_name", "")),
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            algo_name=str(payload.get("algo_name", "dreamer_v3")),
            env_id=str(payload.get("env_id", "")),
            run_name=str(payload.get("run_name", "")),
            action_type=str(payload.get("action_type", "discrete")),
            cnn_keys=tuple(payload.get("cnn_keys", ()) or ()),
            baseline_mean_reward=payload.get("baseline_mean_reward"),
            baseline_std_reward=payload.get("baseline_std_reward"),
            baseline_clean_margin=payload.get("baseline_clean_margin"),
            task_tokens=tuple(payload.get("task_tokens", ()) or ()),
            probe_representation=ProbeRepresentation.from_dict(payload.get("probe_representation")),
        )


@dataclass
class ExperienceEntry:
    task_profile: TaskProfile
    attack_name: str
    best_config: Dict[str, Any]
    result_summary: Dict[str, Any]
    utility: float
    source_run_dir: str
    created_at: str
    notes: List[str] = field(default_factory=list)
    probe_representation: Optional[ProbeRepresentation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_profile": self.task_profile.to_dict(),
            "attack_name": self.attack_name,
            "best_config": self.best_config,
            "result_summary": self.result_summary,
            "utility": self.utility,
            "source_run_dir": self.source_run_dir,
            "created_at": self.created_at,
            "notes": self.notes,
            "probe_representation": (
                None if self.probe_representation is None else self.probe_representation.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperienceEntry":
        return cls(
            task_profile=TaskProfile.from_dict(dict(payload.get("task_profile", {}))),
            attack_name=str(payload.get("attack_name", "")),
            best_config=dict(payload.get("best_config", {})),
            result_summary=dict(payload.get("result_summary", {})),
            utility=float(payload.get("utility", 0.0)),
            source_run_dir=str(payload.get("source_run_dir", "")),
            created_at=str(payload.get("created_at", "")),
            notes=[str(item) for item in payload.get("notes", [])],
            probe_representation=ProbeRepresentation.from_dict(payload.get("probe_representation")),
        )


@dataclass(frozen=True)
class TaskSpec:
    name: str
    checkpoint_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttackSearchSpace:
    attack_name: str
    epsilons: Sequence[float]
    step_candidates: Sequence[int]
    restarts: Sequence[int] = (1,)
    rhos: Sequence[float] = (0.75,)
    allocation_modes: Sequence[str] = ("fixed", "margin_linear")
    min_step_fractions: Sequence[float] = (0.25, 0.5)
    margin_low: float = 0.1
    margin_high: float = 1.5

    def candidates(self, task: TaskSpec, seed: int = 0) -> List[TrialConfig]:
        configs: List[TrialConfig] = []
        for epsilon in self.epsilons:
            for steps in self.step_candidates:
                for restart in self.restarts:
                    for rho in self.rhos:
                        for mode in self.allocation_modes:
                            if mode == "fixed":
                                configs.append(
                                    TrialConfig(
                                        task_name=task.name,
                                        checkpoint_path=task.checkpoint_path,
                                        attack_name=self.attack_name,
                                        epsilon=float(epsilon),
                                        steps=int(steps),
                                        restarts=int(restart),
                                        rho=float(rho),
                                        seed=seed,
                                        allocation=StepAllocationConfig(mode="fixed"),
                                    )
                                )
                            else:
                                for fraction in self.min_step_fractions:
                                    min_steps = max(1, int(round(float(steps) * float(fraction))))
                                    configs.append(
                                        TrialConfig(
                                            task_name=task.name,
                                            checkpoint_path=task.checkpoint_path,
                                            attack_name=self.attack_name,
                                            epsilon=float(epsilon),
                                            steps=int(steps),
                                            restarts=int(restart),
                                            rho=float(rho),
                                            seed=seed,
                                            allocation=StepAllocationConfig(
                                                mode=mode,
                                                min_steps=min_steps,
                                                margin_low=self.margin_low,
                                                margin_high=self.margin_high,
                                            ),
                                        )
                                    )
        return configs


@dataclass
class SearchConfig:
    output_dir: str
    search_mode: str = "reflexion"
    initialization_mode: str = "task_conditioned"
    scout_episodes: int = 3
    confirm_episodes: int = 10
    proposal_batch_size: int = 4
    confirm_top_k: int = 2
    max_trials_per_attack: int = 8
    runtime_budget_seconds: float = 600.0
    capture_video_for_confirm: bool = True
    tensorboard_for_confirm: bool = True
    accelerator: str = "cuda"
    devices: int = 1
    precision: str = "32-true"
    seed: int = 0
    agent_backend: str = "transformers"
    agent_model: str = "Qwen/Qwen2.5-7B-Instruct"
    agent_api_key_env: str = "OPENAI_API_KEY"
    agent_base_url: str = ""
    agent_max_candidates: int = 24
    agent_context_limit: int = 8
    agent_max_new_tokens: int = 384
    agent_temperature: float = 0.0
    agent_cache_dir: str = ""
    reflection_history_limit: int = 8
    experience_store_path: str = "logs/agent_experience.jsonl"
    experience_retrieval_limit: int = 6
    experience_retrieval_mode: str = "structured"
    experience_latent_projection: str = "pca"
    experience_latent_dim: int = 16
    experience_hybrid_weight: float = 0.6
    experience_probe_max_steps: int = 32

    def ensure_output_dir(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
