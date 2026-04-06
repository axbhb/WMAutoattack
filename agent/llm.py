from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Union

from agent.memory import RetrievedExperience
from agent.schema import (
    AttackSearchSpace,
    AuditReport,
    ExperienceEntry,
    ReflectionNote,
    ReflectionStrategy,
    StepAllocationConfig,
    TaskSpec,
    TaskProfile,
    TrialConfig,
    TrialResult,
)


def normalized_reward_drop(baseline: TrialResult, trial: TrialResult) -> float:
    scale = abs(baseline.mean_reward) + 1.0
    return (baseline.mean_reward - trial.mean_reward) / scale


def scalarized_utility(
    baseline: TrialResult,
    trial: TrialResult,
    runtime_weight: float = 0.15,
    variance_weight: float = 0.05,
    flip_weight: float = 0.25,
) -> float:
    drop = normalized_reward_drop(baseline, trial)
    telemetry = trial.telemetry or {}
    flip_rate = float(telemetry.get("flip_rate", 0.0))
    variability = trial.std_reward / (abs(baseline.mean_reward) + 1.0)
    runtime_penalty = math.log1p(max(trial.elapsed_seconds, 0.0))
    return drop + flip_weight * flip_rate - runtime_weight * runtime_penalty - variance_weight * variability


def pareto_front(baseline: TrialResult, trials: Sequence[TrialResult]) -> List[TrialResult]:
    candidates = [trial for trial in trials if not trial.config.is_baseline]
    front = []
    for candidate in candidates:
        dominated = False
        candidate_drop = normalized_reward_drop(baseline, candidate)
        for other in candidates:
            if other is candidate:
                continue
            other_drop = normalized_reward_drop(baseline, other)
            if (
                other_drop >= candidate_drop
                and other.elapsed_seconds <= candidate.elapsed_seconds
                and (other_drop > candidate_drop or other.elapsed_seconds < candidate.elapsed_seconds)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    front.sort(key=lambda trial: (-normalized_reward_drop(baseline, trial), trial.elapsed_seconds))
    return front


class AttackerAgent(Protocol):
    def propose(self, state: "AttackSearchState", batch_size: int) -> List[TrialConfig]:
        ...


class AuditorAgent(Protocol):
    def audit(self, state: "AttackSearchState", result: TrialResult) -> AuditReport:
        ...


@dataclass
class AttackSearchState:
    task: TaskSpec
    search_space: AttackSearchSpace
    baseline_result: TrialResult
    runtime_budget_seconds: float
    task_profile: Optional[TaskProfile] = None
    prior_experiences: List[RetrievedExperience] = field(default_factory=list)
    proposed_keys: set = field(default_factory=set)
    scout_results: Dict[str, TrialResult] = field(default_factory=dict)
    confirmed_results: Dict[str, TrialResult] = field(default_factory=dict)
    audits: Dict[str, AuditReport] = field(default_factory=dict)
    result_history: List[TrialResult] = field(default_factory=list)
    audit_history: List[AuditReport] = field(default_factory=list)
    reflections: List[ReflectionNote] = field(default_factory=list)

    def record_result(self, result: TrialResult) -> None:
        self.proposed_keys.add(result.config.key())
        if result.stage == "confirm":
            self.confirmed_results[result.config.key()] = result
        else:
            self.scout_results[result.config.key()] = result
        self.result_history.append(result)

    def record_audit(self, audit: AuditReport) -> None:
        self.audits[audit.trial_key] = audit
        self.audit_history.append(audit)

    def record_reflection(self, reflection: ReflectionNote) -> None:
        self.reflections.append(reflection)

    def all_results(self) -> List[TrialResult]:
        return list(self.result_history)

    def best_result(self) -> Optional[TrialResult]:
        candidates = list(self.confirmed_results.values()) or list(self.scout_results.values())
        if len(candidates) == 0:
            return None
        return max(candidates, key=lambda trial: scalarized_utility(self.baseline_result, trial))

    def recent_audits(self, limit: int = 3) -> List[AuditReport]:
        return self.audit_history[-limit:]

    def recent_results(self, limit: int = 6) -> List[TrialResult]:
        return self.result_history[-limit:]

    def recent_reflections(self, limit: int = 6) -> List[ReflectionNote]:
        return self.reflections[-limit:]

    def top_prior_experiences(self, limit: int = 4) -> List[RetrievedExperience]:
        return self.prior_experiences[:limit]


class HeuristicAttackerAgent:
    def propose(self, state: AttackSearchState, batch_size: int) -> List[TrialConfig]:
        base_candidates = [c for c in state.search_space.candidates(state.task) if c.key() not in state.proposed_keys]
        candidates = self._candidate_pool(state, base_candidates)
        if len(candidates) == 0:
            return []

        ranked = self._rank_candidates(state, candidates)
        proposals = ranked[:batch_size]
        for proposal in proposals:
            state.proposed_keys.add(proposal.key())
        return proposals

    def _candidate_pool(self, state: AttackSearchState, base_candidates: Sequence[TrialConfig]) -> List[TrialConfig]:
        adaptive_candidates = self._adaptive_variants(state)
        merged: List[TrialConfig] = []
        seen = set()
        for candidate in list(adaptive_candidates) + list(base_candidates):
            key = candidate.key()
            if key in seen or key in state.proposed_keys:
                continue
            merged.append(candidate)
            seen.add(key)
        return merged

    def _rank_candidates(self, state: AttackSearchState, candidates: Iterable[TrialConfig]) -> List[TrialConfig]:
        best = state.best_result()
        recommendations = [audit.recommendations for audit in state.recent_audits()]
        ranked = sorted(
            candidates,
            key=lambda candidate: self._candidate_score(state, candidate, best, recommendations),
            reverse=True,
        )
        return ranked

    def _candidate_score(
        self,
        state: AttackSearchState,
        candidate: TrialConfig,
        best: Optional[TrialResult],
        recommendations: Sequence[Dict[str, Any]],
    ) -> float:
        score = 0.0
        if best is None:
            score += self._initial_candidate_score(state, candidate)
        else:
            score -= abs(candidate.epsilon - best.config.epsilon)
            score -= abs(candidate.steps - best.config.steps) / 6.0
            score += 0.3 if candidate.allocation.mode == best.config.allocation.mode else 0.0
            if candidate.allocation.mode != "fixed":
                best_min = best.config.allocation.min_steps or max(1, best.config.steps // 2)
                candidate_min = candidate.allocation.min_steps or candidate.steps
                score -= abs(candidate_min - best_min) / 6.0

        for recommendation in recommendations:
            score += 0.5 * float(recommendation.get("epsilon_bias", 0.0)) * candidate.epsilon
            score += 0.1 * float(recommendation.get("steps_bias", 0.0)) * candidate.steps
            preferred = recommendation.get("prefer_allocation")
            if preferred and candidate.allocation.mode == preferred:
                score += 1.5
            blocked = recommendation.get("avoid_allocation")
            if blocked and candidate.allocation.mode == blocked:
                score -= 1.5

        if state.runtime_budget_seconds < 300 and candidate.allocation.mode == "margin_linear":
            score += 0.5
        return score

    def _initial_candidate_score(self, state: AttackSearchState, candidate: TrialConfig) -> float:
        score = 0.0
        score -= abs(candidate.epsilon - self._median_value(state.search_space.epsilons))
        score -= abs(candidate.steps - self._median_value(state.search_space.step_candidates)) / 10.0
        score += 0.25 if candidate.allocation.mode == "margin_linear" else 0.0
        target_epsilon, target_steps, prefer_margin = self._task_conditioned_defaults(state)
        score -= abs(candidate.epsilon - target_epsilon) * 0.75
        score -= abs(candidate.steps - target_steps) / 5.0
        if prefer_margin and candidate.allocation.mode == "margin_linear":
            score += 1.0
        for retrieved in state.top_prior_experiences(4):
            entry = retrieved.entry
            config = entry.best_config
            score += retrieved.score * 0.4
            score -= retrieved.score * 0.25 * abs(candidate.epsilon - float(config.get("epsilon", candidate.epsilon)))
            score -= retrieved.score * 0.05 * abs(candidate.steps - int(config.get("steps", candidate.steps)))
            alloc = str(config.get("allocation", {}).get("mode", "fixed"))
            if alloc == candidate.allocation.mode:
                score += 0.6 * retrieved.score
        return score

    def _median_value(self, values: Sequence[Union[float, int]]) -> float:
        ordered = sorted(float(v) for v in values)
        return ordered[len(ordered) // 2]

    def _adaptive_variants(self, state: AttackSearchState) -> List[TrialConfig]:
        best = state.best_result()
        if best is None and len(state.reflections) == 0 and len(state.prior_experiences) == 0:
            return self._task_conditioned_initials(state)

        anchors: List[TrialConfig] = []
        if best is not None:
            anchors.append(best.config)
        for retrieved in state.top_prior_experiences(4):
            candidate = self._candidate_from_experience(state, retrieved.entry)
            if candidate is not None:
                anchors.append(candidate)
        for reflection in reversed(state.recent_reflections(4)):
            matched = state.confirmed_results.get(reflection.trial_key) or state.scout_results.get(reflection.trial_key)
            if matched is not None:
                anchors.append(matched.config)

        candidates: List[TrialConfig] = []
        for anchor in anchors:
            candidates.extend(self._local_neighbors(state, anchor))

        for reflection in state.recent_reflections(4):
            anchor = self._find_reflection_anchor(state, reflection)
            if anchor is None:
                continue
            candidates.extend(self._strategy_variants(state, anchor, reflection.strategy))
        return candidates

    def _candidate_from_experience(self, state: AttackSearchState, entry: ExperienceEntry) -> Optional[TrialConfig]:
        config = entry.best_config
        allocation = dict(config.get("allocation", {}))
        epsilon = self._clip_epsilon(
            state, float(config.get("epsilon", self._median_value(state.search_space.epsilons)))
        )
        steps = self._clip_steps(
            state, int(config.get("steps", self._median_value(state.search_space.step_candidates)))
        )
        if epsilon is None or steps is None:
            return None
        allocation_mode = str(allocation.get("mode", "fixed"))
        min_steps = allocation.get("min_steps")
        if allocation_mode != "fixed" and min_steps is not None:
            min_steps = max(1, min(steps, int(min_steps)))
        return TrialConfig(
            task_name=state.task.name,
            checkpoint_path=state.task.checkpoint_path,
            attack_name=state.search_space.attack_name,
            epsilon=epsilon,
            steps=steps,
            restarts=int(config.get("restarts", 1)),
            rho=float(config.get("rho", 0.75)),
            seed=int(config.get("seed", 0)),
            allocation=StepAllocationConfig(
                mode=allocation_mode,
                min_steps=None if allocation_mode == "fixed" else min_steps,
                margin_low=float(allocation.get("margin_low", state.search_space.margin_low)),
                margin_high=float(allocation.get("margin_high", state.search_space.margin_high)),
                epsilon_scale_low=float(allocation.get("epsilon_scale_low", 1.0)),
                epsilon_scale_high=float(allocation.get("epsilon_scale_high", 1.0)),
            ),
        )

    def _task_conditioned_initials(self, state: AttackSearchState) -> List[TrialConfig]:
        target_epsilon, target_steps, prefer_margin = self._task_conditioned_defaults(state)
        variants: List[TrialConfig] = []
        step_delta = self._step_delta(state)
        epsilon_delta = self._epsilon_delta(state)
        modes = ["margin_linear", "fixed"] if prefer_margin else ["fixed", "margin_linear"]
        for epsilon in (target_epsilon - epsilon_delta, target_epsilon, target_epsilon + epsilon_delta):
            for steps in (target_steps, target_steps + step_delta):
                for mode in modes:
                    min_steps = None if mode == "fixed" else max(1, int(round(steps * 0.5)))
                    candidate = self._build_candidate(state, epsilon, steps, mode, min_steps)
                    if candidate is not None:
                        variants.append(candidate)
        return variants

    def _task_conditioned_defaults(self, state: AttackSearchState) -> tuple[float, int, bool]:
        base_epsilon = self._median_value(state.search_space.epsilons)
        base_steps = int(round(self._median_value(state.search_space.step_candidates)))
        prefer_margin = True
        profile = state.task_profile
        clean_margin = 0.0 if profile is None or profile.baseline_clean_margin is None else profile.baseline_clean_margin
        if clean_margin >= 2.5:
            base_epsilon += self._epsilon_delta(state)
            base_steps += self._step_delta(state)
        elif 0.0 < clean_margin <= 1.0:
            base_epsilon -= self._epsilon_delta(state)
        if state.search_space.attack_name == "fab":
            base_steps = max(base_steps, int(round(self._median_value(state.search_space.step_candidates) + self._step_delta(state))))
        if state.search_space.attack_name == "square":
            base_steps = max(base_steps, int(round(self._median_value(state.search_space.step_candidates) + 2 * self._step_delta(state))))
        if profile is not None and profile.action_type == "continuous":
            base_steps += self._step_delta(state)
        epsilon = self._clip_epsilon(state, base_epsilon) or base_epsilon
        steps = self._clip_steps(state, base_steps) or base_steps
        return float(epsilon), int(steps), prefer_margin

    def _build_candidate(
        self,
        state: AttackSearchState,
        epsilon: float,
        steps: int,
        allocation_mode: str,
        min_steps: Optional[int],
    ) -> Optional[TrialConfig]:
        epsilon_value = self._clip_epsilon(state, epsilon)
        step_value = self._clip_steps(state, steps)
        if epsilon_value is None or step_value is None:
            return None
        if allocation_mode == "fixed":
            allocation = StepAllocationConfig(mode="fixed")
        else:
            allocation = StepAllocationConfig(
                mode="margin_linear",
                min_steps=max(1, min(step_value, min_steps if min_steps is not None else max(1, step_value // 2))),
                margin_low=state.search_space.margin_low,
                margin_high=state.search_space.margin_high,
                epsilon_scale_low=1.0,
                epsilon_scale_high=1.0,
            )
        return TrialConfig(
            task_name=state.task.name,
            checkpoint_path=state.task.checkpoint_path,
            attack_name=state.search_space.attack_name,
            epsilon=epsilon_value,
            steps=step_value,
            allocation=allocation,
        )

    def _find_reflection_anchor(self, state: AttackSearchState, reflection: ReflectionNote) -> Optional[TrialConfig]:
        result = state.confirmed_results.get(reflection.trial_key) or state.scout_results.get(reflection.trial_key)
        return None if result is None else result.config

    def _local_neighbors(self, state: AttackSearchState, anchor: TrialConfig) -> List[TrialConfig]:
        epsilon_delta = self._epsilon_delta(state)
        step_delta = self._step_delta(state)
        epsilon_values = [anchor.epsilon - epsilon_delta, anchor.epsilon, anchor.epsilon + epsilon_delta]
        step_values = [anchor.steps - step_delta, anchor.steps, anchor.steps + step_delta]
        variants: List[TrialConfig] = []
        for epsilon in epsilon_values:
            for steps in step_values:
                candidate = self._clone_candidate(state, anchor, epsilon=epsilon, steps=steps)
                if candidate is not None:
                    variants.append(candidate)
        return variants

    def _strategy_variants(
        self,
        state: AttackSearchState,
        anchor: TrialConfig,
        strategy: ReflectionStrategy,
    ) -> List[TrialConfig]:
        epsilon_values = self._epsilon_candidates_from_strategy(state, anchor, strategy)
        step_values = self._step_candidates_from_strategy(state, anchor, strategy)
        allocation_modes = self._allocation_candidates_from_strategy(anchor, strategy)
        variants: List[TrialConfig] = []
        for epsilon in epsilon_values:
            for steps in step_values:
                for allocation_mode in allocation_modes:
                    min_steps = anchor.allocation.min_steps
                    if allocation_mode == "margin_linear":
                        min_steps = max(1, int(round(steps * 0.25))) if min_steps is None else min(min_steps, steps)
                    else:
                        min_steps = None
                    candidate = self._clone_candidate(
                        state,
                        anchor,
                        epsilon=epsilon,
                        steps=steps,
                        allocation_mode=allocation_mode,
                        min_steps=min_steps,
                    )
                    if candidate is not None:
                        variants.append(candidate)
        return variants

    def _epsilon_candidates_from_strategy(
        self,
        state: AttackSearchState,
        anchor: TrialConfig,
        strategy: ReflectionStrategy,
    ) -> List[float]:
        delta = self._epsilon_delta(state)
        values = [anchor.epsilon]
        if strategy.target_epsilon is not None:
            values.append(strategy.target_epsilon)
        action = strategy.epsilon_action
        if action == "up":
            values.extend([anchor.epsilon + delta, anchor.epsilon * 1.5])
        elif action == "down":
            values.extend([anchor.epsilon - delta, anchor.epsilon * 0.75])
        elif action == "local":
            values.extend([anchor.epsilon - delta, anchor.epsilon + delta])
        elif strategy.search_action == "expand":
            values.extend([anchor.epsilon + delta, anchor.epsilon * 1.5])
        elif strategy.search_action == "shrink":
            values.extend([anchor.epsilon - delta])
        return self._normalize_epsilon_values(state, values)

    def _step_candidates_from_strategy(
        self,
        state: AttackSearchState,
        anchor: TrialConfig,
        strategy: ReflectionStrategy,
    ) -> List[int]:
        delta = self._step_delta(state)
        values = [anchor.steps]
        if strategy.target_steps is not None:
            values.append(strategy.target_steps)
        action = strategy.steps_action
        if action == "up":
            values.extend([anchor.steps + delta, int(round(anchor.steps * 1.5))])
        elif action == "down":
            values.extend([anchor.steps - delta, int(round(anchor.steps * 0.75))])
        elif action == "local":
            values.extend([anchor.steps - delta, anchor.steps + delta])
        elif strategy.search_action == "expand":
            values.extend([anchor.steps + delta, int(round(anchor.steps * 1.5))])
        elif strategy.search_action == "shrink":
            values.extend([anchor.steps - delta])
        return self._normalize_step_values(state, values)

    def _allocation_candidates_from_strategy(self, anchor: TrialConfig, strategy: ReflectionStrategy) -> List[str]:
        mode = strategy.allocation_action
        if mode == "prefer_margin_linear":
            return ["margin_linear", anchor.allocation.mode]
        if mode == "prefer_fixed":
            return ["fixed", anchor.allocation.mode]
        return [anchor.allocation.mode]

    def _clone_candidate(
        self,
        state: AttackSearchState,
        anchor: TrialConfig,
        *,
        epsilon: float,
        steps: int,
        allocation_mode: Optional[str] = None,
        min_steps: Optional[int] = None,
    ) -> Optional[TrialConfig]:
        epsilon_value = self._clip_epsilon(state, epsilon)
        step_value = self._clip_steps(state, steps)
        if epsilon_value is None or step_value is None:
            return None
        mode = allocation_mode or anchor.allocation.mode
        allocation = anchor.allocation
        if mode == "fixed":
            min_steps = None
        else:
            min_steps = max(1, min(step_value, min_steps if min_steps is not None else max(1, step_value // 2)))
        return TrialConfig(
            task_name=anchor.task_name,
            checkpoint_path=anchor.checkpoint_path,
            attack_name=anchor.attack_name,
            epsilon=epsilon_value,
            steps=step_value,
            restarts=anchor.restarts,
            rho=anchor.rho,
            seed=anchor.seed,
            allocation=anchor.allocation.__class__(
                mode=mode,
                min_steps=min_steps,
                margin_low=allocation.margin_low,
                margin_high=allocation.margin_high,
                epsilon_scale_low=allocation.epsilon_scale_low,
                epsilon_scale_high=allocation.epsilon_scale_high,
            ),
        )

    def _clip_epsilon(self, state: AttackSearchState, value: float) -> Optional[float]:
        max_base = max(float(epsilon) for epsilon in state.search_space.epsilons)
        min_base = min(float(epsilon) for epsilon in state.search_space.epsilons)
        lower = max(0.5, min_base / 2.0)
        upper = max_base * 2.0
        clipped = max(lower, min(upper, float(value)))
        if clipped <= 0:
            return None
        return round(clipped, 2)

    def _clip_steps(self, state: AttackSearchState, value: int) -> Optional[int]:
        max_base = max(int(step) for step in state.search_space.step_candidates)
        min_base = min(int(step) for step in state.search_space.step_candidates)
        lower = max(1, min_base // 2)
        upper = max_base * 2
        clipped = int(max(lower, min(upper, int(round(value)))))
        return clipped if clipped > 0 else None

    def _normalize_epsilon_values(self, state: AttackSearchState, values: Sequence[float]) -> List[float]:
        normalized: List[float] = []
        seen = set()
        for value in values:
            clipped = self._clip_epsilon(state, value)
            if clipped is None or clipped in seen:
                continue
            normalized.append(clipped)
            seen.add(clipped)
        return normalized

    def _normalize_step_values(self, state: AttackSearchState, values: Sequence[int]) -> List[int]:
        normalized: List[int] = []
        seen = set()
        for value in values:
            clipped = self._clip_steps(state, value)
            if clipped is None or clipped in seen:
                continue
            normalized.append(clipped)
            seen.add(clipped)
        return normalized

    def _epsilon_delta(self, state: AttackSearchState) -> float:
        sorted_eps = sorted(float(epsilon) for epsilon in state.search_space.epsilons)
        if len(sorted_eps) < 2:
            return max(1.0, sorted_eps[0] * 0.25)
        return max(1.0, min(b - a for a, b in zip(sorted_eps[:-1], sorted_eps[1:])))

    def _step_delta(self, state: AttackSearchState) -> int:
        sorted_steps = sorted(int(step) for step in state.search_space.step_candidates)
        if len(sorted_steps) < 2:
            return max(1, int(round(sorted_steps[0] * 0.25)))
        return max(1, min(b - a for a, b in zip(sorted_steps[:-1], sorted_steps[1:])))


class HeuristicAuditorAgent:
    def audit(self, state: AttackSearchState, result: TrialResult) -> AuditReport:
        baseline = state.baseline_result
        drop = normalized_reward_drop(baseline, result)
        telemetry = result.telemetry or {}
        flip_rate = float(telemetry.get("flip_rate", 0.0))
        clean_margin = float(telemetry.get("clean_margin_mean", 0.0))
        adv_margin = float(telemetry.get("adv_margin_mean", 0.0))

        failure_tags = []
        recommendations = {"epsilon_bias": 0.0, "steps_bias": 0.0}
        root_cause = "attack_effective"
        summary_parts = []
        strategy = ReflectionStrategy()

        if result.elapsed_seconds > state.runtime_budget_seconds:
            failure_tags.append("runtime_over_budget")
            recommendations["steps_bias"] = recommendations.get("steps_bias", 0.0) - 1.0
            recommendations["prefer_allocation"] = "margin_linear"
            root_cause = "runtime_constraint"
            summary_parts.append("runtime exceeded budget")
            strategy.search_action = "shrink"
            strategy.steps_action = "down"
            strategy.allocation_action = "prefer_margin_linear"
            strategy.target_steps = max(1, int(round(result.config.steps * 0.75)))
            strategy.confidence = 0.8

        if flip_rate < 0.1:
            failure_tags.append("no_flip")
            recommendations["epsilon_bias"] = recommendations.get("epsilon_bias", 0.0) + 1.0
            recommendations["steps_bias"] = recommendations.get("steps_bias", 0.0) + 0.5
            root_cause = "insufficient_action_change"
            summary_parts.append("action flips are rare")
            if clean_margin > 1.0:
                failure_tags.append("high_clean_margin")
                summary_parts.append("clean action margins are large")
            strategy.search_action = "expand"
            strategy.epsilon_action = "up"
            strategy.steps_action = "up"
            strategy.target_epsilon = round(result.config.epsilon * 1.5, 2)
            strategy.target_steps = max(result.config.steps + 2, int(round(result.config.steps * 1.5)))
            strategy.confidence = 0.9

        if flip_rate >= 0.1 and drop < 0.1:
            failure_tags.append("flip_without_reward_drop")
            recommendations["prefer_allocation"] = "margin_linear"
            recommendations["epsilon_bias"] = recommendations.get("epsilon_bias", 0.0) + 0.25
            root_cause = "low_impact_state_selection"
            summary_parts.append("actions flip but reward barely changes")
            strategy.search_action = "refine"
            strategy.epsilon_action = "keep"
            strategy.steps_action = "up"
            strategy.allocation_action = "prefer_margin_linear"
            strategy.target_epsilon = result.config.epsilon
            strategy.target_steps = max(result.config.steps + 2, int(round(result.config.steps * 1.25)))
            strategy.confidence = 0.75

        if drop >= 0.2:
            failure_tags.append("effective")
            summary_parts.append("reward drop is strong")
            if result.elapsed_seconds <= state.runtime_budget_seconds:
                recommendations["steps_bias"] = recommendations.get("steps_bias", 0.0) - 0.25
                strategy.search_action = "refine"
                strategy.epsilon_action = "local"
                strategy.steps_action = "local"
                strategy.target_epsilon = result.config.epsilon
                strategy.target_steps = result.config.steps
                strategy.confidence = max(strategy.confidence, 0.7)

        if adv_margin >= clean_margin and flip_rate < 0.2:
            failure_tags.append("insufficient_margin_reduction")
            recommendations["epsilon_bias"] = recommendations.get("epsilon_bias", 0.0) + 0.5
            summary_parts.append("attacked logits still show strong original preference")
            strategy.search_action = "expand"
            strategy.epsilon_action = "up"
            strategy.steps_action = "up"
            strategy.target_epsilon = round(result.config.epsilon * 1.25, 2)
            strategy.target_steps = max(result.config.steps + 2, int(round(result.config.steps * 1.25)))
            strategy.confidence = max(strategy.confidence, 0.8)

        if result.std_reward > abs(baseline.mean_reward) * 0.4:
            failure_tags.append("high_variance")
            summary_parts.append("episode variance is high")
            if strategy.search_action == "keep":
                strategy.search_action = "refine"
                strategy.epsilon_action = "local"
                strategy.steps_action = "local"
                strategy.target_epsilon = result.config.epsilon
                strategy.target_steps = result.config.steps
                strategy.confidence = max(strategy.confidence, 0.6)

        if len(failure_tags) == 0:
            failure_tags.append("neutral")
            summary_parts.append("trial is neither clearly good nor clearly bad")
            strategy.search_action = "refine"
            strategy.epsilon_action = "local"
            strategy.steps_action = "local"
            strategy.target_epsilon = result.config.epsilon
            strategy.target_steps = result.config.steps

        return AuditReport(
            trial_key=result.config.key(),
            failure_tags=failure_tags,
            summary="; ".join(summary_parts),
            root_cause=root_cause,
            recommendations=recommendations,
            strategy=strategy,
        )


@dataclass
class StructuredLLMResponse:
    output_text: str


class StructuredLLMAttackerAgent:
    def __init__(
        self,
        client: Any,
        model: str,
        max_candidates: int = 24,
        context_limit: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        self.max_candidates = max(4, max_candidates)
        self.context_limit = max(1, context_limit)
        self.heuristic = HeuristicAttackerAgent()

    def propose(self, state: AttackSearchState, batch_size: int) -> List[TrialConfig]:
        base_candidates = [c for c in state.search_space.candidates(state.task) if c.key() not in state.proposed_keys]
        candidates = self.heuristic._candidate_pool(state, base_candidates)
        if len(candidates) == 0:
            return []

        ranked = self.heuristic._rank_candidates(state, candidates)
        shortlist = self._shortlist_candidates(ranked)
        payload = self._proposal_payload(state, shortlist, batch_size)
        response = self.client.responses.create(
            model=self.model,
            instructions=self._proposal_instructions(batch_size),
            input=json.dumps(payload, ensure_ascii=False),
            text={"format": self._proposal_schema(batch_size)},
            store=False,
        )
        try:
            parsed = self._parse_json_output(response)
            proposals = self._select_proposals_from_llm(parsed, shortlist, batch_size)
        except Exception:
            proposals = []
        if len(proposals) == 0:
            proposals = shortlist[:batch_size]
        for proposal in proposals:
            state.proposed_keys.add(proposal.key())
        return proposals

    def _proposal_instructions(self, batch_size: int) -> str:
        return (
            "You are the attacker-side agent for DreamerV3 AutoAttack search. "
            "Choose parameter settings that maximize reward drop and action flip rate under the runtime budget. "
            "Use the task profile and retrieved prior experiences to pick a strong initial configuration when the current task has little or no history. "
            "Favor Pareto-improving candidates, balance exploitation with some exploration, and select at most "
            "{} candidate_ids from the provided menu. Return JSON only."
        ).format(batch_size)

    def _proposal_schema(self, batch_size: int) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "name": "attack_search_proposal",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "candidate_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": batch_size,
                    },
                    "summary": {"type": "string"},
                    "per_candidate_notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["candidate_id", "note"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["candidate_ids", "summary", "per_candidate_notes"],
                "additionalProperties": False,
            },
        }

    def _proposal_payload(
        self,
        state: AttackSearchState,
        shortlist: Sequence[TrialConfig],
        batch_size: int,
    ) -> Dict[str, Any]:
        candidate_menu = []
        for index, candidate in enumerate(shortlist):
            candidate_menu.append(
                {
                    "candidate_id": "candidate_{:02d}".format(index),
                    "epsilon": candidate.epsilon,
                    "steps": candidate.steps,
                    "restarts": candidate.restarts,
                    "rho": candidate.rho,
                    "allocation_mode": candidate.allocation.mode,
                    "min_steps": candidate.allocation.min_steps,
                }
            )

        return {
            "task": state.task.to_dict(),
            "task_profile": None if state.task_profile is None else state.task_profile.to_dict(),
            "attack_name": state.search_space.attack_name,
            "runtime_budget_seconds": state.runtime_budget_seconds,
            "requested_batch_size": batch_size,
            "baseline": _compact_result(state.baseline_result),
            "best_trial_so_far": _compact_result(state.best_result()),
            "recent_trials": [_compact_result(result) for result in state.recent_results(self.context_limit)],
            "recent_audits": [audit.to_dict() for audit in state.recent_audits(self.context_limit)],
            "reflection_memory": [reflection.to_dict() for reflection in state.recent_reflections(self.context_limit)],
            "prior_experiences": [item.to_dict() for item in state.top_prior_experiences(self.context_limit)],
            "candidate_menu": candidate_menu,
        }

    def _shortlist_candidates(self, ranked: Sequence[TrialConfig]) -> List[TrialConfig]:
        shortlist = list(ranked[: self.max_candidates])
        diverse_tail = []
        seen_signatures = {
            (candidate.epsilon, candidate.steps, candidate.allocation.mode, candidate.allocation.min_steps)
            for candidate in shortlist
        }
        for candidate in reversed(ranked):
            signature = (candidate.epsilon, candidate.steps, candidate.allocation.mode, candidate.allocation.min_steps)
            if signature in seen_signatures:
                continue
            diverse_tail.append(candidate)
            seen_signatures.add(signature)
            if len(shortlist) + len(diverse_tail) >= self.max_candidates + 4:
                break
        shortlist.extend(reversed(diverse_tail))
        return shortlist

    def _select_proposals_from_llm(
        self,
        parsed: Dict[str, Any],
        shortlist: Sequence[TrialConfig],
        batch_size: int,
    ) -> List[TrialConfig]:
        candidate_map = {"candidate_{:02d}".format(index): candidate for index, candidate in enumerate(shortlist)}
        proposals = []
        seen_keys = set()
        for candidate_id in parsed.get("candidate_ids", []):
            candidate = candidate_map.get(candidate_id)
            if candidate is None or candidate.key() in seen_keys:
                continue
            proposals.append(candidate)
            seen_keys.add(candidate.key())
            if len(proposals) >= batch_size:
                return proposals

        for candidate in shortlist:
            if candidate.key() in seen_keys:
                continue
            proposals.append(candidate)
            seen_keys.add(candidate.key())
            if len(proposals) >= batch_size:
                break
        return proposals

    def _parse_json_output(self, response: Any) -> Dict[str, Any]:
        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise RuntimeError("Structured LLM response did not include output_text.")
        return json.loads(output_text)


class StructuredLLMAuditorAgent:
    def __init__(
        self,
        client: Any,
        model: str,
        context_limit: int = 8,
    ) -> None:
        self.client = client
        self.model = model
        self.context_limit = max(1, context_limit)
        self.heuristic = HeuristicAuditorAgent()

    def audit(self, state: AttackSearchState, result: TrialResult) -> AuditReport:
        payload = {
            "task": state.task.to_dict(),
            "task_profile": None if state.task_profile is None else state.task_profile.to_dict(),
            "attack_name": state.search_space.attack_name,
            "runtime_budget_seconds": state.runtime_budget_seconds,
            "baseline": _compact_result(state.baseline_result),
            "trial": _compact_result(result),
            "best_trial_so_far": _compact_result(state.best_result()),
            "recent_trials": [_compact_result(item) for item in state.recent_results(self.context_limit)],
            "recent_audits": [audit.to_dict() for audit in state.recent_audits(self.context_limit)],
            "reflection_memory": [reflection.to_dict() for reflection in state.recent_reflections(self.context_limit)],
            "prior_experiences": [item.to_dict() for item in state.top_prior_experiences(self.context_limit)],
        }
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=self._audit_instructions(),
                input=json.dumps(payload, ensure_ascii=False),
                text={"format": self._audit_schema()},
                store=False,
            )
            parsed = self._parse_json_output(response)
            return self._audit_from_llm(result, parsed)
        except Exception:
            return self.heuristic.audit(state, result)

    def _audit_instructions(self) -> str:
        return (
            "You are the auditor-side agent for DreamerV3 AutoAttack search. "
            "Diagnose why this trial succeeded or failed. Use reward drop, flip rate, clean-vs-adv margins, "
            "variance, and runtime budget to explain the outcome. Return concise JSON only."
        )

    def _audit_schema(self) -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "name": "attack_search_audit",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "failure_tags": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "root_cause": {"type": "string"},
                    "recommendations": {
                        "type": "object",
                        "properties": {
                            "epsilon_bias": {"type": "number"},
                            "steps_bias": {"type": "number"},
                            "prefer_allocation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                            "avoid_allocation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                        "required": ["epsilon_bias", "steps_bias", "prefer_allocation", "avoid_allocation"],
                        "additionalProperties": False,
                    },
                    "strategy": {
                        "type": "object",
                        "properties": {
                            "search_action": {"type": "string"},
                            "epsilon_action": {"type": "string"},
                            "steps_action": {"type": "string"},
                            "allocation_action": {"type": "string"},
                            "target_epsilon": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                            "target_steps": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                            "confidence": {"type": "number"},
                        },
                        "required": [
                            "search_action",
                            "epsilon_action",
                            "steps_action",
                            "allocation_action",
                            "target_epsilon",
                            "target_steps",
                            "confidence"
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["failure_tags", "summary", "root_cause", "recommendations", "strategy"],
                "additionalProperties": False,
            },
        }

    def _audit_from_llm(self, result: TrialResult, parsed: Dict[str, Any]) -> AuditReport:
        recommendations = dict(parsed.get("recommendations", {}))
        recommendations["epsilon_bias"] = _clamp(float(recommendations.get("epsilon_bias", 0.0)), -2.0, 2.0)
        recommendations["steps_bias"] = _clamp(float(recommendations.get("steps_bias", 0.0)), -2.0, 2.0)
        recommendations = _sanitize_allocation_preferences(recommendations)

        failure_tags = [str(tag) for tag in parsed.get("failure_tags", []) if str(tag).strip()]
        if len(failure_tags) == 0:
            failure_tags = ["neutral"]

        strategy_payload = dict(parsed.get("strategy", {}))
        strategy = ReflectionStrategy(
            search_action=_normalize_search_action(strategy_payload.get("search_action", "keep")),
            epsilon_action=_normalize_direction_action(strategy_payload.get("epsilon_action", "keep")),
            steps_action=_normalize_direction_action(strategy_payload.get("steps_action", "keep")),
            allocation_action=_normalize_allocation_action(strategy_payload.get("allocation_action", "keep")),
            target_epsilon=(
                None if strategy_payload.get("target_epsilon") is None else float(strategy_payload.get("target_epsilon"))
            ),
            target_steps=(
                None if strategy_payload.get("target_steps") is None else int(strategy_payload.get("target_steps"))
            ),
            confidence=_clamp(float(strategy_payload.get("confidence", 0.5)), 0.0, 1.0),
        )

        return AuditReport(
            trial_key=result.config.key(),
            failure_tags=failure_tags,
            summary=str(parsed.get("summary", "")),
            root_cause=str(parsed.get("root_cause", "unknown")),
            recommendations=recommendations,
            strategy=strategy,
        )

    def _parse_json_output(self, response: Any) -> Dict[str, Any]:
        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise RuntimeError("Structured LLM response did not include output_text.")
        return json.loads(output_text)


class OpenAIAttackerAgent(StructuredLLMAttackerAgent):
    pass


class OpenAIAuditorAgent(StructuredLLMAuditorAgent):
    pass


class TransformersAttackerAgent(StructuredLLMAttackerAgent):
    pass


class TransformersAuditorAgent(StructuredLLMAuditorAgent):
    pass


class _TransformersResponsesEndpoint:
    def __init__(self, parent: "TransformersResponsesClient") -> None:
        self.parent = parent

    def create(self, **kwargs) -> StructuredLLMResponse:
        return self.parent.create(**kwargs)


class TransformersResponsesClient:
    _model_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        cache_dir: str = "",
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> None:
        self.cache_dir = cache_dir or ""
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.responses = _TransformersResponsesEndpoint(self)

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        text: Optional[Dict[str, Any]] = None,
        store: bool = False,
    ) -> StructuredLLMResponse:
        del store
        bundle = self._load_bundle(model)
        tokenizer = bundle["tokenizer"]
        model_obj = bundle["model"]
        device = bundle["device"]

        schema_text = ""
        if text and isinstance(text, dict):
            text_format = text.get("format")
            if text_format is not None:
                schema_text = "\nReturn exactly one JSON object that matches this schema:\n{}".format(
                    json.dumps(text_format, ensure_ascii=False)
                )

        system_prompt = instructions + schema_text
        user_prompt = "Structured input:\n{}".format(input)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}

        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = 0.9

        import torch

        with torch.no_grad():
            output_ids = model_obj.generate(**encoded, **generate_kwargs)

        generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        cleaned = _extract_json_object(raw_text) or raw_text
        return StructuredLLMResponse(output_text=cleaned)

    def _load_bundle(self, model_id: str) -> Dict[str, Any]:
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        use_cuda = torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir or None, use_fast=True)
        model_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir or None,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if use_cuda:
            model_kwargs["device_map"] = "auto"
        model_obj = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not use_cuda:
            model_obj.to("cpu")
        model_obj.eval()
        if hasattr(model_obj, "generation_config"):
            model_obj.generation_config.do_sample = False
            model_obj.generation_config.temperature = None
            model_obj.generation_config.top_p = None
            model_obj.generation_config.top_k = None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        bundle = {
            "tokenizer": tokenizer,
            "model": model_obj,
            "device": "cuda" if use_cuda else "cpu",
        }
        self._model_cache[model_id] = bundle
        return bundle


def build_reflection_note(baseline: TrialResult, result: TrialResult, audit: AuditReport) -> ReflectionNote:
    telemetry = result.telemetry or {}
    return ReflectionNote(
        trial_key=result.config.key(),
        attack_name=result.config.attack_name,
        stage=result.stage,
        summary=audit.summary,
        root_cause=audit.root_cause,
        failure_tags=list(audit.failure_tags),
        reward_drop=normalized_reward_drop(baseline, result),
        flip_rate=float(telemetry.get("flip_rate", 0.0)),
        clean_margin_mean=float(telemetry.get("clean_margin_mean", 0.0)),
        adv_margin_mean=float(telemetry.get("adv_margin_mean", 0.0)),
        elapsed_seconds=result.elapsed_seconds,
        strategy=audit.strategy,
    )


def _compact_result(result: Optional[TrialResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "stage": result.stage,
        "config": {
            "attack_name": result.config.attack_name,
            "epsilon": result.config.epsilon,
            "steps": result.config.steps,
            "restarts": result.config.restarts,
            "rho": result.config.rho,
            "allocation": result.config.allocation.to_dict(),
        },
        "mean_reward": result.mean_reward,
        "std_reward": result.std_reward,
        "median_reward": result.median_reward,
        "elapsed_seconds": result.elapsed_seconds,
        "telemetry": result.telemetry,
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sanitize_allocation_preferences(recommendations: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {"fixed", "margin_linear", None}
    prefer = recommendations.get("prefer_allocation")
    avoid = recommendations.get("avoid_allocation")
    recommendations["prefer_allocation"] = prefer if prefer in allowed else None
    recommendations["avoid_allocation"] = avoid if avoid in allowed else None
    return recommendations


def _normalize_search_action(value: Any) -> str:
    text = str(value).strip().lower()
    if text in ("expand", "broaden", "explore", "widen", "increase_range"):
        return "expand"
    if text in ("shrink", "narrow", "reduce", "tighten"):
        return "shrink"
    if text in ("refine", "local", "scout", "focus", "zoom_in"):
        return "refine"
    return "keep"


def _normalize_direction_action(value: Any) -> str:
    text = str(value).strip().lower()
    if text in ("up", "increase", "raise", "larger", "more"):
        return "up"
    if text in ("down", "decrease", "lower", "smaller", "less"):
        return "down"
    if text in ("local", "around", "nearby", "tune", "refine"):
        return "local"
    return "keep"


def _normalize_allocation_action(value: Any) -> str:
    text = str(value).strip().lower()
    if text in ("prefer_margin_linear", "use_margin_linear", "margin_linear"):
        return "prefer_margin_linear"
    if text in ("prefer_fixed", "use_fixed", "fixed"):
        return "prefer_fixed"
    return "keep"


def _extract_json_object(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    candidate = _strip_code_fence(text)
    if _is_valid_json(candidate):
        return candidate

    start = candidate.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                maybe = candidate[start : index + 1]
                if _is_valid_json(maybe):
                    return maybe
    return None


def _strip_code_fence(text: str) -> str:
    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
    match = fence_pattern.match(text)
    return match.group(1).strip() if match else text


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False
