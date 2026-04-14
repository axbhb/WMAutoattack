from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PlayerStateSnapshot:
    actions: Tensor
    recurrent_state: Tensor
    stochastic_state: Tensor


@dataclass
class PolicyAttackOutput:
    is_continuous: bool
    logits: Tuple[Tensor, ...]
    actions: Tuple[Tensor, ...]
    action_tensor: Optional[Tensor] = None
    std_tensor: Optional[Tensor] = None
    dists: Tuple[Any, ...] = ()


def _extract_policy_outputs(
    player: Any,
    obs: Dict[str, Tensor],
    snapshot: PlayerStateSnapshot,
    mask: Optional[Dict[str, Tensor]],
) -> PolicyAttackOutput:
    if hasattr(player, "get_policy_outputs"):
        outputs = player.get_policy_outputs(obs, mask=mask, state=snapshot)
        return PolicyAttackOutput(
            is_continuous=bool(outputs.get("is_continuous", False)),
            logits=tuple(outputs.get("logits") or ()),
            actions=tuple(outputs.get("actions") or ()),
            action_tensor=outputs.get("action_tensor"),
            std_tensor=outputs.get("std_tensor"),
            dists=tuple(outputs.get("dists") or ()),
        )

    logits = tuple(player.get_policy_logits(obs, mask=mask, state=snapshot))
    actions = tuple(logit.argmax(dim=-1) for logit in logits)
    return PolicyAttackOutput(is_continuous=False, logits=logits, actions=actions)


def _continuous_target_action(action_tensor: Tensor) -> Tensor:
    return action_tensor.detach().clamp(-0.999, 0.999)


def _continuous_negative_log_likelihood(policy: PolicyAttackOutput, clean_policy: PolicyAttackOutput) -> Tensor:
    if len(policy.dists) == 0 or clean_policy.action_tensor is None:
        raise RuntimeError("Continuous policy outputs must expose a differentiable distribution and target action.")
    target_action = _continuous_target_action(clean_policy.action_tensor)
    return (-policy.dists[0].log_prob(target_action)).mean()


def _continuous_action_shift(policy: PolicyAttackOutput, clean_policy: PolicyAttackOutput) -> Tensor:
    if policy.action_tensor is None or clean_policy.action_tensor is None:
        raise RuntimeError("Continuous policy outputs must expose action tensors.")
    return (policy.action_tensor - clean_policy.action_tensor.detach()).abs().mean()


def _extract_attack_features(
    player: Any,
    obs: Dict[str, Tensor],
    snapshot: PlayerStateSnapshot,
    mask: Optional[Dict[str, Tensor]],
) -> Tuple[PolicyAttackOutput, Dict[str, Tensor]]:
    player.restore_states(snapshot)
    embedded_obs = player.encoder(obs)
    next_recurrent_state = player.rssm.recurrent_model(
        torch.cat((snapshot.stochastic_state, snapshot.actions), -1),
        snapshot.recurrent_state,
    )
    if player.decoupled_rssm:
        _, next_stochastic_state = player.rssm._representation(embedded_obs)
    else:
        _, next_stochastic_state = player.rssm._representation(next_recurrent_state, embedded_obs)
    next_stochastic_state = next_stochastic_state.view(
        *next_stochastic_state.shape[:-2], player.stochastic_size * player.discrete_size
    )
    actor_input = torch.cat((next_stochastic_state, next_recurrent_state), -1)
    actor_hidden = player.actor.model(actor_input)
    policy = _extract_policy_outputs(player, obs, snapshot, mask)
    if policy.is_continuous:
        if policy.action_tensor is None or policy.std_tensor is None:
            raise RuntimeError("Continuous policy outputs must expose action and std tensors.")
        policy_feature = torch.cat((policy.action_tensor.reshape(-1), policy.std_tensor.reshape(-1)), dim=0).float()
    else:
        policy_feature = torch.cat([logit.reshape(-1).float() for logit in policy.logits], dim=0)
    features = {
        "encoder": embedded_obs,
        "recurrent": next_recurrent_state,
        "stochastic": next_stochastic_state,
        "actor_hidden": actor_hidden,
        "policy": policy_feature,
    }
    return policy, features


def _normalized_feature_loss(clean: Tensor, adv: Tensor) -> Tensor:
    clean_flat = clean.reshape(1, -1).float()
    adv_flat = adv.reshape(1, -1).float()
    clean_norm = F.normalize(clean_flat, dim=1)
    adv_norm = F.normalize(adv_flat, dim=1)
    return F.mse_loss(adv_norm, clean_norm)


class APGDAttack(ABC):
    def __init__(
        self,
        epsilon: float = 8.0,
        steps: int = 20,
        restarts: int = 1,
        rho: float = 0.75,
        seed: int = 0,
        cnn_keys: Optional[Sequence[str]] = None,
        continuous_action_threshold: float = 0.05,
    ) -> None:
        self.epsilon = float(epsilon) / 255.0
        self.steps = max(int(steps), 1)
        self.restarts = max(int(restarts), 1)
        self.rho = float(rho)
        self.seed = int(seed)
        self.cnn_keys = tuple(cnn_keys or ())
        self.continuous_action_threshold = float(continuous_action_threshold)

    def perturb(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        mask: Optional[Dict[str, Tensor]] = None,
        greedy: bool = False,
    ) -> Dict[str, Tensor]:
        if len(self.cnn_keys) == 0:
            return obs

        attacked_obs = {k: v.detach().clone() for k, v in obs.items()}
        snapshot = player.clone_states()
        best_obs = attacked_obs
        best_loss = None
        best_success = False

        for restart in range(self.restarts):
            generator = torch.Generator(device=obs[self.cnn_keys[0]].device.type)
            generator.manual_seed(self.seed + restart)
            current_obs = {k: v.detach().clone() for k, v in obs.items()}
            for key in self.cnn_keys:
                noise = torch.empty_like(current_obs[key]).uniform_(-self.epsilon, self.epsilon, generator=generator)
                current_obs[key] = self._project(current_obs[key] + noise, obs[key])

            restart_best_obs, restart_best_loss, restart_success = self._run_restart(
                player, current_obs, obs, snapshot, mask, greedy
            )
            if restart_success and not best_success:
                best_success = True
                best_loss = restart_best_loss
                best_obs = restart_best_obs
            elif restart_success == best_success and (best_loss is None or restart_best_loss > best_loss):
                best_loss = restart_best_loss
                best_obs = restart_best_obs
            player.restore_states(snapshot)
            if best_success:
                break

        return best_obs

    def _run_restart(
        self,
        player: Any,
        start_obs: Dict[str, Tensor],
        clean_obs: Dict[str, Tensor],
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
        greedy: bool,
    ) -> Tuple[Dict[str, Tensor], Tensor, bool]:
        del greedy
        x_adv = {k: v.detach().clone() for k, v in start_obs.items()}
        clean_policy = _extract_policy_outputs(player, clean_obs, snapshot, mask)
        step_size = max(self.epsilon * 2.0, 1.0 / 255.0)
        check_steps = max(int(self.steps * self.rho), 1)
        loss_history = []
        best_loss = None
        best_obs = x_adv
        best_success = False

        for step in range(self.steps):
            for key in self.cnn_keys:
                x_adv[key] = x_adv[key].detach().clone().requires_grad_(True)

            loss, gradients, policy = self._compute_loss_and_gradients(player, x_adv, clean_policy, snapshot, mask)
            loss_history.append(loss.item())
            is_success = self._is_success(policy, clean_policy)

            if is_success and not best_success:
                best_success = True
                best_loss = loss.detach()
                best_obs = {k: v.detach().clone() for k, v in x_adv.items()}
            elif is_success == best_success and (best_loss is None or loss > best_loss):
                best_loss = loss.detach()
                best_obs = {k: v.detach().clone() for k, v in x_adv.items()}
            if is_success:
                break

            with torch.no_grad():
                for key in self.cnn_keys:
                    updated = x_adv[key] + step_size * gradients[key].sign()
                    x_adv[key] = self._project(updated, clean_obs[key])

            if (step + 1) % check_steps == 0 and len(loss_history) >= check_steps:
                window = loss_history[-check_steps:]
                if max(window) <= max(loss_history[:-check_steps] + [-float("inf")]):
                    step_size = max(step_size / 2.0, 1.0 / 255.0)

        return best_obs, best_loss if best_loss is not None else torch.tensor(float("-inf")), best_success

    def _compute_loss_and_gradients(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Dict[str, Tensor], PolicyAttackOutput]:
        policy = _extract_policy_outputs(player, obs, snapshot, mask)
        loss = self._loss(policy, clean_policy)
        gradients = torch.autograd.grad(loss, [obs[key] for key in self.cnn_keys], retain_graph=False, create_graph=False)
        return loss.detach(), {k: g.detach() for k, g in zip(self.cnn_keys, gradients)}, policy

    def _project(self, x_adv: Tensor, x_clean: Tensor) -> Tensor:
        delta = torch.clamp(x_adv - x_clean, min=-self.epsilon, max=self.epsilon)
        return torch.clamp(x_clean + delta, min=-0.5, max=0.5).detach()

    def _is_success(self, policy: PolicyAttackOutput, clean_policy: PolicyAttackOutput) -> bool:
        if policy.is_continuous:
            return _continuous_action_shift(policy, clean_policy).item() > self.continuous_action_threshold
        return any(logit.argmax(dim=-1).ne(target).any().item() for logit, target in zip(policy.logits, clean_policy.actions))

    @abstractmethod
    def _loss(self, policy: PolicyAttackOutput, clean_policy: PolicyAttackOutput) -> Tensor:
        raise NotImplementedError


class APGDCrossEntropyAttack(APGDAttack):
    def _loss(self, policy: PolicyAttackOutput, clean_policy: PolicyAttackOutput) -> Tensor:
        if policy.is_continuous:
            return _continuous_negative_log_likelihood(policy, clean_policy)
        losses = [
            F.cross_entropy(logit.view(-1, logit.shape[-1]), target.view(-1))
            for logit, target in zip(policy.logits, clean_policy.actions)
        ]
        return torch.stack(losses).sum()


class APGDDLRAttack(APGDAttack):
    def _loss(self, policy: PolicyAttackOutput, clean_policy: PolicyAttackOutput) -> Tensor:
        if policy.is_continuous:
            if policy.action_tensor is None or clean_policy.action_tensor is None or policy.std_tensor is None:
                raise RuntimeError("Continuous APGD-DLR requires action and standard deviation tensors.")
            scaled_delta = (policy.action_tensor - clean_policy.action_tensor.detach()) / (policy.std_tensor + 1e-6)
            return scaled_delta.pow(2).mean()

        losses = []
        for logits, target in zip(policy.logits, clean_policy.actions):
            flattened_logits = logits.view(-1, logits.shape[-1])
            flattened_target = target.view(-1)
            if flattened_logits.shape[-1] < 3:
                # DLR is defined for at least 3 classes. Fall back to margin-style loss for smaller action spaces.
                selected = flattened_logits.gather(1, flattened_target.unsqueeze(1)).squeeze(1)
                top2 = torch.topk(flattened_logits, k=min(2, flattened_logits.shape[-1]), dim=1).values
                competitor = top2[:, -1]
                losses.append((competitor - selected).mean())
                continue

            sorted_logits, sorted_indices = flattened_logits.sort(dim=1)
            top1 = sorted_logits[:, -1]
            top3 = sorted_logits[:, -3]
            selected = flattened_logits.gather(1, flattened_target.unsqueeze(1)).squeeze(1)
            is_top1 = sorted_indices[:, -1].eq(flattened_target)
            alternative = torch.where(is_top1, sorted_logits[:, -2], top1)
            numerator = selected - alternative
            denominator = top1 - top3 + 1e-12
            losses.append((-numerator / denominator).mean())
        return torch.stack(losses).sum()


class FABLinfAttack:
    def __init__(
        self,
        epsilon: float = 8.0,
        steps: int = 20,
        restarts: int = 1,
        rho: float = 0.75,
        seed: int = 0,
        cnn_keys: Optional[Sequence[str]] = None,
        eta: float = 1.05,
        beta: float = 0.9,
        continuous_action_threshold: float = 0.05,
    ) -> None:
        self.epsilon = float(epsilon) / 255.0
        self.steps = max(int(steps), 1)
        self.restarts = max(int(restarts), 1)
        self.rho = float(rho)
        self.seed = int(seed)
        self.cnn_keys = tuple(cnn_keys or ())
        self.eta = float(eta)
        self.beta = float(beta)
        self.continuous_action_threshold = float(continuous_action_threshold)

    def perturb(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        mask: Optional[Dict[str, Tensor]] = None,
        greedy: bool = False,
    ) -> Dict[str, Tensor]:
        del greedy
        if len(self.cnn_keys) == 0:
            return obs

        snapshot = player.clone_states()
        best_obs = {k: v.detach().clone() for k, v in obs.items()}
        best_distance = None

        clean_policy = _extract_policy_outputs(player, obs, snapshot, mask)

        for restart in range(self.restarts):
            generator = torch.Generator(device=obs[self.cnn_keys[0]].device.type)
            generator.manual_seed(self.seed + restart)
            current_obs = {k: v.detach().clone() for k, v in obs.items()}
            for key in self.cnn_keys:
                noise = torch.empty_like(current_obs[key]).uniform_(-self.epsilon, self.epsilon, generator=generator)
                current_obs[key] = self._project(current_obs[key] + noise, obs[key])

            current_obs = self._run_restart(player, current_obs, obs, clean_policy, snapshot, mask)
            distance = max(torch.max(torch.abs(current_obs[key] - obs[key])).item() for key in self.cnn_keys)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_obs = current_obs
            player.restore_states(snapshot)

        return best_obs

    def _run_restart(
        self,
        player: Any,
        start_obs: Dict[str, Tensor],
        clean_obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        x_adv = {k: v.detach().clone() for k, v in start_obs.items()}
        best_adv = {k: v.detach().clone() for k, v in start_obs.items()}
        best_margin = None

        for _ in range(self.steps):
            step_adv, margin = self._fab_step(player, x_adv, clean_obs, clean_policy, snapshot, mask)
            x_adv = step_adv
            if best_margin is None or margin < best_margin:
                best_margin = margin
                best_adv = {k: v.detach().clone() for k, v in x_adv.items()}
            if margin <= 0:
                break

        return best_adv

    def _fab_step(
        self,
        player: Any,
        x_adv: Dict[str, Tensor],
        clean_obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Tuple[Dict[str, Tensor], float]:
        differentiable_obs = {k: v.detach().clone().requires_grad_(k in self.cnn_keys) for k, v in x_adv.items()}
        policy = _extract_policy_outputs(player, differentiable_obs, snapshot, mask)

        if policy.is_continuous:
            action_shift = _continuous_action_shift(policy, clean_policy)
            margin_tensor = torch.as_tensor(self.continuous_action_threshold, device=action_shift.device) - action_shift
            margin_objective = action_shift
            grad_margin = torch.autograd.grad(
                margin_objective, [differentiable_obs[key] for key in self.cnn_keys], retain_graph=False
            )
            grad_diff = {key: gm.detach() for key, gm in zip(self.cnn_keys, grad_margin)}
            norm_l1 = sum(g.abs().reshape(g.shape[0], -1).sum(dim=1).mean() for g in grad_diff.values()) + 1e-12
            distance = (margin_tensor.detach().abs() / norm_l1).item()
            best_grad_diff = grad_diff
            best_margin = margin_tensor.detach().item()
        else:
            logits_seq = policy.logits

            candidate_steps = []
            candidate_margins = []
            for logits, target in zip(logits_seq, clean_policy.actions):
                flat_logits = logits.view(-1, logits.shape[-1])
                flat_target = target.view(-1)
                row_idx = torch.arange(flat_logits.shape[0], device=flat_logits.device)
                target_logits = flat_logits[row_idx, flat_target]
                competitor_idx = flat_logits.argmax(dim=-1)
                target_is_top1 = competitor_idx.eq(flat_target)
                top2_idx = torch.topk(flat_logits, k=min(2, flat_logits.shape[-1]), dim=1).indices[:, -2]
                competitor_idx = torch.where(target_is_top1, top2_idx, competitor_idx)
                competitor_logits = flat_logits[row_idx, competitor_idx]
                margin = (target_logits - competitor_logits).mean()

                margin_objective = (competitor_logits - target_logits).sum()
                grad_margin = torch.autograd.grad(
                    margin_objective, [differentiable_obs[key] for key in self.cnn_keys], retain_graph=True
                )
                grad_diff = {key: gm.detach() for key, gm in zip(self.cnn_keys, grad_margin)}

                norm_l1 = sum(g.abs().reshape(g.shape[0], -1).sum(dim=1).mean() for g in grad_diff.values()) + 1e-12
                distance = (margin.detach().abs() / norm_l1).item()
                candidate_steps.append((distance, grad_diff))
                candidate_margins.append(margin.detach().item())

            _, best_grad_diff = min(candidate_steps, key=lambda item: item[0])
            best_margin = min(candidate_margins)

        with torch.no_grad():
            updated_obs = {k: v.detach().clone() for k, v in x_adv.items()}
            for key in self.cnn_keys:
                signed_update = best_grad_diff[key].sign()
                updated_obs[key] = x_adv[key] + self.eta * self.epsilon * signed_update
                blended = clean_obs[key] + self.beta * (updated_obs[key] - clean_obs[key])
                updated_obs[key] = self._project(blended, clean_obs[key])
        return updated_obs, best_margin

    def _project(self, x_adv: Tensor, x_clean: Tensor) -> Tensor:
        delta = torch.clamp(x_adv - x_clean, min=-self.epsilon, max=self.epsilon)
        return torch.clamp(x_clean + delta, min=-0.5, max=0.5).detach()


class SquareAttack:
    def __init__(
        self,
        epsilon: float = 8.0,
        steps: int = 60,
        restarts: int = 1,
        rho: float = 0.75,
        seed: int = 0,
        cnn_keys: Optional[Sequence[str]] = None,
        saliency_refresh: int = 12,
        min_square_size: int = 1,
        continuous_action_threshold: float = 0.05,
    ) -> None:
        self.rho = float(rho)
        self.epsilon = float(epsilon) / 255.0
        self.steps = max(int(steps), 1)
        self.restarts = max(int(restarts), 1)
        self.seed = int(seed)
        self.cnn_keys = tuple(cnn_keys or ())
        self.saliency_refresh = max(int(saliency_refresh), 1)
        self.min_square_size = max(int(min_square_size), 1)
        self.candidates_per_step = 2
        self.continuous_action_threshold = float(continuous_action_threshold)

    def perturb(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        mask: Optional[Dict[str, Tensor]] = None,
        greedy: bool = False,
    ) -> Dict[str, Tensor]:
        del greedy
        if len(self.cnn_keys) == 0:
            return obs

        snapshot = player.clone_states()
        clean_policy = _extract_policy_outputs(player, obs, snapshot, mask)
        best_obs = {k: v.detach().clone() for k, v in obs.items()}
        best_loss, _ = self._compute_loss(player, obs, clean_policy, snapshot, mask)
        best_success = False

        for restart in range(self.restarts):
            generator = torch.Generator(device=obs[self.cnn_keys[0]].device.type)
            generator.manual_seed(self.seed + restart)
            candidate = {k: v.detach().clone() for k, v in obs.items()}
            current_loss, current_success = self._compute_loss(player, candidate, clean_policy, snapshot, mask)
            saliency = self._compute_saliency(player, candidate, clean_policy, snapshot, mask)
            for step_idx in range(self.steps):
                if current_success:
                    break
                if step_idx % self.saliency_refresh == 0:
                    saliency = self._compute_saliency(player, candidate, clean_policy, snapshot, mask)
                proposal = None
                proposal_loss = current_loss
                proposal_success = current_success
                for _ in range(self.candidates_per_step):
                    step_candidate = self._square_step(candidate, obs, generator, saliency, step_idx)
                    step_loss, step_success = self._compute_loss(player, step_candidate, clean_policy, snapshot, mask)
                    if proposal is None or self._should_replace(proposal_loss, proposal_success, step_loss, step_success):
                        proposal = step_candidate
                        proposal_loss = step_loss
                        proposal_success = step_success
                if proposal is not None and self._should_replace(current_loss, current_success, proposal_loss, proposal_success):
                    candidate = proposal
                    current_loss = proposal_loss
                    current_success = proposal_success
                    if self._should_replace(best_loss, best_success, proposal_loss, proposal_success):
                        best_loss = proposal_loss
                        best_success = proposal_success
                        best_obs = {k: v.detach().clone() for k, v in proposal.items()}
                    if proposal_success:
                        break
            player.restore_states(snapshot)
            if best_success:
                break

        return best_obs

    def _square_step(
        self,
        current_obs: Dict[str, Tensor],
        clean_obs: Dict[str, Tensor],
        generator: torch.Generator,
        saliency: Dict[str, Tensor],
        step_idx: int,
    ) -> Dict[str, Tensor]:
        proposal = {k: v.detach().clone() for k, v in current_obs.items()}
        for key in self.cnn_keys:
            value = proposal[key]
            _, _, _, height, width = value.shape
            p = self._square_size_ratio(step_idx / max(self.steps - 1, 1))
            square_size = max(int(round(min(height, width) * p)), self.min_square_size)
            top, left = self._sample_square_location(saliency[key], square_size, height, width, generator)
            random_sign = self._sample_signed_patch(value, saliency[key], top, left, square_size, generator)
            updated = value.clone()
            updated[:, :, :, top : top + square_size, left : left + square_size] = (
                clean_obs[key][:, :, :, top : top + square_size, left : left + square_size] + random_sign * self.epsilon
            )
            proposal[key] = self._project(updated, clean_obs[key])
        return proposal

    def _compute_saliency(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        differentiable_obs = {k: v.detach().clone().requires_grad_(k in self.cnn_keys) for k, v in obs.items()}
        policy = _extract_policy_outputs(player, differentiable_obs, snapshot, mask)
        if policy.is_continuous:
            loss = _continuous_negative_log_likelihood(policy, clean_policy)
        else:
            losses = [
                F.cross_entropy(logit.view(-1, logit.shape[-1]), target.view(-1))
                for logit, target in zip(policy.logits, clean_policy.actions)
            ]
            loss = torch.stack(losses).sum()
        gradients = torch.autograd.grad(loss, [differentiable_obs[key] for key in self.cnn_keys], retain_graph=False)
        saliency = {}
        for key, grad in zip(self.cnn_keys, gradients):
            saliency[key] = grad.detach().abs()
        return saliency

    def _square_size_ratio(self, progress: float) -> float:
        if progress < 0.05:
            return 0.8
        if progress < 0.15:
            return 0.5
        if progress < 0.3:
            return 0.35
        if progress < 0.5:
            return 0.25
        if progress < 0.7:
            return 0.15
        if progress < 0.85:
            return 0.1
        return 0.06

    def _sample_square_location(
        self,
        saliency: Tensor,
        square_size: int,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> Tuple[int, int]:
        score_map = saliency.mean(dim=(0, 1, 2))
        if square_size >= height or square_size >= width:
            return 0, 0
        valid_h = height - square_size + 1
        valid_w = width - square_size + 1
        pooled = F.avg_pool2d(score_map.unsqueeze(0).unsqueeze(0), kernel_size=square_size, stride=1)[0, 0]
        probs = pooled.reshape(-1)
        probs_sum = probs.sum()
        if not torch.isfinite(probs_sum) or probs_sum <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        else:
            probs = probs / probs_sum
        flat_idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).item()
        return flat_idx // valid_w, flat_idx % valid_w

    def _sample_signed_patch(
        self,
        current_value: Tensor,
        saliency: Tensor,
        top: int,
        left: int,
        square_size: int,
        generator: torch.Generator,
    ) -> Tensor:
        del current_value
        patch = torch.where(
            torch.rand((1, 1, saliency.shape[2], square_size, square_size), generator=generator, device=saliency.device)
            > 0.5,
            torch.ones((1, 1, saliency.shape[2], square_size, square_size), device=saliency.device),
            -torch.ones((1, 1, saliency.shape[2], square_size, square_size), device=saliency.device),
        )
        # Prefer the most salient temporal slice when observations contain frame stacks.
        channels = saliency.shape[2]
        channel_groups = 3 if channels % 3 == 0 else 1
        num_groups = max(channels // channel_groups, 1)
        saliency_patch = saliency[:, :, :, top : top + square_size, left : left + square_size]
        grouped_saliency = saliency_patch.reshape(
            saliency_patch.shape[0], saliency_patch.shape[1], num_groups, channel_groups, square_size, square_size
        )
        temporal_scores = grouped_saliency.mean(dim=(0, 1, 3, 4, 5))
        target_group = temporal_scores.argmax().item()
        temporal_mask = torch.zeros_like(patch)
        start = target_group * channel_groups
        end = min(start + channel_groups, channels)
        temporal_mask[:, :, start:end] = 1
        return patch * temporal_mask

    def _compute_loss(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Tuple[Tensor, bool]:
        policy = _extract_policy_outputs(player, obs, snapshot, mask)
        if policy.is_continuous:
            loss = _continuous_negative_log_likelihood(policy, clean_policy).detach()
            success = _continuous_action_shift(policy, clean_policy).item() > self.continuous_action_threshold
        else:
            losses = [
                F.cross_entropy(logit.view(-1, logit.shape[-1]), target.view(-1))
                for logit, target in zip(policy.logits, clean_policy.actions)
            ]
            loss = torch.stack(losses).sum().detach()
            success = any(logit.argmax(dim=-1).ne(target).any().item() for logit, target in zip(policy.logits, clean_policy.actions))
        return loss, success

    def _project(self, x_adv: Tensor, x_clean: Tensor) -> Tensor:
        delta = torch.clamp(x_adv - x_clean, min=-self.epsilon, max=self.epsilon)
        return torch.clamp(x_clean + delta, min=-0.5, max=0.5).detach()

    def _should_replace(self, current_loss: Tensor, current_success: bool, new_loss: Tensor, new_success: bool) -> bool:
        if new_success and not current_success:
            return True
        if new_success == current_success and new_loss > current_loss:
            return True
        return False


class TwoStageMomentumAttack:
    def __init__(
        self,
        epsilon: float = 8.0,
        steps: int = 20,
        restarts: int = 1,
        rho: float = 0.75,
        seed: int = 0,
        cnn_keys: Optional[Sequence[str]] = None,
        tau: float = 0.15,
        momentum: float = 0.9,
        beta: float = 0.6,
        stage1_ratio: float = 0.4,
        alpha: float = 0.1,
        continuous_action_threshold: float = 0.05,
    ) -> None:
        del rho
        self.epsilon = float(epsilon) / 255.0
        self.steps = max(int(steps), 1)
        self.restarts = max(int(restarts), 1)
        self.seed = int(seed)
        self.cnn_keys = tuple(cnn_keys or ())
        self.tau = float(tau)
        self.momentum = float(momentum)
        self.beta = float(beta)
        self.stage1_ratio = min(max(float(stage1_ratio), 0.1), 0.9)
        self.alpha = float(alpha)
        self.continuous_action_threshold = float(continuous_action_threshold)

    def perturb(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        mask: Optional[Dict[str, Tensor]] = None,
        greedy: bool = False,
    ) -> Dict[str, Tensor]:
        del greedy
        if len(self.cnn_keys) == 0:
            return obs

        snapshot = player.clone_states()
        clean_policy, clean_features = _extract_attack_features(player, obs, snapshot, mask)
        best_obs = {k: v.detach().clone() for k, v in obs.items()}
        best_score = torch.tensor(float("-inf"), device=obs[self.cnn_keys[0]].device)
        best_success = False

        for restart in range(self.restarts):
            generator = torch.Generator(device=obs[self.cnn_keys[0]].device.type)
            generator.manual_seed(self.seed + restart)
            current_obs = {k: v.detach().clone() for k, v in obs.items()}
            for key in self.cnn_keys:
                noise = torch.empty_like(current_obs[key]).uniform_(-self.epsilon, self.epsilon, generator=generator)
                current_obs[key] = self._project(current_obs[key] + noise, obs[key])

            current_obs = self._run_stage_one(player, current_obs, obs, clean_policy, clean_features, snapshot, mask)
            restart_best_obs, restart_best_score, restart_success = self._run_stage_two(
                player,
                current_obs,
                obs,
                clean_policy,
                clean_features,
                snapshot,
                mask,
            )
            if restart_success and not best_success:
                best_success = True
                best_score = restart_best_score
                best_obs = restart_best_obs
            elif restart_success == best_success and restart_best_score > best_score:
                best_score = restart_best_score
                best_obs = restart_best_obs
            player.restore_states(snapshot)
            if best_success:
                break

        return best_obs

    def _run_stage_one(
        self,
        player: Any,
        start_obs: Dict[str, Tensor],
        clean_obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        clean_features: Dict[str, Tensor],
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        x_adv = {k: v.detach().clone() for k, v in start_obs.items()}
        stage_steps = max(int(round(self.steps * self.stage1_ratio)), 1)
        step_size = max(self.epsilon / max(self.steps, 1), 1.0 / 255.0)
        best_valid_obs = {k: v.detach().clone() for k, v in x_adv.items()}

        for _ in range(stage_steps):
            differentiable_obs = {
                k: v.detach().clone().requires_grad_(k in self.cnn_keys)
                for k, v in x_adv.items()
            }
            attack_loss, fidelity_loss, _, _ = self._compute_attack_and_fidelity(
                player,
                differentiable_obs,
                clean_policy,
                clean_features,
                snapshot,
                mask,
            )
            current_fidelity = float(fidelity_loss.detach().item())
            if current_fidelity >= self.tau:
                break

            objective = attack_loss - self.alpha * fidelity_loss
            gradients = torch.autograd.grad(objective, [differentiable_obs[key] for key in self.cnn_keys], retain_graph=False)
            candidate_obs = {k: v.detach().clone() for k, v in x_adv.items()}
            with torch.no_grad():
                for key, grad in zip(self.cnn_keys, gradients):
                    candidate_obs[key] = self._project(x_adv[key] + step_size * grad.sign(), clean_obs[key])

            _, candidate_fidelity, _, _ = self._compute_attack_and_fidelity(
                player,
                candidate_obs,
                clean_policy,
                clean_features,
                snapshot,
                mask,
            )
            if float(candidate_fidelity.detach().item()) > self.tau:
                break
            x_adv = candidate_obs
            best_valid_obs = {k: v.detach().clone() for k, v in x_adv.items()}

        return best_valid_obs

    def _run_stage_two(
        self,
        player: Any,
        start_obs: Dict[str, Tensor],
        clean_obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        clean_features: Dict[str, Tensor],
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Tuple[Dict[str, Tensor], Tensor, bool]:
        x_adv = {k: v.detach().clone() for k, v in start_obs.items()}
        best_obs = {k: v.detach().clone() for k, v in x_adv.items()}
        best_score = torch.tensor(float("-inf"), device=clean_obs[self.cnn_keys[0]].device)
        best_success = False
        momentum_buffer = {key: torch.zeros_like(x_adv[key]) for key in self.cnn_keys}
        stage_steps = max(self.steps - max(int(round(self.steps * self.stage1_ratio)), 1), 1)
        step_size = max(self.epsilon / max(self.steps, 1), 1.0 / 255.0)

        for _ in range(stage_steps):
            differentiable_obs = {
                k: v.detach().clone().requires_grad_(k in self.cnn_keys)
                for k, v in x_adv.items()
            }
            attack_loss, fidelity_loss, policy, _ = self._compute_attack_and_fidelity(
                player,
                differentiable_obs,
                clean_policy,
                clean_features,
                snapshot,
                mask,
            )
            objective = attack_loss - self.beta * fidelity_loss
            gradients = torch.autograd.grad(objective, [differentiable_obs[key] for key in self.cnn_keys], retain_graph=False)
            score = objective.detach()
            is_success = self._is_success(policy, clean_policy)

            if is_success and not best_success:
                best_success = True
                best_score = score
                best_obs = {k: v.detach().clone() for k, v in x_adv.items()}
            elif is_success == best_success and score > best_score:
                best_score = score
                best_obs = {k: v.detach().clone() for k, v in x_adv.items()}

            with torch.no_grad():
                for key, grad in zip(self.cnn_keys, gradients):
                    normalized_grad = grad / grad.abs().mean().clamp_min(1e-12)
                    momentum_buffer[key] = self.momentum * momentum_buffer[key] + (1.0 - self.momentum) * normalized_grad
                    x_adv[key] = self._project(x_adv[key] + step_size * momentum_buffer[key].sign(), clean_obs[key])

            if is_success:
                break

        return best_obs, best_score, best_success

    def _compute_attack_and_fidelity(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        clean_policy: PolicyAttackOutput,
        clean_features: Dict[str, Tensor],
        snapshot: PlayerStateSnapshot,
        mask: Optional[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, PolicyAttackOutput, Dict[str, Tensor]]:
        policy, adv_features = _extract_attack_features(player, obs, snapshot, mask)
        if policy.is_continuous:
            attack_loss = _continuous_negative_log_likelihood(policy, clean_policy)
        else:
            attack_loss = torch.stack(
                [
                    F.cross_entropy(logit.view(-1, logit.shape[-1]), target.view(-1))
                    for logit, target in zip(policy.logits, clean_policy.actions)
                ]
            ).sum()
        fidelity_terms = []
        for name, clean_feature in clean_features.items():
            fidelity_terms.append(_normalized_feature_loss(clean_feature.detach(), adv_features[name]))
        fidelity_loss = torch.stack(fidelity_terms).mean()
        return attack_loss, fidelity_loss, policy, adv_features

    def _is_success(self, policy: PolicyAttackOutput, clean_policy: PolicyAttackOutput) -> bool:
        if policy.is_continuous:
            return _continuous_action_shift(policy, clean_policy).item() > self.continuous_action_threshold
        return any(logit.argmax(dim=-1).ne(target).any().item() for logit, target in zip(policy.logits, clean_policy.actions))

    def _project(self, x_adv: Tensor, x_clean: Tensor) -> Tensor:
        delta = torch.clamp(x_adv - x_clean, min=-self.epsilon, max=self.epsilon)
        return torch.clamp(x_clean + delta, min=-0.5, max=0.5).detach()


def build_attack(cfg: Dict[str, Any]) -> Optional[Any]:
    attack_cfg = getattr(cfg, "attack", None)
    if attack_cfg is None or not attack_cfg.enabled:
        return None

    attack_name = attack_cfg.name.lower()
    attack_cls: Any
    if attack_name == "apgd_ce":
        attack_cls = APGDCrossEntropyAttack
    elif attack_name == "apgd_dlr":
        attack_cls = APGDDLRAttack
    elif attack_name == "fab":
        attack_cls = FABLinfAttack
    elif attack_name == "square":
        attack_cls = SquareAttack
    elif attack_name == "two_stage":
        attack_cls = TwoStageMomentumAttack
    else:
        raise ValueError(f"Unsupported attack '{attack_cfg.name}'.")
    kwargs = {
        "epsilon": attack_cfg.epsilon,
        "steps": attack_cfg.steps,
        "restarts": attack_cfg.restarts,
        "rho": attack_cfg.rho,
        "seed": attack_cfg.seed,
        "cnn_keys": cfg.algo.cnn_keys.encoder,
    }
    if attack_name == "two_stage":
        kwargs.update(
            tau=getattr(attack_cfg, "tau", 0.15),
            momentum=getattr(attack_cfg, "momentum", 0.9),
            beta=getattr(attack_cfg, "beta", 0.6),
            stage1_ratio=getattr(attack_cfg, "stage1_ratio", 0.4),
            alpha=getattr(attack_cfg, "alpha", 0.1),
        )
    return attack_cls(**kwargs)
