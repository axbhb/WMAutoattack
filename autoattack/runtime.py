from __future__ import annotations

import json
import math
import os
import pathlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.attacks import APGDCrossEntropyAttack, APGDDLRAttack, FABLinfAttack, SquareAttack
from sheeprl.algos.dreamer_v3.utils import prepare_obs
from autoattack.probe import ProbeFeatureCollector
from agent.memory import tokenize_task_name
from agent.schema import SearchConfig, TaskProfile, TaskSpec, TrialConfig, TrialResult, TrialTelemetry
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict


@dataclass
class _TaskContext:
    task: TaskSpec
    cfg: Dict[str, Any]
    state: Dict[str, Any]


class MarginStepAllocator:
    def allocate(self, trial: TrialConfig, clean_margin: float) -> Tuple[int, float]:
        allocation = trial.allocation
        if allocation.mode == "fixed":
            return trial.steps, trial.epsilon

        min_steps = allocation.min_steps or max(1, trial.steps // 2)
        if allocation.margin_high <= allocation.margin_low:
            hardness = 1.0
        else:
            hardness = (clean_margin - allocation.margin_low) / (allocation.margin_high - allocation.margin_low)
        hardness = max(0.0, min(1.0, hardness))
        steps = int(round(min_steps + hardness * (trial.steps - min_steps)))
        epsilon_scale = allocation.epsilon_scale_low + hardness * (
            allocation.epsilon_scale_high - allocation.epsilon_scale_low
        )
        epsilon = max(trial.epsilon * epsilon_scale, 0.0)
        return max(1, steps), epsilon


class DreamerV3SearchExecutor:
    def __init__(self, search_config: SearchConfig) -> None:
        self.search_config = search_config
        self.output_dir = search_config.ensure_output_dir()
        self.fabric = Fabric(
            accelerator=search_config.accelerator,
            devices=search_config.devices,
            precision=search_config.precision,
        )
        self.fabric.launch()
        torch.set_float32_matmul_precision("high")
        self._contexts: Dict[str, _TaskContext] = {}
        self._allocator = MarginStepAllocator()

        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath

    def run_trial(
        self,
        trial: TrialConfig,
        *,
        stage: str,
        num_episodes: int,
        persist_artifacts: bool,
    ) -> TrialResult:
        context = self._get_context(TaskSpec(name=trial.task_name, checkpoint_path=trial.checkpoint_path))
        eval_cfg = self._build_eval_cfg(
            context.cfg,
            trial,
            capture_video=persist_artifacts and self.search_config.capture_video_for_confirm,
        )
        trial_dir = self._create_trial_dir(trial, stage)
        summary_writer = SummaryWriter(trial_dir) if persist_artifacts and self.search_config.tensorboard_for_confirm else None
        probe_collector = (
            ProbeFeatureCollector(self.search_config.experience_probe_max_steps)
            if self.search_config.experience_retrieval_mode in ("latent", "hybrid")
            else None
        )

        env = make_env(eval_cfg, eval_cfg.seed, 0, str(trial_dir), "test", vector_env_idx=0)()
        try:
            action_space = env.action_space
            obs_space = env.observation_space
            is_continuous = isinstance(action_space, gym.spaces.Box)
            is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
            actions_dim = tuple(
                action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
            )
            _, _, _, _, player = build_agent(
                self.fabric,
                actions_dim,
                is_continuous,
                eval_cfg,
                obs_space,
                context.state["world_model"],
                context.state["actor"],
            )
            player.num_envs = 1
            player.eval()

            returns = []
            telemetry = TrialTelemetry()
            start_time = time.perf_counter()
            for episode_idx in range(num_episodes):
                obs = env.reset(seed=eval_cfg.seed + episode_idx)[0]
                player.init_states()
                done = False
                cumulative_rew = 0.0
                while not done:
                    torch_obs = prepare_obs(self.fabric, obs, cnn_keys=eval_cfg.algo.cnn_keys.encoder)
                    action_mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                    action_mask = action_mask if len(action_mask) > 0 else None

                    snapshot = player.clone_states()
                    clean_logits = player.get_policy_logits(torch_obs, mask=action_mask, state=snapshot)
                    clean_actions, clean_margin = self._actions_and_margin(clean_logits)
                    clean_probe_features = None
                    if probe_collector is not None and probe_collector.should_collect():
                        clean_probe_features = self._extract_probe_features(player, torch_obs, action_mask, snapshot)

                    attack_steps = 0
                    attack_epsilon = 0.0
                    attacked_obs = torch_obs
                    adv_actions = clean_actions
                    adv_margin = clean_margin
                    flipped = False
                    adv_probe_features = clean_probe_features
                    if not trial.is_baseline:
                        attack_steps, attack_epsilon = self._allocator.allocate(trial, clean_margin)
                        attacker = self._create_attack(trial, attack_steps, attack_epsilon, eval_cfg.algo.cnn_keys.encoder)
                        attacked_obs = attacker.perturb(player, torch_obs, action_mask, greedy=False)
                        adv_logits = player.get_policy_logits(attacked_obs, mask=action_mask, state=snapshot)
                        adv_actions, adv_margin = self._actions_and_margin(adv_logits)
                        flipped = adv_actions != clean_actions
                        if probe_collector is not None and probe_collector.should_collect():
                            adv_probe_features = self._extract_probe_features(player, attacked_obs, action_mask, snapshot)
                    telemetry.update(clean_margin, adv_margin, flipped, attack_steps, attack_epsilon)
                    if probe_collector is not None and clean_probe_features is not None:
                        probe_collector.update(clean_probe_features, adv_probe_features)

                    if trial.is_baseline:
                        with torch.no_grad():
                            real_actions = player.get_actions(torch_obs, greedy=False, mask=action_mask)
                    else:
                        real_actions = player.get_actions(attacked_obs, greedy=False, mask=action_mask)

                    if player.actor.is_continuous:
                        real_actions = torch.stack(real_actions, -1).cpu().numpy()
                    else:
                        real_actions = torch.stack(
                            [real_action.argmax(dim=-1) for real_action in real_actions], dim=-1
                        ).cpu().numpy()

                    obs, reward, terminated, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
                    done = bool(terminated or truncated)
                    cumulative_rew += float(reward)

                returns.append(cumulative_rew)
                if summary_writer is not None:
                    summary_writer.add_scalar("episode_reward", cumulative_rew, episode_idx)

            elapsed = time.perf_counter() - start_time
            telemetry_dict = telemetry.to_dict()
            probe_representation = None
            if probe_collector is not None:
                probe_representation = probe_collector.build(
                    source_stage=stage,
                    extras={
                        "mean_return": float(np.mean(returns)),
                        "std_return": float(np.std(returns)),
                        "flip_rate": float(telemetry_dict.get("flip_rate", 0.0)),
                        "clean_margin_mean": float(telemetry_dict.get("clean_margin_mean", 0.0)),
                        "adv_margin_mean": float(telemetry_dict.get("adv_margin_mean", 0.0)),
                        "allocated_steps_mean": float(telemetry_dict.get("allocated_steps_mean", 0.0)),
                        "allocated_epsilon_mean": float(telemetry_dict.get("allocated_epsilon_mean", 0.0)),
                    },
                )
            result = TrialResult(
                config=trial,
                stage=stage,
                num_episodes=num_episodes,
                mean_reward=float(np.mean(returns)),
                std_reward=float(np.std(returns)),
                median_reward=float(np.median(returns)),
                min_reward=float(np.min(returns)),
                max_reward=float(np.max(returns)),
                elapsed_seconds=float(elapsed),
                returns=[float(ret) for ret in returns],
                telemetry=telemetry_dict,
                artifact_dir=str(trial_dir),
                notes=[] if persist_artifacts else ["artifacts_disabled_for_scout"],
                probe_representation=probe_representation,
            )
            self._persist_result(result, summary_writer)
            return result
        finally:
            env.close()
            if summary_writer is not None:
                summary_writer.close()

    def _get_context(self, task: TaskSpec) -> _TaskContext:
        if task.checkpoint_path in self._contexts:
            return self._contexts[task.checkpoint_path]

        checkpoint_path = Path(task.checkpoint_path)
        cfg = OmegaConf.load(checkpoint_path.parent.parent / "config.yaml")
        cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        context = _TaskContext(task=task, cfg=cfg, state=state)
        self._contexts[task.checkpoint_path] = context
        return context

    def describe_task(self, task: TaskSpec, baseline_result: TrialResult) -> TaskProfile:
        context = self._get_context(task)
        env_id = str(getattr(context.cfg.env, "id", task.name))
        run_name = str(getattr(context.cfg, "run_name", ""))
        cnn_keys = tuple(getattr(context.cfg.algo.cnn_keys, "encoder", ()) or ())
        action_type = "continuous" if bool(getattr(context.cfg.algo, "is_continuous", False)) else "discrete"
        telemetry = baseline_result.telemetry or {}
        return TaskProfile(
            task_name=task.name,
            checkpoint_path=task.checkpoint_path,
            algo_name="dreamer_v3",
            env_id=env_id,
            run_name=run_name,
            action_type=action_type,
            cnn_keys=cnn_keys,
            baseline_mean_reward=baseline_result.mean_reward,
            baseline_std_reward=baseline_result.std_reward,
            baseline_clean_margin=float(telemetry.get("clean_margin_mean", 0.0)),
            task_tokens=tuple(tokenize_task_name(task.name) + tokenize_task_name(env_id)),
            probe_representation=baseline_result.probe_representation,
        )

    def _build_eval_cfg(self, base_cfg: Dict[str, Any], trial: TrialConfig, *, capture_video: bool) -> Dict[str, Any]:
        cfg = dotdict(json.loads(json.dumps(base_cfg)))
        cfg.env.capture_video = capture_video
        cfg.env.num_envs = 1
        cfg.env.sync_env = True
        cfg.fabric.devices = self.search_config.devices
        cfg.fabric.accelerator = self.search_config.accelerator
        cfg.metric.log_level = 0
        cfg.disable_grads = trial.is_baseline
        return cfg

    def _create_trial_dir(self, trial: TrialConfig, stage: str) -> Path:
        trial_dir = self.output_dir / trial.task_name / trial.attack_name / stage / trial.short_name()
        trial_dir.mkdir(parents=True, exist_ok=True)
        return trial_dir

    def _create_attack(self, trial: TrialConfig, steps: int, epsilon: float, cnn_keys: Sequence[str]) -> Any:
        kwargs = {
            "epsilon": epsilon,
            "steps": steps,
            "restarts": trial.restarts,
            "rho": trial.rho,
            "seed": trial.seed,
            "cnn_keys": tuple(cnn_keys),
        }
        if trial.attack_name == "apgd_ce":
            return APGDCrossEntropyAttack(**kwargs)
        if trial.attack_name == "apgd_dlr":
            return APGDDLRAttack(**kwargs)
        if trial.attack_name == "fab":
            return FABLinfAttack(**kwargs)
        if trial.attack_name == "square":
            return SquareAttack(**kwargs)
        raise ValueError(f"Unsupported attack '{trial.attack_name}'")

    def _actions_and_margin(self, logits_seq: Sequence[Tensor]) -> Tuple[Tuple[int, ...], float]:
        actions = []
        margins = []
        for logits in logits_seq:
            flattened = logits.view(-1, logits.shape[-1])
            action = flattened.argmax(dim=-1)[0].item()
            actions.append(int(action))
            topk = torch.topk(flattened[0], k=min(2, flattened.shape[-1])).values
            if topk.numel() == 1:
                margin = abs(topk[0].item())
            else:
                margin = (topk[0] - topk[1]).item()
            margins.append(float(margin))
        return tuple(actions), float(np.mean(margins)) if len(margins) > 0 else 0.0

    def _extract_probe_features(
        self,
        player: Any,
        obs: Dict[str, Tensor],
        action_mask: Optional[Dict[str, Tensor]],
        snapshot: Any,
    ) -> Dict[str, Tensor]:
        with torch.no_grad():
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
            policy_logits = player.actor.get_logits(actor_input, action_mask)
            logits_vector = torch.cat([logit.reshape(-1).float() for logit in policy_logits], dim=0)
            return {
                "encoder": embedded_obs.reshape(-1),
                "recurrent": next_recurrent_state.reshape(-1),
                "stochastic": next_stochastic_state.reshape(-1),
                "actor_hidden": actor_hidden.reshape(-1),
                "logits": logits_vector,
            }

    def _persist_result(self, result: TrialResult, summary_writer: Optional[SummaryWriter]) -> None:
        artifact_dir = Path(result.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifact_dir / "summary.json"
        summary_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if summary_writer is None:
            return
        summary_writer.add_scalar("mean_reward", result.mean_reward, result.num_episodes)
        summary_writer.add_scalar("std_reward", result.std_reward, result.num_episodes)
        summary_writer.add_scalar("median_reward", result.median_reward, result.num_episodes)
        summary_writer.add_scalar("elapsed_seconds", result.elapsed_seconds, result.num_episodes)
        for key, value in result.telemetry.items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                summary_writer.add_scalar(f"telemetry/{key}", value, result.num_episodes)
