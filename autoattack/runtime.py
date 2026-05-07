from __future__ import annotations

import json
import math
import os
import pathlib
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import imageio
import numpy as np
import torch
from lightning import Fabric
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from sheeprl.algos.dreamer_v3.attacks import (
    APGDCrossEntropyAttack,
    APGDDLRAttack,
    FABLinfAttack,
    SquareAttack,
    TwoStageMomentumAttack,
)
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
    action_type: Optional[str] = None
    victim_family: str = "dreamer"
    tdmpc2_agent: Any = None
    tdmpc2_modules: Any = None
    iris_agent: Any = None
    iris_modules: Any = None


@dataclass
class _TDMPC2StateSnapshot:
    prev_mean: Tensor
    t0: bool


@dataclass
class _IRISStateSnapshot:
    hx: Tensor
    cx: Tensor


class _TDMPC2Player:
    def __init__(self, agent: Any, obs_key: str = "state") -> None:
        self.agent = agent
        self.obs_key = obs_key

    def clone_states(self, t0: bool = False) -> _TDMPC2StateSnapshot:
        return _TDMPC2StateSnapshot(prev_mean=self.agent._prev_mean.detach().clone(), t0=bool(t0))

    def restore_states(self, snapshot: _TDMPC2StateSnapshot) -> None:
        self.agent._prev_mean.copy_(snapshot.prev_mean)

    def _obs_tensor(self, obs: Dict[str, Tensor]) -> Tensor:
        tensor = obs[self.obs_key]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.agent.device)

    def get_policy_outputs(
        self,
        obs: Dict[str, Tensor],
        mask: Optional[Dict[str, Tensor]] = None,
        state: Optional[_TDMPC2StateSnapshot] = None,
    ) -> Dict[str, Any]:
        del mask
        if state is not None:
            self.restore_states(state)
        obs_tensor = self._obs_tensor(obs)
        z = self.agent.model.encode(obs_tensor, task=None)
        _, info = self.agent.model.pi(z, task=None)
        mean = info["mean"]
        std = info["log_std"].exp()
        return {
            "is_continuous": True,
            "logits": (),
            "actions": (),
            "action_tensor": mean,
            "std_tensor": std,
            "dists": (Normal(mean, std),),
            "latent": z,
            "mean_tensor": mean,
            "log_std_tensor": info["log_std"],
        }

    def get_env_action(self, obs: Dict[str, Tensor], *, t0: bool) -> Tensor:
        obs_tensor = self._obs_tensor(obs).squeeze(0).detach().cpu()
        return self.agent.act(obs_tensor, t0=t0, eval_mode=False)

    def extract_probe_features(self, obs: Dict[str, Tensor], snapshot: _TDMPC2StateSnapshot) -> Dict[str, Tensor]:
        outputs = self.get_policy_outputs(obs, state=snapshot)
        latent = outputs["latent"].reshape(-1)
        mean = outputs["mean_tensor"].reshape(-1)
        log_std = outputs["log_std_tensor"].reshape(-1)
        q_value = self.agent.model.Q(outputs["latent"], outputs["mean_tensor"], task=None, return_type="avg").reshape(-1)
        actor_hidden = torch.cat((mean, log_std), dim=0)
        policy_features = torch.cat((mean, log_std, q_value), dim=0)
        return {
            "encoder": latent,
            "recurrent": latent,
            "stochastic": q_value,
            "actor_hidden": actor_hidden,
            "logits": policy_features,
        }


class _IRISPlayer:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def reset(self, n: int = 1) -> None:
        self.agent.actor_critic.reset(n=n)

    def clone_states(self) -> _IRISStateSnapshot:
        hx = self.agent.actor_critic.hx
        cx = self.agent.actor_critic.cx
        if hx is None or cx is None:
            self.reset()
            hx = self.agent.actor_critic.hx
            cx = self.agent.actor_critic.cx
        return _IRISStateSnapshot(hx=hx.detach().clone(), cx=cx.detach().clone())

    def restore_states(self, snapshot: _IRISStateSnapshot) -> None:
        self.agent.actor_critic.hx = snapshot.hx.detach().clone()
        self.agent.actor_critic.cx = snapshot.cx.detach().clone()

    def _policy_input(self, obs: Dict[str, Tensor]) -> Tensor:
        rgb = obs["rgb"]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.to(self.agent.device)
        if self.agent.actor_critic.use_original_obs:
            return rgb
        _, _, reconstructions = self.agent.tokenizer(
            rgb,
            should_preprocess=True,
            should_postprocess=True,
        )
        return torch.clamp(reconstructions, 0.0, 1.0)

    def get_policy_outputs(
        self,
        obs: Dict[str, Tensor],
        mask: Optional[Dict[str, Tensor]] = None,
        state: Optional[_IRISStateSnapshot] = None,
    ) -> Dict[str, Any]:
        del mask
        if state is not None:
            self.restore_states(state)
        policy_input = self._policy_input(obs)
        outputs = self.agent.actor_critic(policy_input)
        logits = outputs.logits_actions[:, -1]
        return {
            "is_continuous": False,
            "logits": (logits,),
            "actions": (logits.argmax(dim=-1),),
            "actor_hidden": self.agent.actor_critic.hx.detach().clone(),
            "value_tensor": outputs.means_values[:, -1].detach().clone(),
        }

    def get_env_action(
        self,
        obs: Dict[str, Tensor],
        *,
        state: _IRISStateSnapshot,
        should_sample: bool = True,
        temperature: float = 0.5,
    ) -> Tensor:
        self.restore_states(state)
        policy_input = self._policy_input(obs)
        logits = self.agent.actor_critic(policy_input).logits_actions[:, -1] / temperature
        if should_sample:
            return torch.distributions.Categorical(logits=logits).sample()
        return logits.argmax(dim=-1)

    def extract_probe_features(self, obs: Dict[str, Tensor], snapshot: _IRISStateSnapshot) -> Dict[str, Tensor]:
        self.restore_states(snapshot)
        rgb = obs["rgb"]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.to(self.agent.device)
        encoded = self.agent.tokenizer.encode(rgb, should_preprocess=True)
        policy_outputs = self.get_policy_outputs(obs, state=snapshot)
        logits_vector = torch.cat([logit.reshape(-1).float() for logit in tuple(policy_outputs.get("logits") or ())], dim=0)
        actor_hidden = policy_outputs["actor_hidden"].reshape(-1).float()
        value_tensor = policy_outputs["value_tensor"].reshape(-1).float()
        return {
            "encoder": encoded.z_quantized.reshape(-1).float(),
            "recurrent": actor_hidden,
            "stochastic": encoded.tokens.reshape(-1).float(),
            "actor_hidden": actor_hidden,
            "logits": torch.cat((logits_vector, value_tensor), dim=0),
        }


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

    def _get_algo_name(self, context: _TaskContext) -> str:
        if context.victim_family == "tdmpc2":
            return "tdmpc2"
        if context.victim_family == "iris":
            return "iris"
        algo = getattr(context.cfg, "algo", None)
        algo_name = str(getattr(algo, "name", "dreamer_v3"))
        if algo_name.startswith("p2e_dv2"):
            return "dreamer_v2"
        if algo_name.startswith("p2e_dv3"):
            return "dreamer_v3"
        return algo_name

    def _get_build_agent(self, context: _TaskContext):
        algo_name = self._get_algo_name(context)
        if algo_name == "dreamer_v2":
            from sheeprl.algos.dreamer_v2.agent import build_agent as build_agent_impl
        elif algo_name == "dreamer_v3":
            from sheeprl.algos.dreamer_v3.agent import build_agent as build_agent_impl
        else:
            raise RuntimeError(f"Unsupported algorithm '{algo_name}' for attack search.")
        return build_agent_impl

    def run_trial(
        self,
        trial: TrialConfig,
        *,
        stage: str,
        num_episodes: int,
        persist_artifacts: bool,
    ) -> TrialResult:
        context = self._get_context(TaskSpec(name=trial.task_name, checkpoint_path=trial.checkpoint_path))
        if context.victim_family == "tdmpc2":
            return self._run_tdmpc2_trial(
                context,
                trial=trial,
                stage=stage,
                num_episodes=num_episodes,
                persist_artifacts=persist_artifacts,
            )
        if context.victim_family == "iris":
            return self._run_iris_trial(
                context,
                trial=trial,
                stage=stage,
                num_episodes=num_episodes,
                persist_artifacts=persist_artifacts,
            )
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
            build_agent = self._get_build_agent(context)
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
            if not trial.is_baseline:
                self._unwrap_player_modules(player)

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
                    clean_policy = player.get_policy_outputs(torch_obs, mask=action_mask, state=snapshot)
                    clean_actions, clean_margin = self._actions_and_margin(clean_policy)
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
                        adv_policy = player.get_policy_outputs(attacked_obs, mask=action_mask, state=snapshot)
                        adv_actions, adv_margin = self._actions_and_margin(adv_policy)
                        flipped = self._actions_flipped(clean_policy, adv_policy)
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
                        real_actions = torch.stack(real_actions, -1).detach().cpu().numpy()
                    else:
                        real_actions = torch.stack(
                            [real_action.argmax(dim=-1) for real_action in real_actions], dim=-1
                        ).detach().cpu().numpy()

                    obs, reward, terminated, truncated, _ = env.step(real_actions.reshape(env.action_space.shape))
                    done = bool(terminated or truncated)
                    cumulative_rew += float(reward)

                returns.append(cumulative_rew)
                if summary_writer is not None:
                    summary_writer.add_scalar("episode_reward", cumulative_rew, episode_idx)

            elapsed = time.perf_counter() - start_time
            returns_float = [float(ret) for ret in returns]
            mean_reward = float(sum(returns_float) / len(returns_float))
            std_reward = float(statistics.pstdev(returns_float)) if len(returns_float) > 1 else 0.0
            median_reward = float(statistics.median(returns_float))
            min_reward = float(min(returns_float))
            max_reward = float(max(returns_float))
            telemetry_dict = telemetry.to_dict()
            probe_representation = None
            if probe_collector is not None:
                probe_representation = probe_collector.build(
                    source_stage=stage,
                    extras={
                        "mean_return": mean_reward,
                        "std_return": std_reward,
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
                mean_reward=mean_reward,
                std_reward=std_reward,
                median_reward=median_reward,
                min_reward=min_reward,
                max_reward=max_reward,
                elapsed_seconds=float(elapsed),
                returns=returns_float,
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

    def _load_tdmpc2_modules(self) -> Dict[str, Any]:
        root = Path(self.search_config.tdmpc2_root).resolve() / "tdmpc2"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import importlib

        common = importlib.import_module("common")
        parser = importlib.import_module("common.parser")
        return {
            "MODEL_SIZE": common.MODEL_SIZE,
            "TASK_SET": common.TASK_SET,
            "cfg_to_dataclass": parser.cfg_to_dataclass,
            "make_env": importlib.import_module("envs").make_env,
            "TDMPC2": importlib.import_module("tdmpc2").TDMPC2,
        }

    def _load_iris_modules(self) -> Dict[str, Any]:
        root = Path(self.search_config.iris_root).resolve()
        src_root = root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        import importlib
        import importlib.util
        from hydra.utils import instantiate

        agent_spec = importlib.util.spec_from_file_location("iris_agent_module", src_root / "agent.py")
        if agent_spec is None or agent_spec.loader is None:
            raise ImportError(f"Failed to load IRIS agent module from {src_root / 'agent.py'}")
        iris_agent_module = importlib.util.module_from_spec(agent_spec)
        agent_spec.loader.exec_module(iris_agent_module)

        return {
            "instantiate": instantiate,
            "Agent": iris_agent_module.Agent,
            "ActorCritic": importlib.import_module("models.actor_critic").ActorCritic,
            "WorldModel": importlib.import_module("models.world_model").WorldModel,
            "make_atari": importlib.import_module("envs.wrappers").make_atari,
        }

    def _build_tdmpc2_cfg(self, task: TaskSpec) -> Dict[str, Any]:
        modules = self._load_tdmpc2_modules()
        cfg = OmegaConf.load(Path(self.search_config.tdmpc2_root) / "tdmpc2" / "config.yaml")
        cfg.task = task.name
        cfg.obs = self.search_config.tdmpc2_obs
        cfg.model_size = 5
        cfg.checkpoint = task.checkpoint_path
        cfg.data_dir = ""
        cfg.enable_wandb = False
        cfg.wandb_project = ""
        cfg.wandb_entity = ""
        cfg.save_video = False
        cfg.save_agent = False
        cfg.compile = False
        cfg.seed = self.search_config.seed
        cfg.exp_name = "wma_tdmpc2_attack"
        cfg.work_dir = (
            Path(self.search_config.output_dir)
            / "_tdmpc2_eval"
            / task.name
            / str(self.search_config.seed)
            / cfg.exp_name
        )
        cfg.task_title = cfg.task.replace("-", " ").title()
        cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

        model_size = int(cfg.model_size)
        if model_size not in modules["MODEL_SIZE"]:
            raise ValueError(
                f"Invalid TD-MPC2 model size '{model_size}'. "
                f"Expected one of {sorted(modules['MODEL_SIZE'].keys())}."
            )
        for key, value in modules["MODEL_SIZE"][model_size].items():
            cfg[key] = value

        cfg.multitask = cfg.task in modules["TASK_SET"]
        if cfg.multitask:
            cfg.task_title = cfg.task.upper()
            cfg.task_dim = 96 if cfg.task == "mt80" or model_size in {1, 317} else 64
        else:
            cfg.task_dim = 0
        cfg.tasks = modules["TASK_SET"].get(cfg.task, [cfg.task])
        parsed = modules["cfg_to_dataclass"](cfg)
        return parsed, modules

    def _build_iris_cfg(self, task: TaskSpec) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        modules = self._load_iris_modules()
        root = Path(self.search_config.iris_root).resolve()
        tokenizer_cfg = OmegaConf.load(root / "config" / "tokenizer" / "default.yaml")
        world_model_cfg = OmegaConf.load(root / "config" / "world_model" / "default.yaml")
        actor_critic_cfg = OmegaConf.load(root / "config" / "actor_critic" / "default.yaml")
        env_cfg = OmegaConf.load(root / "config" / "env" / "default.yaml")
        env_id = task.name if task.name.endswith("NoFrameskip-v4") else f"{task.name}NoFrameskip-v4"
        env_cfg.test.id = env_id
        return {
            "tokenizer": tokenizer_cfg,
            "world_model": world_model_cfg,
            "actor_critic": actor_critic_cfg,
            "env": env_cfg,
            "env_id": env_id,
        }, modules

    def _run_tdmpc2_trial(
        self,
        context: _TaskContext,
        *,
        trial: TrialConfig,
        stage: str,
        num_episodes: int,
        persist_artifacts: bool,
    ) -> TrialResult:
        trial_dir = self._create_trial_dir(trial, stage)
        summary_writer = SummaryWriter(trial_dir) if persist_artifacts and self.search_config.tensorboard_for_confirm else None
        probe_collector = (
            ProbeFeatureCollector(self.search_config.experience_probe_max_steps)
            if self.search_config.experience_retrieval_mode in ("latent", "hybrid")
            else None
        )
        env = context.tdmpc2_modules["make_env"](context.cfg)
        player = _TDMPC2Player(context.tdmpc2_agent, obs_key=self.search_config.tdmpc2_obs)
        try:
            returns = []
            telemetry = TrialTelemetry()
            start_time = time.perf_counter()
            for episode_idx in range(num_episodes):
                del episode_idx
                obs = env.reset()
                done = False
                cumulative_rew = 0.0
                step_idx = 0
                frames = None
                while not done:
                    obs_dict = {self.search_config.tdmpc2_obs: obs.unsqueeze(0).to(context.tdmpc2_agent.device)}
                    snapshot = player.clone_states(t0=(step_idx == 0))
                    clean_policy = player.get_policy_outputs(obs_dict, state=snapshot)
                    clean_actions, clean_margin = self._actions_and_margin(clean_policy)
                    clean_probe_features = None
                    if probe_collector is not None and probe_collector.should_collect():
                        clean_probe_features = player.extract_probe_features(obs_dict, snapshot)

                    attack_steps = 0
                    attack_epsilon = 0.0
                    attacked_obs = obs_dict
                    adv_actions = clean_actions
                    adv_margin = clean_margin
                    flipped = False
                    adv_probe_features = clean_probe_features
                    if not trial.is_baseline:
                        attack_steps, attack_epsilon = self._allocator.allocate(trial, clean_margin)
                        attacker = self._create_attack(
                            trial,
                            attack_steps,
                            attack_epsilon,
                            (self.search_config.tdmpc2_obs,),
                            clip_min=None,
                            clip_max=None,
                        )
                        attacked_obs = attacker.perturb(player, obs_dict, None, greedy=False)
                        adv_policy = player.get_policy_outputs(attacked_obs, state=snapshot)
                        adv_actions, adv_margin = self._actions_and_margin(adv_policy)
                        flipped = self._actions_flipped(clean_policy, adv_policy)
                        if probe_collector is not None and probe_collector.should_collect():
                            adv_probe_features = player.extract_probe_features(attacked_obs, snapshot)
                    telemetry.update(clean_margin, adv_margin, flipped, attack_steps, attack_epsilon)
                    if probe_collector is not None and clean_probe_features is not None:
                        probe_collector.update(clean_probe_features, adv_probe_features)

                    acting_obs = attacked_obs if not trial.is_baseline else obs_dict
                    real_action = player.get_env_action(acting_obs, t0=(step_idx == 0))
                    obs, reward, done, info = env.step(real_action)
                    del info
                    cumulative_rew += float(reward)
                    step_idx += 1
                returns.append(cumulative_rew)
                if summary_writer is not None:
                    summary_writer.add_scalar("episode_reward", cumulative_rew, len(returns) - 1)

            elapsed = time.perf_counter() - start_time
            returns_float = [float(ret) for ret in returns]
            mean_reward = float(sum(returns_float) / len(returns_float))
            std_reward = float(statistics.pstdev(returns_float)) if len(returns_float) > 1 else 0.0
            median_reward = float(statistics.median(returns_float))
            min_reward = float(min(returns_float))
            max_reward = float(max(returns_float))
            telemetry_dict = telemetry.to_dict()
            probe_representation = None
            if probe_collector is not None:
                probe_representation = probe_collector.build(
                    source_stage=stage,
                    extras={
                        "mean_return": mean_reward,
                        "std_return": std_reward,
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
                mean_reward=mean_reward,
                std_reward=std_reward,
                median_reward=median_reward,
                min_reward=min_reward,
                max_reward=max_reward,
                elapsed_seconds=float(elapsed),
                returns=returns_float,
                telemetry=telemetry_dict,
                artifact_dir=str(trial_dir),
                notes=[] if persist_artifacts else ["artifacts_disabled_for_scout"],
                probe_representation=probe_representation,
            )
            self._persist_result(result, summary_writer)
            return result
        finally:
            if summary_writer is not None:
                summary_writer.close()

    def _run_iris_trial(
        self,
        context: _TaskContext,
        *,
        trial: TrialConfig,
        stage: str,
        num_episodes: int,
        persist_artifacts: bool,
    ) -> TrialResult:
        trial_dir = self._create_trial_dir(trial, stage)
        summary_writer = SummaryWriter(trial_dir) if persist_artifacts and self.search_config.tensorboard_for_confirm else None
        probe_collector = (
            ProbeFeatureCollector(self.search_config.experience_probe_max_steps)
            if self.search_config.experience_retrieval_mode in ("latent", "hybrid")
            else None
        )
        env_cfg = context.cfg["env"]
        env = context.iris_modules["make_atari"](
            id=env_cfg.test.id,
            size=env_cfg.test.size,
            max_episode_steps=env_cfg.test.max_episode_steps,
            noop_max=env_cfg.test.noop_max,
            frame_skip=env_cfg.test.frame_skip,
            done_on_life_loss=env_cfg.test.done_on_life_loss,
            clip_reward=env_cfg.test.clip_reward,
        )
        player = _IRISPlayer(context.iris_agent)
        device = context.iris_agent.device
        try:
            returns = []
            telemetry = TrialTelemetry()
            start_time = time.perf_counter()
            for episode_idx in range(num_episodes):
                del episode_idx
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                player.reset(n=1)
                done = False
                cumulative_rew = 0.0
                while not done:
                    rgb = torch.as_tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0).div(255.0)
                    obs_dict = {"rgb": rgb}
                    snapshot = player.clone_states()
                    clean_policy = player.get_policy_outputs(obs_dict, state=snapshot)
                    clean_actions, clean_margin = self._actions_and_margin(clean_policy)
                    clean_probe_features = None
                    if probe_collector is not None and probe_collector.should_collect():
                        clean_probe_features = player.extract_probe_features(obs_dict, snapshot)

                    attack_steps = 0
                    attack_epsilon = 0.0
                    attacked_obs = obs_dict
                    adv_actions = clean_actions
                    adv_margin = clean_margin
                    flipped = False
                    adv_probe_features = clean_probe_features
                    if not trial.is_baseline:
                        attack_steps, attack_epsilon = self._allocator.allocate(trial, clean_margin)
                        attacker = self._create_attack(
                            trial,
                            attack_steps,
                            attack_epsilon,
                            ("rgb",),
                            clip_min=0.0,
                            clip_max=1.0,
                        )
                        attacked_obs = attacker.perturb(player, obs_dict, None, greedy=False)
                        adv_policy = player.get_policy_outputs(attacked_obs, state=snapshot)
                        adv_actions, adv_margin = self._actions_and_margin(adv_policy)
                        flipped = self._actions_flipped(clean_policy, adv_policy)
                        if probe_collector is not None and probe_collector.should_collect():
                            adv_probe_features = player.extract_probe_features(attacked_obs, snapshot)
                    telemetry.update(clean_margin, adv_margin, flipped, attack_steps, attack_epsilon)
                    if probe_collector is not None and clean_probe_features is not None:
                        probe_collector.update(clean_probe_features, adv_probe_features)

                    acting_obs = attacked_obs if not trial.is_baseline else obs_dict
                    real_action = player.get_env_action(acting_obs, state=snapshot, should_sample=True, temperature=0.5)
                    step_out = env.step(int(real_action.item()))
                    if len(step_out) == 5:
                        obs, reward, terminated, truncated, _ = step_out
                        done = bool(terminated or truncated)
                    else:
                        obs, reward, done, _ = step_out
                        done = bool(done)
                    cumulative_rew += float(reward)

                returns.append(cumulative_rew)
                if summary_writer is not None:
                    summary_writer.add_scalar("episode_reward", cumulative_rew, len(returns) - 1)

            elapsed = time.perf_counter() - start_time
            returns_float = [float(ret) for ret in returns]
            mean_reward = float(sum(returns_float) / len(returns_float))
            std_reward = float(statistics.pstdev(returns_float)) if len(returns_float) > 1 else 0.0
            median_reward = float(statistics.median(returns_float))
            min_reward = float(min(returns_float))
            max_reward = float(max(returns_float))
            telemetry_dict = telemetry.to_dict()
            probe_representation = None
            if probe_collector is not None:
                probe_representation = probe_collector.build(
                    source_stage=stage,
                    extras={
                        "mean_return": mean_reward,
                        "std_return": std_reward,
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
                mean_reward=mean_reward,
                std_reward=std_reward,
                median_reward=median_reward,
                min_reward=min_reward,
                max_reward=max_reward,
                elapsed_seconds=float(elapsed),
                returns=returns_float,
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

        if self.search_config.victim_family == "tdmpc2":
            cfg, modules = self._build_tdmpc2_cfg(task)
            env = modules["make_env"](cfg)
            close_env = getattr(env, "close", None)
            if callable(close_env):
                try:
                    close_env()
                except AttributeError:
                    pass
            agent = modules["TDMPC2"](cfg)
            agent.load(task.checkpoint_path)
            context = _TaskContext(
                task=task,
                cfg=cfg,
                state={},
                action_type="continuous",
                victim_family="tdmpc2",
                tdmpc2_agent=agent,
                tdmpc2_modules=modules,
            )
            self._contexts[task.checkpoint_path] = context
            return context

        if self.search_config.victim_family == "iris":
            cfg, modules = self._build_iris_cfg(task)
            env_cfg = cfg["env"]
            env = modules["make_atari"](
                id=env_cfg.test.id,
                size=env_cfg.test.size,
                max_episode_steps=env_cfg.test.max_episode_steps,
                noop_max=env_cfg.test.noop_max,
                frame_skip=env_cfg.test.frame_skip,
                done_on_life_loss=env_cfg.test.done_on_life_loss,
                clip_reward=env_cfg.test.clip_reward,
            )
            num_actions = env.action_space.n
            env.close()
            tokenizer = modules["instantiate"](cfg["tokenizer"])
            world_model = modules["WorldModel"](
                obs_vocab_size=tokenizer.vocab_size,
                act_vocab_size=num_actions,
                config=modules["instantiate"](cfg["world_model"]),
            )
            actor_critic_kwargs = OmegaConf.to_container(cfg["actor_critic"], resolve=True)
            actor_critic = modules["ActorCritic"](**actor_critic_kwargs, act_vocab_size=num_actions)
            device = torch.device(self.fabric.device)
            agent = modules["Agent"](tokenizer, world_model, actor_critic).to(device)
            agent.load(Path(task.checkpoint_path), device=device)
            context = _TaskContext(
                task=task,
                cfg=cfg,
                state={},
                action_type="discrete",
                victim_family="iris",
                iris_agent=agent,
                iris_modules=modules,
            )
            self._contexts[task.checkpoint_path] = context
            return context

        checkpoint_path = Path(task.checkpoint_path)
        cfg = OmegaConf.load(self._resolve_checkpoint_config_path(checkpoint_path))
        cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        context = _TaskContext(task=task, cfg=cfg, state=state, victim_family="dreamer")
        self._contexts[task.checkpoint_path] = context
        return context

    def _resolve_checkpoint_config_path(self, checkpoint_path: Path) -> Path:
        candidates = (
            checkpoint_path.parent.parent / "config.yaml",
            checkpoint_path.parent.parent.parent / ".hydra" / "config.yaml",
            checkpoint_path.parent.parent.parent / "config.yaml",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not find checkpoint config for "
            f"{checkpoint_path}. Checked: {', '.join(str(path) for path in candidates)}"
        )

    def describe_task(self, task: TaskSpec, baseline_result: TrialResult) -> TaskProfile:
        context = self._get_context(task)
        if context.victim_family == "tdmpc2":
            env_id = task.name
            run_name = Path(task.checkpoint_path).name
            cnn_keys = (self.search_config.tdmpc2_obs,)
        elif context.victim_family == "iris":
            env_id = context.cfg["env_id"]
            run_name = Path(task.checkpoint_path).name
            cnn_keys = ("rgb",)
        else:
            env_id = str(getattr(context.cfg.env, "id", task.name))
            run_name = str(getattr(context.cfg, "run_name", ""))
            cnn_keys = tuple(getattr(context.cfg.algo.cnn_keys, "encoder", ()) or ())
        action_type = self._get_action_type(context)
        telemetry = baseline_result.telemetry or {}
        return TaskProfile(
            task_name=task.name,
            checkpoint_path=task.checkpoint_path,
            algo_name="tdmpc2" if context.victim_family == "tdmpc2" else self._get_algo_name(context),
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

    def _get_action_type(self, context: _TaskContext) -> str:
        if context.action_type is not None:
            return context.action_type

        if context.victim_family == "tdmpc2":
            context.action_type = "continuous"
            return context.action_type
        if context.victim_family == "iris":
            context.action_type = "discrete"
            return context.action_type

        # Infer action type from the actual evaluation environment instead of
        # relying on checkpoint config fields that may be absent or stale.
        env = make_env(context.cfg, context.cfg.seed, 0, str(self.output_dir), "test", vector_env_idx=0)()
        try:
            context.action_type = "continuous" if isinstance(env.action_space, gym.spaces.Box) else "discrete"
        finally:
            env.close()
        return context.action_type

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

    def _create_attack(
        self,
        trial: TrialConfig,
        steps: int,
        epsilon: float,
        cnn_keys: Sequence[str],
        *,
        clip_min: Optional[float] = -0.5,
        clip_max: Optional[float] = 0.5,
    ) -> Any:
        kwargs = {
            "epsilon": epsilon,
            "steps": steps,
            "restarts": trial.restarts,
            "rho": trial.rho,
            "seed": trial.seed,
            "cnn_keys": tuple(cnn_keys),
            "clip_min": clip_min,
            "clip_max": clip_max,
        }
        if trial.attack_name == "apgd_ce":
            return APGDCrossEntropyAttack(**kwargs)
        if trial.attack_name == "apgd_dlr":
            return APGDDLRAttack(**kwargs)
        if trial.attack_name == "fab":
            return FABLinfAttack(**kwargs)
        if trial.attack_name == "square":
            return SquareAttack(**kwargs)
        if trial.attack_name == "two_stage":
            return TwoStageMomentumAttack(**kwargs)
        raise ValueError(f"Unsupported attack '{trial.attack_name}'")

    def _actions_and_margin(self, policy_outputs: Dict[str, Any]) -> Tuple[Tuple[float, ...], float]:
        if bool(policy_outputs.get("is_continuous", False)):
            action_tensor = policy_outputs.get("action_tensor")
            std_tensor = policy_outputs.get("std_tensor")
            if action_tensor is None or std_tensor is None:
                raise RuntimeError("Continuous policy outputs must expose action_tensor and std_tensor.")
            actions = tuple(float(v) for v in action_tensor.detach().reshape(-1).cpu().tolist())
            confidence = 1.0 / std_tensor.detach().abs().mean().clamp_min(1e-6)
            return actions, float(confidence.item())

        actions = []
        margins = []
        for logits in tuple(policy_outputs.get("logits") or ()):
            flattened = logits.view(-1, logits.shape[-1])
            action = flattened.argmax(dim=-1)[0].item()
            actions.append(float(action))
            topk = torch.topk(flattened[0], k=min(2, flattened.shape[-1])).values
            if topk.numel() == 1:
                margin = abs(topk[0].item())
            else:
                margin = (topk[0] - topk[1]).item()
            margins.append(float(margin))
        return tuple(actions), float(np.mean(margins)) if len(margins) > 0 else 0.0

    def _actions_flipped(
        self,
        clean_policy: Dict[str, Any],
        adv_policy: Dict[str, Any],
        continuous_threshold: float = 0.05,
    ) -> bool:
        if bool(clean_policy.get("is_continuous", False)):
            clean_action = clean_policy.get("action_tensor")
            adv_action = adv_policy.get("action_tensor")
            if clean_action is None or adv_action is None:
                return False
            return bool((adv_action - clean_action).abs().mean().item() > continuous_threshold)
        clean_actions = tuple(policy.argmax(dim=-1) for policy in tuple(clean_policy.get("logits") or ()))
        adv_actions = tuple(policy.argmax(dim=-1) for policy in tuple(adv_policy.get("logits") or ()))
        return any(a.ne(c).any().item() for a, c in zip(adv_actions, clean_actions))

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
            policy_outputs = player.actor.get_policy_outputs(actor_input, action_mask)
            if bool(policy_outputs.get("is_continuous", False)):
                action_tensor = policy_outputs.get("action_tensor")
                std_tensor = policy_outputs.get("std_tensor")
                if action_tensor is None or std_tensor is None:
                    raise RuntimeError("Continuous policy outputs must expose action and std tensors.")
                logits_vector = torch.cat((action_tensor.reshape(-1), std_tensor.reshape(-1)), dim=0).float()
            else:
                policy_logits = tuple(policy_outputs.get("logits") or ())
                logits_vector = torch.cat([logit.reshape(-1).float() for logit in policy_logits], dim=0)
            return {
                "encoder": embedded_obs.reshape(-1),
                "recurrent": next_recurrent_state.reshape(-1),
                "stochastic": next_stochastic_state.reshape(-1),
                "actor_hidden": actor_hidden.reshape(-1),
                "logits": logits_vector,
            }

    def _unwrap_player_modules(self, player: Any) -> None:
        def _unwrap(module: Any) -> Any:
            return getattr(module, "module", module)

        player.encoder = _unwrap(player.encoder)
        player.rssm.recurrent_model = _unwrap(player.rssm.recurrent_model)
        if getattr(player.rssm, "transition_model", None) is not None:
            player.rssm.transition_model = _unwrap(player.rssm.transition_model)
        player.rssm.representation_model = _unwrap(player.rssm.representation_model)
        player.actor = _unwrap(player.actor)

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
