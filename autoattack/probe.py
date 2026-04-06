from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from agent.schema import ProbeRepresentation


def _flatten_feature(tensor: Tensor) -> Tensor:
    return tensor.detach().reshape(-1).float().cpu()


def _tensor_descriptors(tensor: Tensor) -> Dict[str, float]:
    if tensor.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "abs_mean": 0.0,
            "l2": 0.0,
        }
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        "abs_mean": float(tensor.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(tensor).item() / math.sqrt(float(tensor.numel()))),
    }


@dataclass
class _ScalarMoments:
    count: int = 0
    value_sum: float = 0.0
    square_sum: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.value_sum += value
        self.square_sum += value * value

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value_sum / self.count

    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        mean = self.mean()
        variance = max(0.0, self.square_sum / self.count - mean * mean)
        return math.sqrt(variance)


class ProbeFeatureCollector:
    FEATURE_ORDER = ("encoder", "recurrent", "stochastic", "actor_hidden", "logits")
    DESCRIPTOR_ORDER = ("mean", "std", "abs_mean", "l2")
    CHANNEL_ORDER = ("clean", "adv", "delta")

    def __init__(self, max_steps: int = 32) -> None:
        self.max_steps = max(1, int(max_steps))
        self._steps = 0
        self._stats: Dict[str, Dict[str, _ScalarMoments]] = {}

    @property
    def steps(self) -> int:
        return self._steps

    def should_collect(self) -> bool:
        return self._steps < self.max_steps

    def update(self, clean_features: Dict[str, Tensor], adv_features: Optional[Dict[str, Tensor]] = None) -> None:
        if not self.should_collect():
            return
        self._steps += 1
        for feature_name in self.FEATURE_ORDER:
            clean_tensor = _flatten_feature(clean_features.get(feature_name, torch.zeros(1)))
            adv_tensor = clean_tensor if adv_features is None else _flatten_feature(
                adv_features.get(feature_name, clean_features.get(feature_name, torch.zeros(1)))
            )
            delta_tensor = adv_tensor - clean_tensor
            self._record(feature_name, "clean", _tensor_descriptors(clean_tensor))
            self._record(feature_name, "adv", _tensor_descriptors(adv_tensor))
            self._record(feature_name, "delta", _tensor_descriptors(delta_tensor))

    def build(self, source_stage: str, extras: Optional[Dict[str, float]] = None) -> Optional[ProbeRepresentation]:
        if self._steps == 0:
            return None
        feature_stats: Dict[str, Dict[str, float]] = {}
        teacher_vector = []
        for feature_name in self.FEATURE_ORDER:
            channel_stats = self._stats.get(feature_name, {})
            feature_stats[feature_name] = {}
            for channel_name in self.CHANNEL_ORDER:
                for descriptor in self.DESCRIPTOR_ORDER:
                    key = f"{channel_name}_{descriptor}"
                    moments = channel_stats.get(key, _ScalarMoments())
                    mean_value = moments.mean()
                    std_value = moments.std()
                    feature_stats[feature_name][f"{key}_mean"] = mean_value
                    feature_stats[feature_name][f"{key}_std"] = std_value
                    teacher_vector.extend((mean_value, std_value))

        extras = extras or {}
        if len(extras) > 0:
            feature_stats["extras"] = {}
            for key in sorted(extras):
                value = float(extras[key])
                feature_stats["extras"][key] = value
                teacher_vector.append(value)

        return ProbeRepresentation(
            version="v1",
            source_stage=source_stage,
            feature_stats=feature_stats,
            teacher_vector=tuple(float(value) for value in teacher_vector),
            compression="summary_stats",
            num_samples=self._steps,
        )

    def _record(self, feature_name: str, channel_name: str, descriptors: Dict[str, float]) -> None:
        feature_slot = self._stats.setdefault(feature_name, {})
        for descriptor_name, value in descriptors.items():
            feature_slot.setdefault(f"{channel_name}_{descriptor_name}", _ScalarMoments()).update(float(value))
