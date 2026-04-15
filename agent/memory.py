from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agent.schema import ExperienceEntry, ProbeRepresentation, TaskProfile


def tokenize_task_name(value: str) -> List[str]:
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", value)
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [token for token in tokens if token and token not in {"v4", "v5", "noframeskip"}]


@dataclass
class RetrievedExperience:
    entry: ExperienceEntry
    score: float

    def to_dict(self) -> Dict[str, object]:
        return {"score": self.score, "entry": self.entry.to_dict()}


class ExperienceMemoryStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Optional[List[ExperienceEntry]] = None

    def entries(self) -> List[ExperienceEntry]:
        if self._entries is not None:
            return self._entries
        if not self.path.exists():
            self._entries = []
            return self._entries
        loaded: List[ExperienceEntry] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                loaded.append(ExperienceEntry.from_dict(json.loads(line)))
        self._entries = loaded
        return self._entries

    def append(self, entry: ExperienceEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        if self._entries is not None:
            self._entries.append(entry)

    def extend(self, entries: Iterable[ExperienceEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def retrieve(
        self,
        task_profile: TaskProfile,
        attack_name: str,
        *,
        limit: int = 6,
        mode: str = "structured",
        query_probe: Optional[ProbeRepresentation] = None,
        latent_projection: str = "pca",
        latent_dim: int = 16,
        hybrid_weight: float = 0.6,
    ) -> List[RetrievedExperience]:
        normalized_mode = str(mode).strip().lower()
        if limit <= 0 or normalized_mode in {"none", "disabled", "off"}:
            return []
        if normalized_mode == "latent":
            latent = self._retrieve_latent(
                task_profile,
                attack_name,
                limit=limit,
                query_probe=query_probe,
                latent_projection=latent_projection,
                latent_dim=latent_dim,
            )
            if len(latent) > 0:
                return latent
            return self._retrieve_structured(task_profile, attack_name, limit=limit)
        if normalized_mode == "hybrid":
            return self._retrieve_hybrid(
                task_profile,
                attack_name,
                limit=limit,
                query_probe=query_probe,
                latent_projection=latent_projection,
                latent_dim=latent_dim,
                hybrid_weight=hybrid_weight,
            )
        return self._retrieve_structured(task_profile, attack_name, limit=limit)

    def _retrieve_structured(
        self,
        task_profile: TaskProfile,
        attack_name: str,
        *,
        limit: int = 6,
    ) -> List[RetrievedExperience]:
        scored: List[RetrievedExperience] = []
        for entry in self.entries():
            if entry.attack_name != attack_name:
                continue
            score = self._score(task_profile, entry.task_profile)
            if score <= 0:
                continue
            scored.append(RetrievedExperience(entry=entry, score=score))
        scored.sort(key=lambda item: (-item.score, -item.entry.utility, item.entry.created_at), reverse=False)
        return scored[:limit]

    def _retrieve_latent(
        self,
        task_profile: TaskProfile,
        attack_name: str,
        *,
        limit: int = 6,
        query_probe: Optional[ProbeRepresentation] = None,
        latent_projection: str = "pca",
        latent_dim: int = 16,
    ) -> List[RetrievedExperience]:
        probe = query_probe or task_profile.probe_representation
        query_vector = self._probe_vector(probe)
        if query_vector is None:
            return []

        candidate_entries: List[ExperienceEntry] = []
        candidate_vectors: List[np.ndarray] = []
        for entry in self.entries():
            if entry.attack_name != attack_name:
                continue
            vector = self._experience_probe_vector(entry)
            if vector is None:
                continue
            candidate_entries.append(entry)
            candidate_vectors.append(vector)

        if len(candidate_entries) == 0:
            return []

        query_latent, stored_latents = self._project_latent_space(
            query_vector,
            candidate_vectors,
            latent_projection=latent_projection,
            latent_dim=latent_dim,
        )
        scored: List[RetrievedExperience] = []
        for entry, stored_latent in zip(candidate_entries, stored_latents):
            cosine = self._cosine_similarity(query_latent, stored_latent)
            score = 12.0 * (0.5 * (cosine + 1.0))
            score += 0.5 * np.tanh(float(entry.utility))
            if score <= 0:
                continue
            scored.append(RetrievedExperience(entry=entry, score=float(score)))
        scored.sort(key=lambda item: (-item.score, -item.entry.utility, item.entry.created_at), reverse=False)
        return scored[:limit]

    def _retrieve_hybrid(
        self,
        task_profile: TaskProfile,
        attack_name: str,
        *,
        limit: int = 6,
        query_probe: Optional[ProbeRepresentation] = None,
        latent_projection: str = "pca",
        latent_dim: int = 16,
        hybrid_weight: float = 0.6,
    ) -> List[RetrievedExperience]:
        hybrid_weight = min(max(float(hybrid_weight), 0.0), 1.0)
        structured_entries = self._retrieve_structured(task_profile, attack_name, limit=max(limit * 4, limit))
        latent_entries = self._retrieve_latent(
            task_profile,
            attack_name,
            limit=max(limit * 4, limit),
            query_probe=query_probe,
            latent_projection=latent_projection,
            latent_dim=latent_dim,
        )
        structured_map = {id(item.entry): item for item in structured_entries}
        latent_map = {id(item.entry): item for item in latent_entries}
        if len(latent_map) == 0:
            return structured_entries[:limit]

        merged: List[RetrievedExperience] = []
        visited = set(structured_map) | set(latent_map)
        for key in visited:
            latent_score = latent_map.get(key).score if key in latent_map else 0.0
            structured_score = structured_map.get(key).score if key in structured_map else 0.0
            combined = hybrid_weight * latent_score + (1.0 - hybrid_weight) * structured_score
            entry = latent_map.get(key, structured_map.get(key)).entry
            merged.append(RetrievedExperience(entry=entry, score=float(combined)))
        merged.sort(key=lambda item: (-item.score, -item.entry.utility, item.entry.created_at), reverse=False)
        return merged[:limit]

    def build_entry(
        self,
        *,
        task_profile: TaskProfile,
        attack_name: str,
        best_config: Dict[str, object],
        result_summary: Dict[str, object],
        utility: float,
        source_run_dir: str,
        notes: Optional[Sequence[str]] = None,
        probe_representation: Optional[ProbeRepresentation] = None,
    ) -> ExperienceEntry:
        return ExperienceEntry(
            task_profile=task_profile,
            attack_name=attack_name,
            best_config=dict(best_config),
            result_summary=dict(result_summary),
            utility=float(utility),
            source_run_dir=source_run_dir,
            created_at=datetime.now().isoformat(timespec="seconds"),
            notes=list(notes or ()),
            probe_representation=probe_representation or task_profile.probe_representation,
        )

    def _score(self, query: TaskProfile, stored: TaskProfile) -> float:
        score = 0.0
        if query.algo_name and stored.algo_name == query.algo_name:
            score += 2.0
        if query.action_type and stored.action_type == query.action_type:
            score += 1.0
        if query.env_id and stored.env_id == query.env_id:
            score += 4.0
        query_tokens = set(query.task_tokens or tokenize_task_name(query.task_name))
        stored_tokens = set(stored.task_tokens or tokenize_task_name(stored.task_name))
        if query_tokens and stored_tokens:
            overlap = len(query_tokens & stored_tokens)
            union = len(query_tokens | stored_tokens)
            score += 3.0 * (overlap / union)
        if query.run_name and stored.run_name and query.run_name == stored.run_name:
            score += 1.0
        query_margin = query.baseline_clean_margin
        stored_margin = stored.baseline_clean_margin
        if query_margin is not None and stored_margin is not None:
            score += max(0.0, 1.0 - abs(query_margin - stored_margin) / 3.0)
        return score

    def _experience_probe_vector(self, entry: ExperienceEntry) -> Optional[np.ndarray]:
        return self._probe_vector(entry.probe_representation or entry.task_profile.probe_representation)

    def _probe_vector(self, probe: Optional[ProbeRepresentation]) -> Optional[np.ndarray]:
        if probe is None:
            return None
        values = np.asarray(list(probe.teacher_vector), dtype=np.float32)
        if values.size == 0:
            return None
        if np.any(~np.isfinite(values)):
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values

    def _project_latent_space(
        self,
        query_vector: np.ndarray,
        stored_vectors: Sequence[np.ndarray],
        *,
        latent_projection: str,
        latent_dim: int,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if len(stored_vectors) == 0:
            return query_vector, []
        matrix = self._pad_matrix([*stored_vectors, query_vector])
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        projection = str(latent_projection).strip().lower()
        if projection == "pca" and centered.shape[0] > 1 and centered.shape[1] > 1:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            target_dim = max(1, min(int(latent_dim), vh.shape[0], centered.shape[0] - 1, centered.shape[1]))
            if target_dim > 0:
                basis = vh[:target_dim].T
                projected = centered @ basis
            else:
                projected = centered
        else:
            projected = centered
        stored_projected = [projected[index] for index in range(len(stored_vectors))]
        query_projected = projected[len(stored_vectors)]
        return query_projected, stored_projected

    def _pad_matrix(self, rows: Sequence[np.ndarray]) -> np.ndarray:
        width = max(row.shape[0] for row in rows)
        padded = np.zeros((len(rows), width), dtype=np.float32)
        for index, row in enumerate(rows):
            padded[index, : row.shape[0]] = row
        return padded

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
