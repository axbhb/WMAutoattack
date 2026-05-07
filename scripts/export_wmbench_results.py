#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/share/guozhix/WMAutoattack")
LOG_ROOT = ROOT / "logs" / "attack_search"
EXPORT_DATE = datetime.utcnow().strftime("%Y%m%d")
EXPORT_DIR = ROOT / f"wmbench_export_{EXPORT_DATE}"
ZIP_PATH = ROOT / f"wmbench_export_{EXPORT_DATE}.zip"

ATTACK_NAMES = {"baseline", "apgd_ce", "apgd_dlr", "fab", "two_stage", "square"}
STAGE_ORDER = {"baseline": 0, "scout": 1, "confirm": 2}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model: str
    suite: str
    method: str
    initialization: str
    retrieval_mode: str
    llm_model: str
    raw_group: str
    roots: tuple[Path, ...]
    notes: str = ""


EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec(
        "dv3_atari_main_random",
        "dreamerv3",
        "atari",
        "Random-Init",
        "random",
        "none",
        "heuristic",
        "dreamerv3_atari",
        (LOG_ROOT / "random_init_no_rag_26_tasks_2944",),
        "26-task Atari DreamerV3 random-init baseline; 4 attacks (no square).",
    ),
    ExperimentSpec(
        "dv3_atari_main_latent",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "heuristic",
        "dreamerv3_atari",
        (LOG_ROOT / "probe_scale_26_tasks_2945",),
        "26-task Atari DreamerV3 latent retrieval; 4 attacks (no square).",
    ),
    ExperimentSpec(
        "dv3_atari_claudini",
        "dreamerv3",
        "atari",
        "Claudini",
        "random",
        "none",
        "heuristic",
        "dreamerv3_atari",
        (LOG_ROOT / "claudini_style_atari_20_tasks_2905",),
        "Directory name says 20_tasks but actual result set covers 26 Atari tasks; 4 attacks (no square).",
    ),
    ExperimentSpec(
        "dv3_dmc_main_random",
        "dreamerv3",
        "dmc",
        "Random-Init",
        "random",
        "none",
        "heuristic",
        "dreamerv3_dmc",
        (LOG_ROOT / "random_init_no_rag_dmc_20_tasks_aligned_2964",),
    ),
    ExperimentSpec(
        "dv3_dmc_main_latent",
        "dreamerv3",
        "dmc",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "heuristic",
        "dreamerv3_dmc",
        (LOG_ROOT / "probe_scale_dmc_20_tasks_aligned_2966",),
        "Includes resumed run output written back into the same directory.",
    ),
    ExperimentSpec(
        "dv3_dmc_claudini",
        "dreamerv3",
        "dmc",
        "Claudini",
        "random",
        "none",
        "heuristic",
        "dreamerv3_dmc",
        (LOG_ROOT / "claudini_style_dmc_20_tasks_2965",),
    ),
    ExperimentSpec(
        "dv2_atari_cross_model",
        "dreamerv2",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "heuristic",
        "dreamerv2_atari",
        (
            LOG_ROOT / "dreamerv2_alien_latent_3230",
            LOG_ROOT / "atari_dreamerv2_latent_queue_rerun",
        ),
        "Contains one complete Alien validation run plus a full-queue rerun snapshot that may still be incomplete.",
    ),
    ExperimentSpec(
        "rgar_atari_full",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "heuristic",
        "rgar_ablation_atari",
        (LOG_ROOT / "probe_scale_26_tasks_2945",),
        "Analytical alias of dv3_atari_main_latent for RGAR-vs-random tables.",
    ),
    ExperimentSpec(
        "rgar_atari_random_init",
        "dreamerv3",
        "atari",
        "Random-Init",
        "random",
        "none",
        "heuristic",
        "rgar_ablation_atari",
        (LOG_ROOT / "random_init_no_rag_26_tasks_2944",),
        "Analytical alias of dv3_atari_main_random for RGAR-vs-random tables.",
    ),
    ExperimentSpec(
        "llm_qwen_1p5b",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "llm_backbone_ablation",
        (LOG_ROOT / "atari_qwen25_1p5b_latent_queue",),
    ),
    ExperimentSpec(
        "llm_qwen_3b",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "Qwen/Qwen2.5-3B-Instruct",
        "llm_backbone_ablation",
        (LOG_ROOT / "atari_qwen25_3b_latent_queue",),
    ),
    ExperimentSpec(
        "llm_qwen_7b",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "Qwen/Qwen2.5-7B-Instruct",
        "llm_backbone_ablation",
        (LOG_ROOT / "atari_qwen7b_latent_queue",),
    ),
    ExperimentSpec(
        "llm_llama_1b",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "meta-llama/Llama-3.2-1B-Instruct",
        "llm_backbone_ablation",
        (LOG_ROOT / "atari_llama32_1b_latent_queue",),
        "Snapshot may be partial at export time.",
    ),
    ExperimentSpec(
        "llm_llama_3b",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "meta-llama/Llama-3.2-3B-Instruct",
        "llm_backbone_ablation",
        (LOG_ROOT / "atari_llama32_3b_latent_queue",),
        "Snapshot may be partial at export time.",
    ),
    ExperimentSpec(
        "llm_llama_8b",
        "dreamerv3",
        "atari",
        "WM-Bench-RGAR",
        "task_conditioned",
        "latent",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llm_backbone_ablation",
        (LOG_ROOT / "atari_llama31_8b_latent_queue",),
    ),
]


def run_shell(cmd: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        shell=True,
        check=False,
        text=True,
        capture_output=True,
    )
    output = completed.stdout
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    return output.strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_roots_by_group() -> None:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for spec in EXPERIMENTS:
        grouped[spec.raw_group].extend(spec.roots)
    raw_root = EXPORT_DIR / "raw"
    for group, roots in grouped.items():
        group_dir = raw_root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        seen: set[Path] = set()
        for root in roots:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            dst = group_dir / root.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(root, dst)


def safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_trial_summaries(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("summary.json"):
        if path.name != "summary.json":
            continue
        if path.parent.name == "tables":
            continue
        if path.name == "search_summary.json":
            continue
        result.append(path)
    return result


def find_nearest_search_summary(path: Path, root: Path) -> dict[str, Any]:
    for parent in [path.parent] + list(path.parents):
        if parent == root.parent:
            break
        summary = parent / "search_summary.json"
        if summary.exists():
            data = safe_json_load(summary)
            if data:
                return data
    return {}


def find_nearest_search_summary_cached(
    path: Path, root: Path, cache: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    key = (str(root), str(path.parent))
    cached = cache.get(key)
    if cached is not None:
        return cached
    data = find_nearest_search_summary(path, root)
    cache[key] = data
    return data


def parse_trial_identity(summary_path: Path, root: Path) -> tuple[str, str, str, str]:
    rel = summary_path.relative_to(root)
    parts = rel.parts
    attack_idx = None
    attack_name = None
    for idx, part in enumerate(parts):
        if part in ATTACK_NAMES:
            attack_idx = idx
            attack_name = part
            break
    if attack_idx is None or attack_name is None:
        raise ValueError(f"could not parse attack path for {summary_path}")
    task = parts[attack_idx - 1]
    stage = parts[attack_idx + 1] if attack_idx + 1 < len(parts) else ""
    config_key = parts[attack_idx + 2] if attack_idx + 2 < len(parts) else attack_name
    return task, attack_name, stage, config_key


def infer_family_and_size(llm_model: str) -> tuple[str, str]:
    lower = llm_model.lower()
    if "qwen" in lower:
        family = "Qwen"
    elif "llama" in lower:
        family = "LLaMA"
    else:
        family = ""
    for size in ("0.5B", "1B", "1.5B", "3B", "7B", "8B", "9B", "14B", "32B", "70B", "72B"):
        if size.lower() in lower:
            return family, size
    return family, ""


def classify_failure(row: dict[str, Any]) -> str:
    clean = row["clean_mean_reward"]
    std_ratio = (row["std_reward"] or 0.0) / (abs(clean) + 1.0)
    if (row["elapsed_seconds"] or 0.0) >= 590:
        return "runtime_over_budget"
    if (row["flip_rate"] or 0.0) <= 0.01 and (row["normalized_reward_drop"] or 0.0) < 0.05:
        return "no_flip"
    if (row["flip_rate"] or 0.0) > 0.1 and (row["normalized_reward_drop"] or 0.0) < 0.05:
        return "flip_without_reward_drop"
    if (row["clean_margin_mean"] or 0.0) >= 1.0 and (row["normalized_reward_drop"] or 0.0) < 0.1:
        return "high_clean_margin"
    if ((row["clean_margin_mean"] or 0.0) - (row["adv_margin_mean"] or 0.0)) < 0.05 and (
        row["normalized_reward_drop"] or 0.0
    ) < 0.1:
        return "insufficient_margin_reduction"
    if std_ratio > 0.2:
        return "high_variance"
    if (row["normalized_reward_drop"] or 0.0) >= 0.3 or (row["scalarized_utility"] or 0.0) >= 0.2:
        return "effective"
    return "neutral"


def sanitize(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: sanitize(row.get(k, "")) for k in fieldnames})


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.stdev(values)


def build_manifest_rows(root_task_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in EXPERIMENTS:
        unique_tasks: set[str] = set()
        unique_attacks: set[str] = set()
        for root in spec.roots:
            stats = root_task_stats.get(str(root), {})
            unique_tasks.update(stats.get("tasks", set()))
            unique_attacks.update(stats.get("attacks", set()))
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "model": spec.model,
                "suite": spec.suite,
                "num_tasks": len(unique_tasks),
                "num_attacks": len(unique_attacks),
                "method": spec.method,
                "initialization": spec.initialization,
                "retrieval_mode": spec.retrieval_mode,
                "llm_model": spec.llm_model,
                "root_dir": ";".join(str(root) for root in spec.roots),
                "notes": spec.notes,
            }
        )
    return rows


def gather_trials() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    all_trials: list[dict[str, Any]] = []
    root_task_stats: dict[str, dict[str, Any]] = {}
    search_summary_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in EXPERIMENTS:
        for root in spec.roots:
            if not root.exists():
                root_task_stats[str(root)] = {"tasks": set(), "attacks": set()}
                continue
            summary_paths = find_trial_summaries(root)
            task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            task_clean: dict[str, tuple[float | None, float | None]] = {}
            task_stats = {"tasks": set(), "attacks": set()}
            for summary_path in summary_paths:
                try:
                    task, attack_name, stage, config_key = parse_trial_identity(summary_path, root)
                except ValueError:
                    continue
                payload = safe_json_load(summary_path)
                if not payload:
                    continue
                cfg = payload.get("config", {})
                telemetry = payload.get("telemetry", {})
                if attack_name == "baseline":
                    task_clean[task] = (payload.get("mean_reward"), payload.get("std_reward"))
                task_stats["tasks"].add(task)
                if attack_name != "baseline":
                    task_stats["attacks"].add(attack_name)
                search_summary = find_nearest_search_summary_cached(summary_path, root, search_summary_cache)
                row = {
                    "experiment_id": spec.experiment_id,
                    "model": spec.model,
                    "suite": spec.suite,
                    "task": task,
                    "attack_name": attack_name,
                    "method": spec.method,
                    "initialization_mode": search_summary.get("initialization_mode", spec.initialization),
                    "retrieval_mode": search_summary.get("experience_retrieval_mode", spec.retrieval_mode),
                    "llm_model": search_summary.get("agent_model", spec.llm_model),
                    "seed": cfg.get("seed"),
                    "stage": payload.get("stage", stage),
                    "round_index": "",
                    "trial_index": "",
                    "config_key": config_key,
                    "epsilon": cfg.get("epsilon"),
                    "steps": cfg.get("steps"),
                    "restarts": cfg.get("restarts"),
                    "rho": cfg.get("rho"),
                    "allocation_mode": (cfg.get("allocation") or {}).get("mode"),
                    "min_steps": (cfg.get("allocation") or {}).get("min_steps"),
                    "num_episodes": payload.get("num_episodes"),
                    "clean_mean_reward": "",
                    "clean_std_reward": "",
                    "mean_reward": payload.get("mean_reward"),
                    "std_reward": payload.get("std_reward"),
                    "median_reward": payload.get("median_reward"),
                    "min_reward": payload.get("min_reward"),
                    "max_reward": payload.get("max_reward"),
                    "normalized_reward_drop": "",
                    "flip_rate": telemetry.get("flip_rate"),
                    "clean_margin_mean": telemetry.get("clean_margin_mean"),
                    "adv_margin_mean": telemetry.get("adv_margin_mean"),
                    "allocated_steps_mean": telemetry.get("allocated_steps_mean"),
                    "allocated_epsilon_mean": telemetry.get("allocated_epsilon_mean"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "cumulative_elapsed_seconds": "",
                    "scalarized_utility": "",
                    "artifact_dir": payload.get("artifact_dir") or str(summary_path.parent),
                    "_summary_path": str(summary_path),
                    "_mtime": summary_path.stat().st_mtime,
                }
                task_rows[task].append(row)
            for task, rows in task_rows.items():
                clean_mean, clean_std = task_clean.get(task, (None, None))
                rows.sort(key=lambda r: (r["_mtime"], STAGE_ORDER.get(r["stage"], 9), r["config_key"]))
                cumulative_elapsed = 0.0
                for idx, row in enumerate(rows, start=1):
                    row["trial_index"] = idx
                    row["clean_mean_reward"] = clean_mean
                    row["clean_std_reward"] = clean_std
                    mean_reward = row["mean_reward"]
                    std_reward = row["std_reward"] or 0.0
                    elapsed = row["elapsed_seconds"] or 0.0
                    if clean_mean is not None and mean_reward is not None:
                        reward_drop = (clean_mean - mean_reward) / (abs(clean_mean) + 1.0)
                    else:
                        reward_drop = None
                    row["normalized_reward_drop"] = reward_drop
                    if reward_drop is not None:
                        row["scalarized_utility"] = (
                            reward_drop
                            + 0.25 * (row["flip_rate"] or 0.0)
                            - 0.15 * math.log1p(elapsed)
                            - 0.05 * (std_reward / ((abs(clean_mean) + 1.0) if clean_mean is not None else 1.0))
                        )
                    cumulative_elapsed += elapsed
                    row["cumulative_elapsed_seconds"] = cumulative_elapsed
                    all_trials.append(row)
            root_task_stats[str(root)] = task_stats
    return all_trials, root_task_stats


def build_main_best_by_task(all_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_trials:
        if row["attack_name"] == "baseline":
            continue
        grouped[(row["experiment_id"], row["model"], row["suite"], row["task"], row["method"])].append(row)
    best_rows = []
    for (experiment_id, model, suite, task, method), rows in grouped.items():
        ordered = sorted(rows, key=lambda r: r["trial_index"])
        confirm_rows = [r for r in ordered if r["stage"] == "confirm"]
        candidate_rows = confirm_rows or ordered
        if not candidate_rows:
            continue
        best = max(candidate_rows, key=lambda r: ((r["scalarized_utility"] or -1e9), -(r["mean_reward"] or 1e9)))
        best_rows.append(
            {
                "experiment_id": experiment_id,
                "model": model,
                "suite": suite,
                "task": task,
                "method": method,
                "best_attack_name": best["attack_name"],
                "best_config_key": best["config_key"],
                "best_epsilon": best["epsilon"],
                "best_steps": best["steps"],
                "best_restarts": best["restarts"],
                "best_rho": best["rho"],
                "best_allocation_mode": best["allocation_mode"],
                "best_mean_reward": best["mean_reward"],
                "clean_mean_reward": best["clean_mean_reward"],
                "best_normalized_reward_drop": best["normalized_reward_drop"],
                "best_flip_rate": best["flip_rate"],
                "best_scalarized_utility": best["scalarized_utility"],
                "best_elapsed_seconds": best["elapsed_seconds"],
                "total_trials": len(ordered),
                "total_confirm_trials": len(confirm_rows),
                "total_elapsed_seconds": sum(r["elapsed_seconds"] or 0.0 for r in ordered),
                "trials_to_best": best["trial_index"],
            }
        )
    return sorted(best_rows, key=lambda r: (r["experiment_id"], r["task"], r["method"]))


def build_main_aggregate(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    main_ids = {
        "dv3_atari_main_random",
        "dv3_atari_main_latent",
        "dv3_atari_claudini",
        "dv3_dmc_main_random",
        "dv3_dmc_main_latent",
        "dv3_dmc_claudini",
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        if row["experiment_id"] in main_ids and row["model"] == "dreamerv3":
            grouped[(row["model"], row["suite"], row["method"])].append(row)
    output = []
    for (model, suite, method), rows in grouped.items():
        output.append(
            {
                "model": model,
                "suite": suite,
                "method": method,
                "num_tasks": len(rows),
                "mean_best_reward_drop": mean([r["best_normalized_reward_drop"] for r in rows]),
                "std_best_reward_drop": stdev([r["best_normalized_reward_drop"] for r in rows]),
                "mean_best_flip_rate": mean([r["best_flip_rate"] for r in rows]),
                "std_best_flip_rate": stdev([r["best_flip_rate"] for r in rows]),
                "mean_best_utility": mean([r["best_scalarized_utility"] for r in rows]),
                "std_best_utility": stdev([r["best_scalarized_utility"] for r in rows]),
                "mean_total_elapsed_seconds": mean([r["total_elapsed_seconds"] for r in rows]),
                "mean_trials_to_best": mean([r["trials_to_best"] for r in rows]),
            }
        )
    return sorted(output, key=lambda r: (r["suite"], r["method"]))


def build_attack_family_summary(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        grouped[(row["model"], row["suite"], row["method"], row["best_attack_name"])].append(row)
    output = []
    for (model, suite, method, attack_name), rows in grouped.items():
        output.append(
            {
                "model": model,
                "suite": suite,
                "method": method,
                "attack_name": attack_name,
                "num_tasks_where_best": len(rows),
                "mean_reward_drop": mean([r["best_normalized_reward_drop"] for r in rows]),
                "mean_flip_rate": mean([r["best_flip_rate"] for r in rows]),
                "mean_utility": mean([r["best_scalarized_utility"] for r in rows]),
                "mean_runtime": mean([r["best_elapsed_seconds"] for r in rows]),
            }
        )
    return sorted(output, key=lambda r: (r["model"], r["suite"], r["method"], r["attack_name"]))


def build_cross_model(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key = defaultdict(list)
    for row in best_rows:
        if row["suite"] == "atari" and row["method"] == "WM-Bench-RGAR" and row["model"] in {"dreamerv2", "dreamerv3"}:
            rows_by_key[(row["suite"], row["task"], row["model"])].append(row)
    output = []
    for (suite, task, model), rows in rows_by_key.items():
        best = max(rows, key=lambda r: (r["best_scalarized_utility"] or -1e9))
        output.append(
            {
                "suite": suite,
                "task": task,
                "model": model,
                "method": best["method"],
                "best_attack_name": best["best_attack_name"],
                "clean_mean_reward": best["clean_mean_reward"],
                "best_mean_reward": best["best_mean_reward"],
                "best_normalized_reward_drop": best["best_normalized_reward_drop"],
                "best_flip_rate": best["best_flip_rate"],
                "best_scalarized_utility": best["best_scalarized_utility"],
                "total_trials": best["total_trials"],
                "total_elapsed_seconds": best["total_elapsed_seconds"],
            }
        )
    return sorted(output, key=lambda r: (r["task"], r["model"]))


def build_rgar_vs_random(all_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_ids = {"dv3_atari_main_latent", "dv3_atari_main_random"}
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_trials:
        if row["experiment_id"] in target_ids and row["attack_name"] != "baseline":
            grouped[(row["task"], row["model"], row["attack_name"], row["method"], row["experiment_id"])].append(row)
    output = []
    for (task, model, attack_name, method, experiment_id), rows in grouped.items():
        ordered = sorted(rows, key=lambda r: r["trial_index"])
        scout = [r for r in ordered if r["stage"] == "scout"]
        confirm = [r for r in ordered if r["stage"] == "confirm"]
        first_round = scout or ordered[:1]
        final_pool = confirm or ordered
        best_first = max(first_round, key=lambda r: (r["scalarized_utility"] or -1e9))
        best_final = max(final_pool, key=lambda r: (r["scalarized_utility"] or -1e9))
        threshold = 0.9 * (best_final["scalarized_utility"] or 0.0)
        trials_to_threshold = ""
        for row in ordered:
            if (row["scalarized_utility"] or -1e9) >= threshold:
                trials_to_threshold = row["trial_index"]
                break
        output.append(
            {
                "task": task,
                "model": model,
                "attack_name": attack_name,
                "method": method,
                "initialization_mode": "task_conditioned" if experiment_id == "dv3_atari_main_latent" else "random",
                "retrieval_mode": "latent" if experiment_id == "dv3_atari_main_latent" else "none",
                "first_round_best_utility": best_first["scalarized_utility"],
                "first_round_best_reward_drop": best_first["normalized_reward_drop"],
                "first_round_best_flip_rate": best_first["flip_rate"],
                "final_best_utility": best_final["scalarized_utility"],
                "final_best_reward_drop": best_final["normalized_reward_drop"],
                "final_best_flip_rate": best_final["flip_rate"],
                "trials_to_threshold": trials_to_threshold,
                "total_trials": len(ordered),
                "total_elapsed_seconds": sum(r["elapsed_seconds"] or 0.0 for r in ordered),
            }
        )
    return sorted(output, key=lambda r: (r["task"], r["attack_name"], r["method"]))


def build_llm_backbone_summary(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    llm_ids = {
        "llm_qwen_1p5b",
        "llm_qwen_3b",
        "llm_qwen_7b",
        "llm_llama_1b",
        "llm_llama_3b",
        "llm_llama_8b",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        if row["experiment_id"] in llm_ids:
            grouped[row["experiment_id"]].append(row)
    output = []
    for experiment_id, rows in grouped.items():
        llm_model = rows[0].get("llm_model", "")
        family, size = infer_family_and_size(llm_model)
        output.append(
            {
                "llm_family": family,
                "llm_size": size,
                "llm_model": llm_model,
                "suite": rows[0]["suite"],
                "num_tasks": len(rows),
                "mean_best_reward_drop": mean([r["best_normalized_reward_drop"] for r in rows]),
                "mean_best_flip_rate": mean([r["best_flip_rate"] for r in rows]),
                "mean_best_utility": mean([r["best_scalarized_utility"] for r in rows]),
                "mean_trials_to_threshold": mean([0.9 * (r["trials_to_best"] or 0) for r in rows]),
                "mean_total_elapsed_seconds": mean([r["total_elapsed_seconds"] for r in rows]),
                "invalid_proposal_rate": "",
                "notes": experiment_id,
            }
        )
    return sorted(output, key=lambda r: (r["llm_family"], r["llm_size"]))


def build_efficiency_curves(all_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_trials:
        if row["attack_name"] == "baseline":
            continue
        grouped[(row["experiment_id"], row["model"], row["suite"], row["task"], row["method"], row["attack_name"])].append(row)
    output = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["trial_index"])
        best_utility = -1e18
        best_reward_drop = -1e18
        best_flip_rate = -1e18
        cumulative_elapsed = 0.0
        confirm_count = 0
        for idx, row in enumerate(rows, start=1):
            cumulative_elapsed += row["elapsed_seconds"] or 0.0
            if row["stage"] == "confirm":
                confirm_count += 1
            best_utility = max(best_utility, row["scalarized_utility"] if row["scalarized_utility"] is not None else -1e18)
            best_reward_drop = max(
                best_reward_drop, row["normalized_reward_drop"] if row["normalized_reward_drop"] is not None else -1e18
            )
            best_flip_rate = max(best_flip_rate, row["flip_rate"] if row["flip_rate"] is not None else -1e18)
            output.append(
                {
                    "experiment_id": key[0],
                    "model": key[1],
                    "suite": key[2],
                    "task": key[3],
                    "method": key[4],
                    "attack_name": key[5],
                    "trial_index": row["trial_index"],
                    "cumulative_trials": idx,
                    "cumulative_confirm_trials": confirm_count,
                    "cumulative_elapsed_seconds": cumulative_elapsed,
                    "best_so_far_utility": best_utility if best_utility > -1e17 else "",
                    "best_so_far_reward_drop": best_reward_drop if best_reward_drop > -1e17 else "",
                    "best_so_far_flip_rate": best_flip_rate if best_flip_rate > -1e17 else "",
                }
            )
    return sorted(output, key=lambda r: (r["experiment_id"], r["task"], r["attack_name"], r["trial_index"]))


def build_trials_to_threshold(eff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eff_rows:
        grouped[(row["experiment_id"], row["model"], row["suite"], row["task"], row["method"], row["attack_name"])].append(row)
    output = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r["trial_index"])
        final_best = max([r["best_so_far_utility"] for r in rows if r["best_so_far_utility"] != ""], default=None)
        if final_best is None:
            continue
        for pct in (0.8, 0.9, 0.95):
            threshold = final_best * pct
            hit = None
            for row in rows:
                value = row["best_so_far_utility"]
                if value != "" and value >= threshold:
                    hit = row
                    break
            output.append(
                {
                    "experiment_id": key[0],
                    "model": key[1],
                    "suite": key[2],
                    "task": key[3],
                    "method": key[4],
                    "attack_name": key[5],
                    "threshold_type": f"{int(pct * 100)}pct_final_utility",
                    "threshold_value": threshold,
                    "trials_to_threshold": hit["cumulative_trials"] if hit else "",
                    "elapsed_seconds_to_threshold": hit["cumulative_elapsed_seconds"] if hit else "",
                    "reached": 1 if hit else 0,
                }
            )
    return sorted(output, key=lambda r: (r["experiment_id"], r["task"], r["attack_name"], r["threshold_type"]))


def build_failure_mode_summary(all_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_trials:
        if row["attack_name"] == "baseline":
            continue
        tag = classify_failure(row)
        grouped[(row["model"], row["suite"], row["method"], row["attack_name"], tag)].append(row)
    output = []
    for (model, suite, method, attack_name, tag), rows in grouped.items():
        output.append(
            {
                "model": model,
                "suite": suite,
                "method": method,
                "attack_name": attack_name,
                "failure_tag": tag,
                "count": len(rows),
                "mean_reward_drop": mean([r["normalized_reward_drop"] for r in rows]),
                "mean_flip_rate": mean([r["flip_rate"] for r in rows]),
                "mean_utility": mean([r["scalarized_utility"] for r in rows]),
            }
        )
    return sorted(output, key=lambda r: (r["model"], r["suite"], r["method"], r["attack_name"], r["failure_tag"]))


def build_figure_tables(eff_rows: list[dict[str, Any]], rgar_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_trials: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eff_rows:
        by_trials[(row["suite"], row["model"], row["method"], row["cumulative_trials"])].append(row)
    best_utility_vs_trials = []
    for (suite, model, method, cumulative_trials), rows in sorted(by_trials.items()):
        best_utility_vs_trials.append(
            {
                "suite": suite,
                "model": model,
                "method": method,
                "cumulative_trials": cumulative_trials,
                "mean_best_so_far_utility": mean([r["best_so_far_utility"] for r in rows]),
                "std_best_so_far_utility": stdev([r["best_so_far_utility"] for r in rows]),
            }
        )

    by_time: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in eff_rows:
        time_bin = int((row["cumulative_elapsed_seconds"] or 0.0) // 60) * 60
        by_time[(row["suite"], row["model"], row["method"], time_bin)].append(row)
    best_reward_drop_vs_time = []
    for (suite, model, method, time_bin), rows in sorted(by_time.items()):
        best_reward_drop_vs_time.append(
            {
                "suite": suite,
                "model": model,
                "method": method,
                "cumulative_elapsed_seconds": time_bin,
                "mean_best_so_far_reward_drop": mean([r["best_so_far_reward_drop"] for r in rows]),
                "std_best_so_far_reward_drop": stdev([r["best_so_far_reward_drop"] for r in rows]),
            }
        )

    warm_start_grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rgar_rows:
        warm_start_grouped[(row["model"], "atari", row["method"], row["attack_name"])].append(row)
    rgar_warm_start_quality = []
    for (model, suite, method, attack_name), rows in sorted(warm_start_grouped.items()):
        rgar_warm_start_quality.append(
            {
                "suite": suite,
                "model": model,
                "method": method,
                "attack_name": attack_name,
                "mean_first_round_utility": mean([r["first_round_best_utility"] for r in rows]),
                "std_first_round_utility": stdev([r["first_round_best_utility"] for r in rows]),
                "mean_final_utility": mean([r["final_best_utility"] for r in rows]),
                "std_final_utility": stdev([r["final_best_utility"] for r in rows]),
            }
        )
    return best_utility_vs_trials, best_reward_drop_vs_time, rgar_warm_start_quality


def enrich_best_rows_with_llm(best_rows: list[dict[str, Any]], all_trials: list[dict[str, Any]]) -> None:
    llm_by_key: dict[tuple[str, str, str, str, str], str] = {}
    for row in all_trials:
        key = (row["experiment_id"], row["model"], row["suite"], row["task"], row["method"])
        if row["llm_model"]:
            llm_by_key[key] = row["llm_model"]
    for row in best_rows:
        row["llm_model"] = llm_by_key.get((row["experiment_id"], row["model"], row["suite"], row["task"], row["method"]), "")


def create_metadata(root_task_stats: dict[str, dict[str, Any]]) -> None:
    metadata_dir = EXPORT_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    repo_state = "\n".join(
        [
            "[WMAutoattack]",
            run_shell("git rev-parse HEAD", ROOT),
            run_shell("git status --short", ROOT),
            run_shell("git branch --show-current", ROOT),
            "",
            "[sheeprl]",
            run_shell("git rev-parse HEAD", Path("/share/guozhix/sheeprl")),
            run_shell("git status --short", Path("/share/guozhix/sheeprl")),
            run_shell("git branch --show-current", Path("/share/guozhix/sheeprl")),
        ]
    )
    write_text(metadata_dir / "repo_state.txt", repo_state + "\n")

    environment = "\n".join(
        [
            "[python --version]",
            run_shell("python --version", ROOT),
            "",
            "[pip freeze]",
            run_shell("pip freeze", ROOT),
            "",
            "[nvidia-smi]",
            run_shell("nvidia-smi", ROOT),
            "",
            "[squeue -u guozhix]",
            run_shell("squeue -u guozhix", ROOT),
        ]
    )
    write_text(metadata_dir / "environment.txt", environment + "\n")

    manifest_rows = build_manifest_rows(root_task_stats)
    write_csv(
        metadata_dir / "experiment_manifest.csv",
        [
            "experiment_id",
            "model",
            "suite",
            "num_tasks",
            "num_attacks",
            "method",
            "initialization",
            "retrieval_mode",
            "llm_model",
            "root_dir",
            "notes",
        ],
        manifest_rows,
    )


def create_readme(root_task_stats: dict[str, dict[str, Any]]) -> None:
    included = []
    for raw_group in ("dreamerv3_atari", "dreamerv3_dmc", "dreamerv2_atari", "rgar_ablation_atari", "llm_backbone_ablation"):
        count = len(list((EXPORT_DIR / "raw" / raw_group).glob("*"))) if (EXPORT_DIR / "raw" / raw_group).exists() else 0
        included.append(f"- {raw_group}: {count} directories")
    missing = [
        "- DreamerV3 Atari aligned main/Claudini runs use 4 attacks (no square) rather than 5.",
        "- DreamerV2 Atari export includes one complete Alien validation run and a rerun snapshot; full rerun may still be incomplete.",
        "- Some LLaMA backbone queues may be partial snapshots depending on export time.",
    ]
    notes = [
        "- Trial-level tables were reconstructed from per-trial summary.json files instead of relying only on top-level search_summary.json.",
        "- Resumed runs that wrote back into original directories are treated as one merged result root.",
        "- Utility formula matches the export checklist.",
    ]
    readme = "\n".join(
        [
            "# WM-Bench Experiment Export",
            "",
            f"Export date: {datetime.utcnow().strftime('%Y-%m-%d')}",
            "Server: localhost / 6000ada",
            f"Repository commit: {run_shell('git rev-parse HEAD', ROOT)}",
            "",
            "Included experiment groups:",
            *included,
            "",
            "Known missing results:",
            *missing,
            "",
            "Notes:",
            *notes,
            "",
            "Task-count snapshot:",
        ]
    )
    snapshot_lines = []
    for spec in EXPERIMENTS:
        tasks = set()
        for root in spec.roots:
            tasks.update(root_task_stats.get(str(root), {}).get("tasks", set()))
        snapshot_lines.append(f"- {spec.experiment_id}: {len(tasks)} tasks")
    write_text(EXPORT_DIR / "README.md", readme + "\n" + "\n".join(snapshot_lines) + "\n")


def build_export() -> None:
    reuse_raw = os.environ.get("WMBENCH_REUSE_RAW", "").lower() in {"1", "true", "yes"}
    if EXPORT_DIR.exists() and not reuse_raw:
        shutil.rmtree(EXPORT_DIR)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    (EXPORT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "figures_ready").mkdir(parents=True, exist_ok=True)
    if not reuse_raw:
        copy_roots_by_group()
    all_trials, root_task_stats = gather_trials()
    all_trials.sort(key=lambda r: (r["experiment_id"], r["task"], r["trial_index"]))

    best_rows = build_main_best_by_task(all_trials)
    enrich_best_rows_with_llm(best_rows, all_trials)
    main_aggregate = build_main_aggregate(best_rows)
    attack_summary = build_attack_family_summary(best_rows)
    cross_model = build_cross_model(best_rows)
    rgar_rows = build_rgar_vs_random(all_trials)
    llm_summary = build_llm_backbone_summary(best_rows)
    eff_rows = build_efficiency_curves(all_trials)
    threshold_rows = build_trials_to_threshold(eff_rows)
    failure_rows = build_failure_mode_summary(all_trials)
    fig_trials, fig_time, fig_warm = build_figure_tables(eff_rows, rgar_rows)

    create_metadata(root_task_stats)
    create_readme(root_task_stats)

    tables_dir = EXPORT_DIR / "tables"
    write_csv(
        tables_dir / "all_trials.csv",
        [
            "experiment_id",
            "model",
            "suite",
            "task",
            "attack_name",
            "method",
            "initialization_mode",
            "retrieval_mode",
            "llm_model",
            "seed",
            "stage",
            "round_index",
            "trial_index",
            "config_key",
            "epsilon",
            "steps",
            "restarts",
            "rho",
            "allocation_mode",
            "min_steps",
            "num_episodes",
            "clean_mean_reward",
            "clean_std_reward",
            "mean_reward",
            "std_reward",
            "median_reward",
            "min_reward",
            "max_reward",
            "normalized_reward_drop",
            "flip_rate",
            "clean_margin_mean",
            "adv_margin_mean",
            "allocated_steps_mean",
            "allocated_epsilon_mean",
            "elapsed_seconds",
            "cumulative_elapsed_seconds",
            "scalarized_utility",
            "artifact_dir",
        ],
        all_trials,
    )
    write_csv(
        tables_dir / "main_best_by_task.csv",
        [
            "experiment_id",
            "model",
            "suite",
            "task",
            "method",
            "best_attack_name",
            "best_config_key",
            "best_epsilon",
            "best_steps",
            "best_restarts",
            "best_rho",
            "best_allocation_mode",
            "best_mean_reward",
            "clean_mean_reward",
            "best_normalized_reward_drop",
            "best_flip_rate",
            "best_scalarized_utility",
            "best_elapsed_seconds",
            "total_trials",
            "total_confirm_trials",
            "total_elapsed_seconds",
            "trials_to_best",
        ],
        best_rows,
    )
    write_csv(
        tables_dir / "main_aggregate_by_suite.csv",
        [
            "model",
            "suite",
            "method",
            "num_tasks",
            "mean_best_reward_drop",
            "std_best_reward_drop",
            "mean_best_flip_rate",
            "std_best_flip_rate",
            "mean_best_utility",
            "std_best_utility",
            "mean_total_elapsed_seconds",
            "mean_trials_to_best",
        ],
        main_aggregate,
    )
    write_csv(
        tables_dir / "attack_family_summary.csv",
        ["model", "suite", "method", "attack_name", "num_tasks_where_best", "mean_reward_drop", "mean_flip_rate", "mean_utility", "mean_runtime"],
        attack_summary,
    )
    write_csv(
        tables_dir / "cross_model_dreamerv2.csv",
        [
            "suite",
            "task",
            "model",
            "method",
            "best_attack_name",
            "clean_mean_reward",
            "best_mean_reward",
            "best_normalized_reward_drop",
            "best_flip_rate",
            "best_scalarized_utility",
            "total_trials",
            "total_elapsed_seconds",
        ],
        cross_model,
    )
    write_csv(
        tables_dir / "rgar_vs_random_init.csv",
        [
            "task",
            "model",
            "attack_name",
            "method",
            "initialization_mode",
            "retrieval_mode",
            "first_round_best_utility",
            "first_round_best_reward_drop",
            "first_round_best_flip_rate",
            "final_best_utility",
            "final_best_reward_drop",
            "final_best_flip_rate",
            "trials_to_threshold",
            "total_trials",
            "total_elapsed_seconds",
        ],
        rgar_rows,
    )
    write_csv(
        tables_dir / "llm_backbone_summary.csv",
        [
            "llm_family",
            "llm_size",
            "llm_model",
            "suite",
            "num_tasks",
            "mean_best_reward_drop",
            "mean_best_flip_rate",
            "mean_best_utility",
            "mean_trials_to_threshold",
            "mean_total_elapsed_seconds",
            "invalid_proposal_rate",
            "notes",
        ],
        llm_summary,
    )
    write_csv(
        tables_dir / "efficiency_curves.csv",
        [
            "experiment_id",
            "model",
            "suite",
            "task",
            "method",
            "attack_name",
            "trial_index",
            "cumulative_trials",
            "cumulative_confirm_trials",
            "cumulative_elapsed_seconds",
            "best_so_far_utility",
            "best_so_far_reward_drop",
            "best_so_far_flip_rate",
        ],
        eff_rows,
    )
    write_csv(
        tables_dir / "trials_to_threshold.csv",
        [
            "experiment_id",
            "model",
            "suite",
            "task",
            "method",
            "attack_name",
            "threshold_type",
            "threshold_value",
            "trials_to_threshold",
            "elapsed_seconds_to_threshold",
            "reached",
        ],
        threshold_rows,
    )
    write_csv(
        tables_dir / "failure_mode_summary.csv",
        ["model", "suite", "method", "attack_name", "failure_tag", "count", "mean_reward_drop", "mean_flip_rate", "mean_utility"],
        failure_rows,
    )

    fig_dir = EXPORT_DIR / "figures_ready"
    write_csv(
        fig_dir / "best_utility_vs_trials.csv",
        ["suite", "model", "method", "cumulative_trials", "mean_best_so_far_utility", "std_best_so_far_utility"],
        fig_trials,
    )
    write_csv(
        fig_dir / "best_reward_drop_vs_time.csv",
        ["suite", "model", "method", "cumulative_elapsed_seconds", "mean_best_so_far_reward_drop", "std_best_so_far_reward_drop"],
        fig_time,
    )
    write_csv(
        fig_dir / "rgar_warm_start_quality.csv",
        ["suite", "model", "method", "attack_name", "mean_first_round_utility", "std_first_round_utility", "mean_final_utility", "std_final_utility"],
        fig_warm,
    )

    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=str(EXPORT_DIR.parent), base_dir=EXPORT_DIR.name)


def main() -> int:
    build_export()
    print(str(EXPORT_DIR))
    print(str(ZIP_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
