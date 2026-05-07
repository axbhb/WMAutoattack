#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path("/share/guozhix/WMAutoattack")
BASE = ROOT / "logs" / "table3_missing_rows_5atari"
EXPORT_DIR = BASE / "export"
MEMORY_PATH = BASE / "memory_train_only.jsonl"
VARIANTS = {
    "RGAR only, w/o SCAS": BASE / "rgar_only_wo_scas",
    "w/o RGAR and SCAS": BASE / "wo_rgar_scas",
}
TARGET_TASKS = [
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
]
ATTACKS = ["apgd_ce", "apgd_dlr", "fab", "two_stage"]


def normalized_reward_drop(clean_mean: float, attacked_mean: float) -> float:
    return (clean_mean - attacked_mean) / (abs(clean_mean) + 1.0)


def scalarized_utility(clean_mean: float, mean_reward: float, std_reward: float, flip_rate: float, elapsed_seconds: float) -> float:
    drop = normalized_reward_drop(clean_mean, mean_reward)
    return drop + 0.25 * flip_rate - 0.15 * math.log1p(max(elapsed_seconds, 0.0)) - 0.05 * (
        std_reward / (abs(clean_mean) + 1.0)
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_shell(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, text=True, capture_output=True, cwd=str(ROOT))
    return (out.stdout or out.stderr).strip()


def read_memory_count() -> int:
    if not MEMORY_PATH.exists():
        return 0
    with MEMORY_PATH.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def locate_trial_summaries(variant_root: Path, task: str, attack: str):
    attack_root = variant_root / task / attack
    scout = sorted((attack_root / "scout").rglob("summary.json")) if (attack_root / "scout").exists() else []
    confirm = sorted((attack_root / "confirm").rglob("summary.json")) if (attack_root / "confirm").exists() else []
    return scout, confirm


def best_trial(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    return max(rows, key=lambda r: r["utility"])


def trial_row_from_summary(path: Path, clean_mean: float) -> dict:
    payload = load_json(path)
    cfg = payload.get("config", {})
    telemetry = payload.get("telemetry", {})
    return {
        "config_key": path.parent.name,
        "mean_reward": payload["mean_reward"],
        "std_reward": payload["std_reward"],
        "flip_rate": float(telemetry.get("flip_rate", 0.0)),
        "elapsed_seconds": payload["elapsed_seconds"],
        "utility": scalarized_utility(
            clean_mean,
            payload["mean_reward"],
            payload["std_reward"],
            float(telemetry.get("flip_rate", 0.0)),
            payload["elapsed_seconds"],
        ),
        "drop": normalized_reward_drop(clean_mean, payload["mean_reward"]),
        "path": str(path),
    }


def export() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_pair_rows = []
    summary_rows = []
    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "exact_task_list": TARGET_TASKS,
        "attack_families": ATTACKS,
        "git_commit": run_shell("git rev-parse HEAD"),
        "seed": 0,
        "search_mode_static_required_code_change": True,
        "filtered_experience_store_path": str(MEMORY_PATH),
        "filtered_experience_store_entries": read_memory_count(),
        "variants": {},
    }

    for variant_name, variant_root in VARIANTS.items():
        variant_pairs = []
        task_checkpoints = {}
        command_lines = []
        for task in TARGET_TASKS:
            baseline = load_json(variant_root / task / "baseline" / "confirm" / "baseline" / "summary.json")
            clean_mean = baseline["mean_reward"]
            for attack in ATTACKS:
                scout_paths, confirm_paths = locate_trial_summaries(variant_root, task, attack)
                scout_rows = [trial_row_from_summary(p, clean_mean) for p in scout_paths]
                confirm_rows = [trial_row_from_summary(p, clean_mean) for p in confirm_paths]
                first_batch = scout_rows[:4]
                first = best_trial(first_batch)
                final_pool = []
                confirm_by_key = {row["config_key"]: row for row in confirm_rows}
                for scout in scout_rows:
                    final_pool.append(confirm_by_key.get(scout["config_key"], scout))
                final = best_trial(final_pool)
                total_elapsed = sum(r["elapsed_seconds"] for r in scout_rows) + sum(r["elapsed_seconds"] for r in confirm_rows)
                total_trials = len(scout_rows) + len(confirm_rows)
                row = {
                    "variant": variant_name,
                    "task": task,
                    "attack_name": attack,
                    "first_utility": first["utility"] if first else "",
                    "first_drop": first["drop"] if first else "",
                    "first_flip": first["flip_rate"] if first else "",
                    "final_utility": final["utility"] if final else "",
                    "final_drop": final["drop"] if final else "",
                    "final_flip": final["flip_rate"] if final else "",
                    "total_trials": total_trials,
                    "total_elapsed_seconds": total_elapsed,
                    "output_dir": str(variant_root / task / attack),
                }
                per_pair_rows.append(row)
                variant_pairs.append(row)
                if not task_checkpoints.get(task):
                    task_checkpoints[task] = baseline.get("config", {}).get("checkpoint_path", "")
            search_summary = variant_root / task / "search_summary.json"
            if search_summary.exists():
                command_lines.append(str(search_summary))
        pairs = len(variant_pairs)
        summary_rows.append(
            {
                "variant": variant_name,
                "tasks": len(TARGET_TASKS),
                "pairs": pairs,
                "first_utility": round(sum(r["first_utility"] for r in variant_pairs) / pairs, 3) if pairs else "",
                "first_drop": round(sum(r["first_drop"] for r in variant_pairs) / pairs, 3) if pairs else "",
                "first_flip": round(sum(r["first_flip"] for r in variant_pairs) / pairs, 3) if pairs else "",
                "final_utility": round(sum(r["final_utility"] for r in variant_pairs) / pairs, 3) if pairs else "",
                "final_drop": round(sum(r["final_drop"] for r in variant_pairs) / pairs, 3) if pairs else "",
                "final_flip": round(sum(r["final_flip"] for r in variant_pairs) / pairs, 3) if pairs else "",
                "total_elapsed_seconds": round(sum(r["total_elapsed_seconds"] for r in variant_pairs), 3),
            }
        )
        metadata["variants"][variant_name] = {
            "root_dir": str(variant_root),
            "checkpoint_paths": task_checkpoints,
            "search_mode": "static",
            "command_line": (variant_root / "command.txt").read_text(encoding="utf-8").strip()
            if (variant_root / "command.txt").exists()
            else "",
        }

    with (EXPORT_DIR / "table3_missing_rows_5atari_per_pair.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "task",
                "attack_name",
                "first_utility",
                "first_drop",
                "first_flip",
                "final_utility",
                "final_drop",
                "final_flip",
                "total_trials",
                "total_elapsed_seconds",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(per_pair_rows)

    with (EXPORT_DIR / "table3_missing_rows_5atari_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "tasks",
                "pairs",
                "first_utility",
                "first_drop",
                "first_flip",
                "final_utility",
                "final_drop",
                "final_flip",
                "total_elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (EXPORT_DIR / "table3_missing_rows_5atari_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    export()
