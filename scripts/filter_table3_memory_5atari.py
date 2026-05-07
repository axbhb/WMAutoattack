#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("/share/guozhix/WMAutoattack/logs/attack_search/probe_scale_26_tasks_experience_2945.jsonl")
OUTPUT = Path("/share/guozhix/WMAutoattack/logs/table3_missing_rows_5atari/memory_train_only.jsonl")
EXCLUDE_TASKS = {
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
}


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with SOURCE.open("r", encoding="utf-8") as src, OUTPUT.open("w", encoding="utf-8") as dst:
        for line in src:
            entry = json.loads(line)
            task_name = entry.get("task_profile", {}).get("task_name", "")
            if task_name in EXCLUDE_TASKS:
                skipped += 1
                continue
            dst.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"source": str(SOURCE), "output": str(OUTPUT), "kept": kept, "skipped": skipped}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
