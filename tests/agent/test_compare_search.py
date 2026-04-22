import json
from pathlib import Path

from agent.compare_search import _seed_memory_store
from agent.memory import ExperienceMemoryStore
from agent.schema import TaskProfile


def _make_entry(store: ExperienceMemoryStore, checkpoint_path: str, task_name: str):
    profile = TaskProfile(
        task_name=task_name,
        checkpoint_path=checkpoint_path,
        env_id=task_name,
        task_tokens=(task_name.lower(),),
    )
    return store.build_entry(
        task_profile=profile,
        attack_name="apgd_ce",
        best_config={"epsilon": 8.0, "steps": 10, "allocation": {"mode": "fixed"}},
        result_summary={"mean_reward": -10.0},
        utility=1.0,
        source_run_dir="/tmp/run",
    )


def test_seed_memory_store_excludes_target_checkpoint(tmp_path):
    source_path = tmp_path / "source.jsonl"
    source_store = ExperienceMemoryStore(str(source_path))
    source_store.extend(
        [
            _make_entry(source_store, "/tmp/target.ckpt", "pong"),
            _make_entry(source_store, "/tmp/other.ckpt", "alien"),
        ]
    )

    destination_path = tmp_path / "seeded.jsonl"
    summary = _seed_memory_store(
        str(source_path),
        str(destination_path),
        excluded_checkpoints=["/tmp/target.ckpt"],
    )

    assert summary["seeded_entries"] == 1
    assert summary["skipped_entries"] == 1
    lines = [json.loads(line) for line in Path(destination_path).read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["task_profile"]["checkpoint_path"] == "/tmp/other.ckpt"
