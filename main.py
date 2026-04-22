from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parent
UPSTREAM_ROOT = ROOT / "sheeprl"

for path in (ROOT, UPSTREAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _usage() -> str:
    return (
        "Usage: python main.py <command> [args...]\n\n"
        "Commands:\n"
        "  train          Run upstream sheeprl training\n"
        "  eval           Run attack-aware evaluation\n"
        "  baseline       Run evaluation with attack disabled\n"
        "  attack-search  Run Reflexion/RAG attack parameter search\n"
        "  compare-search Run fair Claudini-style vs ours comparison search\n"
    )


def _run_with_argv(entrypoint: Callable[[], None], args: Sequence[str], program_name: str) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [program_name] + list(args)
        entrypoint()
    finally:
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 0 or argv[0] in {"-h", "--help"}:
        print(_usage())
        return 0

    command, forwarded = argv[0], argv[1:]
    if command == "train":
        from sheeprl.cli import run

        if any(arg in {"-h", "--help"} for arg in forwarded):
            forwarded = ["--hydra-help"]
        _run_with_argv(run, forwarded, "sheeprl")
        return 0
    if command == "eval":
        from autoattack.evaluation import run_eval

        run_eval(forwarded, force_baseline=False)
        return 0
    if command == "baseline":
        from autoattack.evaluation import run_baseline

        run_baseline(forwarded)
        return 0
    if command == "attack-search":
        from agent.run_search import main as run_search_main

        _run_with_argv(run_search_main, forwarded, "wmautoattack-attack-search")
        return 0
    if command == "compare-search":
        from agent.compare_search import main as compare_search_main

        _run_with_argv(compare_search_main, forwarded, "wmautoattack-compare-search")
        return 0

    raise SystemExit("Unknown command '{}'.\n\n{}".format(command, _usage()))


if __name__ == "__main__":
    raise SystemExit(main())
