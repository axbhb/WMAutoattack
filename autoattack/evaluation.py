from __future__ import annotations

import sys
from typing import Iterable, List, Sequence

from sheeprl.cli import evaluation as sheeprl_evaluation


def _normalized_args(args: Sequence[str] | None) -> List[str]:
    return [str(arg) for arg in (args or [])]


def run_eval(args: Sequence[str] | None = None, *, force_baseline: bool = False) -> None:
    forwarded = _normalized_args(args)
    if force_baseline and not any(arg.startswith("attack.enabled=") for arg in forwarded):
        forwarded.append("attack.enabled=False")
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + forwarded
        sheeprl_evaluation()
    finally:
        sys.argv = old_argv


def run_baseline(args: Sequence[str] | None = None) -> None:
    run_eval(args=args, force_baseline=True)

