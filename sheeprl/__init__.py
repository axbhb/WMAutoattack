from __future__ import annotations

from pathlib import Path


_INNER_ROOT = Path(__file__).resolve().parent / "sheeprl"
__path__ = [str(_INNER_ROOT)]

_inner_init = _INNER_ROOT / "__init__.py"
exec(compile(_inner_init.read_text(encoding="utf-8"), str(_inner_init), "exec"))

