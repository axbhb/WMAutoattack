import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "sheeprl"

for path in (ROOT, UPSTREAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

