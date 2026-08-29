"""Put the repository root on sys.path so `pytest tests/...` works from any
invocation form, not only `python -m pytest` (which inserts the cwd itself)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
