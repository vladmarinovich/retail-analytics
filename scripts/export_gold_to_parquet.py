"""Deprecated helper preserved for backwards compatibility.

Use ``scripts/rebuild_gold_parquet.py`` instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild_gold_parquet import main  # type: ignore


if __name__ == "__main__":
    main()
