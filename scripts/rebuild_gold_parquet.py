"""Rebuild all GOLD parquet outputs."""
from __future__ import annotations

import sys
import yaml

sys.path.insert(0, "src")

from run_pipeline import run_pipeline  # noqa: E402
from utils.io import get_paths, logger  # noqa: E402


def _gold_targets() -> list[str]:
    config_path = get_paths().configs / "artifacts.yml"
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    artifacts = cfg.get("artifacts", {})
    return [name for name, spec in artifacts.items() if spec.get("layer") == "gold"]


def main() -> None:
    targets = _gold_targets()
    if not targets:
        raise RuntimeError("No gold artifacts found in configuration")
    logger.info("Rebuilding gold parquet outputs: %s", ", ".join(targets))
    run_pipeline(targets)


if __name__ == "__main__":
    main()
