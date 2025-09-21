"""Shared IO helpers for retail analytics pipelines."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

LOGGER_NAME = "retail_analytics"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def _configure_logger() -> logging.Logger:
    level_name = os.getenv("RETAIL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = _configure_logger()


@lru_cache(maxsize=1)
def get_paths(base: str | Path | None = None) -> SimpleNamespace:
    """Return canonical project paths relative to repo root."""
    base_path = Path(base) if base else Path(__file__).resolve().parents[2]
    data_dir = base_path / "data"
    paths = SimpleNamespace(
        base=base_path,
        data=data_dir,
        bronze=data_dir / "bronze",
        silver=data_dir / "silver",
        gold=data_dir / "gold",
        configs=base_path / "configs",
        reports=base_path / "reports",
    )
    return paths


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists. If ``path`` is a file, ensure its parent exists."""
    path_obj = Path(path)
    target = path_obj if path_obj.suffix == "" else path_obj.parent
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV with logging."""
    path_obj = Path(path)
    logger.info("Reading CSV: %s", path_obj)
    df = pd.read_csv(path_obj, **kwargs)
    logger.debug("Loaded %s shape=%s", path_obj.name, df.shape)
    return df


def write_parquet(df: pd.DataFrame, path: str | Path, *, coerce_dtypes: bool = True, **kwargs: Any) -> Path:
    """Write a DataFrame to parquet (pyarrow) with logging."""
    path_obj = Path(path)
    ensure_dir(path_obj)
    logger.info("Writing Parquet: %s rows=%s cols=%s", path_obj, len(df), len(df.columns))
    if coerce_dtypes:
        df = df.copy()
        df = df.convert_dtypes()
    df.to_parquet(path_obj, index=False, engine="pyarrow", **kwargs)
    return path_obj


__all__ = ["get_paths", "ensure_dir", "read_csv", "write_parquet", "logger"]
