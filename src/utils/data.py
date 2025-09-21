"""Data loading helpers."""
from __future__ import annotations

import pandas as pd

from utils.io import get_paths, read_csv
from utils.schemas import transactions_base_schema


def load_transactions() -> pd.DataFrame:
    """Load and validate the canonical transactions base table."""
    paths = get_paths()
    path = paths.silver / "transactions_base.csv"
    df = read_csv(path, parse_dates=["InvoiceDate"])
    df = transactions_base_schema.validate(df, lazy=True)
    df["Country"] = df["Country"].fillna("Unspecified").str.strip()
    df.loc[df["Country"] == "", "Country"] = "Unspecified"
    df["YearMonth"] = df["YearMonth"].astype(str)
    return df


__all__ = ["load_transactions"]
