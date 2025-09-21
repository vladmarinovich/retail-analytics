"""Quick QC checks for GOLD parquet outputs."""
from __future__ import annotations

import sys
from typing import Dict, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from utils.io import get_paths, logger  # noqa: E402

PATHS = get_paths()


def _read_parquet_map() -> Dict[str, pd.DataFrame]:
    gold_dir = PATHS.gold
    data = {}
    for parquet in gold_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet)
        data[parquet.stem] = df
    return data


def _period_summary(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if "period" not in df.columns:
        return None, None
    period = pd.to_datetime(df["period"], errors="coerce")
    if period.isna().all():
        return None, None
    return period.min(), period.max()


def _check_nan_inf(name: str, df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        n_nan = series.isna().sum()
        n_inf = np.isinf(series).sum()
        if n_nan or n_inf:
            logger.warning("[%s] Column '%s' has NaN=%s, Inf=%s", name, col, n_nan, n_inf)


def _check_country_vs_company(data: Dict[str, pd.DataFrame]) -> None:
    if "company_monthly_kpis" not in data or "country_monthly_kpis" not in data:
        return
    company = data["company_monthly_kpis"].copy()
    country = data["country_monthly_kpis"].copy()
    if "net_sales" not in company.columns or "net_sales" not in country.columns:
        return
    comp = company.set_index("YearMonth")["net_sales"].astype(float)
    ctry = (
        country.groupby("YearMonth")["net_sales"].sum().astype(float)
    )
    mismatch = (comp.round(2) - ctry.round(2)).abs()
    max_diff = mismatch.max()
    if max_diff > 1.0:
        logger.error(
            "Company vs Country net_sales mismatch detected. Max diff %.2f", max_diff
        )
    else:
        logger.info(
            "Company vs Country net_sales aligned. Max diff %.2f", max_diff or 0.0
        )


def _top_returns(country: pd.DataFrame, n: int = 5) -> None:
    if "returns_value" not in country.columns:
        return
    top = (
        country.sort_values("returns_value", ascending=False)
        .head(n)[["Country", "YearMonth", "returns_value"]]
    )
    logger.info("Top %s country-month returns:\n%s", n, top.to_string(index=False))


def main() -> None:
    data = _read_parquet_map()
    if not data:
        raise FileNotFoundError(f"No parquet outputs found in {PATHS.gold}")

    for name, df in data.items():
        rows = len(df)
        cols = len(df.columns)
        period_min, period_max = _period_summary(df)
        logger.info(
            "[QC] %s rows=%s cols=%s period=(%s → %s)",
            name,
            rows,
            cols,
            period_min.date() if period_min else "-",
            period_max.date() if period_max else "-",
        )
        _check_nan_inf(
            name,
            df,
            [
                "aov",
                "gross_margin_pct",
                "return_rate_value",
                "return_rate_units",
            ],
        )

    _check_country_vs_company(data)
    if "country_monthly_kpis" in data:
        _top_returns(data["country_monthly_kpis"], n=5)


if __name__ == "__main__":
    main()
