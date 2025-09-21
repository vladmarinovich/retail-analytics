"""Reusable metric helpers for analytics tables."""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(numerator, denominator):
    """Safely divide, returning NaN where denominator is zero or null."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    if isinstance(result, (pd.Series, pd.DataFrame)):
        result = result.replace([np.inf, -np.inf], np.nan)
    elif np.isscalar(result):
        if not np.isfinite(result):
            result = np.nan
    return result


def ensure_period(df: pd.DataFrame, yearmonth_col: str = "YearMonth", period_col: str = "period") -> pd.DataFrame:
    """Ensure a DATE column (first day of month) derived from year-month string."""
    out = df.copy()
    if yearmonth_col not in out.columns:
        raise KeyError(f"Column '{yearmonth_col}' not found in DataFrame")
    ym = out[yearmonth_col].astype(str).str.slice(0, 7)
    out[period_col] = pd.to_datetime(ym + "-01", errors="coerce")
    return out


def calc_aov(df: pd.DataFrame, *, net_sales_col: str = "net_sales", orders_col: str = "orders", target_col: str = "aov") -> pd.DataFrame:
    out = df.copy()
    out[target_col] = safe_div(out[net_sales_col], out[orders_col])
    return out


def calc_return_rate_value(df: pd.DataFrame, *, returns_col: str = "returns_value", base_col: str = "gmv", target_col: str = "return_rate_value") -> pd.DataFrame:
    out = df.copy()
    out[target_col] = safe_div(out[returns_col], out[base_col])
    return out


def calc_return_units(df: pd.DataFrame, *, returns_units_col: str = "return_units_abs", base_units_col: str = "items_sold", target_col: str = "return_rate_units") -> pd.DataFrame:
    out = df.copy()
    out[target_col] = safe_div(out[returns_units_col], out[base_units_col])
    return out


__all__ = [
    "safe_div",
    "ensure_period",
    "calc_aov",
    "calc_return_rate_value",
    "calc_return_units",
]
