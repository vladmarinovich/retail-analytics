"""Build GOLD country-level KPI tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.metrics import (
    calc_aov,
    calc_return_rate_value,
    calc_return_units,
    ensure_period,
    safe_div,
)
from utils.data import load_transactions
from utils.io import get_paths, logger, write_parquet
from utils.schemas import country_monthly_kpis_schema

PATHS = get_paths()
COUNTRY_MONTHLY_PATH = PATHS.gold / "country_monthly_kpis.parquet"
COUNTRY_SNAPSHOT_PATH = PATHS.gold / "country_kpis.parquet"


def build_country_monthly(df: pd.DataFrame) -> pd.DataFrame:
    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    grouped_sales = sales.groupby(["Country", "YearMonth"], dropna=False)
    grouped_returns = returns.groupby(["Country", "YearMonth"], dropna=False)
    grouped_all = df.groupby(["Country", "YearMonth"], dropna=False)

    monthly = grouped_all.agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    monthly = monthly.merge(
        grouped_sales.agg(
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", lambda s: s.dropna().nunique()),
            items_sold=("Quantity", "sum"),
            gmv=("Sales", "sum"),
        ).reset_index(),
        on=["Country", "YearMonth"],
        how="left",
    )

    returns_metrics = grouped_returns.agg(
        returns_value=("Sales", lambda s: np.abs(s.sum())),
        return_units_abs=("Quantity", lambda q: np.abs(q.sum())),
    ).reset_index()
    monthly = monthly.merge(
        returns_metrics,
        on=["Country", "YearMonth"],
        how="left",
    )
    monthly[["returns_value", "return_units_abs"]] = monthly[[
        "returns_value", "return_units_abs"
    ]].fillna(0.0)

    fill_zero = ["orders", "customers", "items_sold", "gmv"]
    for col in fill_zero:
        monthly[col] = monthly[col].fillna(0)
    monthly["orders"] = monthly["orders"].astype("Int64")
    monthly["customers"] = monthly["customers"].astype("Int64")
    monthly["items_sold"] = monthly["items_sold"].astype(float)
    monthly["gmv"] = monthly["gmv"].astype(float)

    monthly = ensure_period(monthly, "YearMonth", "period")
    monthly = calc_aov(monthly)
    monthly = calc_return_rate_value(monthly)
    monthly = calc_return_units(monthly)

    monthly["gross_margin_pct"] = safe_div(monthly["gp_net"], monthly["net_sales"])

    monthly = monthly.sort_values(["Country", "period"]).reset_index(drop=True)
    monthly["net_sales_mom"] = (
        monthly.groupby("Country")["net_sales"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )
    total_per_month = monthly.groupby("YearMonth")["net_sales"].transform("sum")
    monthly["net_sales_share"] = safe_div(monthly["net_sales"], total_per_month)

    # Standardize column order
    cols = [
        "period",
        "YearMonth",
        "Country",
        "orders",
        "customers",
        "items_sold",
        "gmv",
        "returns_value",
        "return_units_abs",
        "net_sales",
        "cogs_net",
        "gp_net",
        "gross_margin_pct",
        "net_sales_share",
        "net_sales_mom",
        "aov",
        "return_rate_value",
        "return_rate_units",
    ]
    monthly = monthly[cols]

    monthly = country_monthly_kpis_schema.validate(monthly, lazy=True)

    money_cols = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net", "aov"]
    monthly[money_cols] = monthly[money_cols].round(2)
    pct_cols = [
        "gross_margin_pct",
        "net_sales_share",
        "net_sales_mom",
        "return_rate_value",
        "return_rate_units",
    ]
    monthly[pct_cols] = monthly[pct_cols].round(4)
    monthly["return_units_abs"] = monthly["return_units_abs"].round(2)
    monthly["items_sold"] = monthly["items_sold"].round(2)

    return monthly


def build_country_snapshot(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    base = df.groupby("Country", dropna=False).agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    sales_agg = sales.groupby("Country", dropna=False).agg(
        gmv=("Sales", "sum"),
        items_sold=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
        buyers=("CustomerID", lambda s: s.dropna().nunique()),
    ).reset_index()

    returns_agg = returns.groupby("Country", dropna=False).agg(
        returns_value=("Sales", lambda s: np.abs(s.sum())),
        return_units_abs=("Quantity", lambda q: np.abs(q.sum())),
    ).reset_index()

    snap = base.merge(sales_agg, on="Country", how="left")
    snap = snap.merge(returns_agg, on="Country", how="left")

    snap[["gmv", "items_sold", "orders", "buyers", "returns_value", "return_units_abs"]] = snap[[
        "gmv", "items_sold", "orders", "buyers", "returns_value", "return_units_abs"
    ]].fillna(0)
    snap["orders"] = snap["orders"].astype("Int64")
    snap["buyers"] = snap["buyers"].astype("Int64")

    first_last = monthly.groupby("Country").agg(
        first_period=("YearMonth", "min"),
        last_period=("YearMonth", "max"),
    ).reset_index()
    snap = snap.merge(first_last, on="Country", how="left")

    snap["gross_margin_pct"] = safe_div(snap["gp_net"], snap["net_sales"])
    total_ns = snap["net_sales"].clip(lower=0).sum() or 1.0
    snap["net_sales_share_total"] = safe_div(snap["net_sales"].clip(lower=0), total_ns)

    money_cols = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    snap[money_cols] = snap[money_cols].round(2)
    snap[["items_sold", "return_units_abs"]] = snap[["items_sold", "return_units_abs"]].round(2)
    snap[["gross_margin_pct", "net_sales_share_total"]] = snap[[
        "gross_margin_pct", "net_sales_share_total"
    ]].round(4)

    cols = [
        "Country",
        "first_period",
        "last_period",
        "orders",
        "buyers",
        "items_sold",
        "return_units_abs",
        "gmv",
        "returns_value",
        "net_sales",
        "cogs_net",
        "gp_net",
        "gross_margin_pct",
        "net_sales_share_total",
    ]
    snap = snap[cols].sort_values("net_sales", ascending=False).reset_index(drop=True)
    return snap


def build_country_tables() -> dict[str, pd.DataFrame]:
    tx = load_transactions()
    monthly = build_country_monthly(tx)
    snapshot = build_country_snapshot(tx, monthly)
    return {
        "country_monthly_kpis": monthly,
        "country_kpis": snapshot,
    }


def main() -> dict[str, pd.DataFrame]:
    outputs = build_country_tables()
    write_parquet(outputs["country_monthly_kpis"], COUNTRY_MONTHLY_PATH)
    write_parquet(outputs["country_kpis"], COUNTRY_SNAPSHOT_PATH)
    logger.info(
        "country_monthly_kpis rows=%s | country_kpis rows=%s",
        len(outputs["country_monthly_kpis"]),
        len(outputs["country_kpis"]),
    )
    return outputs


if __name__ == "__main__":
    main()
