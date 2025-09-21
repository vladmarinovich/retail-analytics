"""Build GOLD product KPI tables."""
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
from utils.io import get_paths, logger, read_csv, write_parquet
from utils.schemas import product_monthly_kpis_schema

PATHS = get_paths()
PRODUCT_MONTHLY_PATH = PATHS.gold / "product_monthly_kpis.parquet"
PRODUCT_SNAPSHOT_PATH = PATHS.gold / "product_kpis.parquet"
PRODUCT_ABC_PATH = PATHS.gold / "product_abc.parquet"
DIM_PRODUCT_PATH = PATHS.silver / "dim_product.csv"


def _load_dim_product() -> pd.DataFrame:
    if not DIM_PRODUCT_PATH.exists():
        return pd.DataFrame(columns=["StockCode", "description_mode"])
    dim = read_csv(DIM_PRODUCT_PATH)
    dim["StockCode"] = dim["StockCode"].astype(str).str.strip().str.upper()
    return dim[["StockCode", "description_mode"]].drop_duplicates()


def build_product_monthly(tx: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    sales = tx[~tx["IsReturn"]].copy()
    returns = tx[tx["IsReturn"]].copy()

    grouped_all = tx.groupby(["StockCode", "YearMonth"], dropna=False)
    grouped_sales = sales.groupby(["StockCode", "YearMonth"], dropna=False)
    grouped_returns = returns.groupby(["StockCode", "YearMonth"], dropna=False)

    monthly = grouped_all.agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    monthly = monthly.merge(
        grouped_sales.agg(
            units_sold=("Quantity", "sum"),
            gmv=("Sales", "sum"),
            orders=("InvoiceNo", "nunique"),
            buyers=("CustomerID", lambda s: s.dropna().nunique()),
        ).reset_index(),
        on=["StockCode", "YearMonth"],
        how="left",
    )

    returns_metrics = grouped_returns.agg(
        return_units_abs=("Quantity", lambda q: np.abs(q.sum())),
        returns_value=("Sales", lambda s: np.abs(s.sum())),
    ).reset_index()

    monthly = monthly.merge(
        returns_metrics,
        on=["StockCode", "YearMonth"],
        how="left",
    )
    monthly[["return_units_abs", "returns_value"]] = monthly[
        ["return_units_abs", "returns_value"]
    ].fillna(0.0)

    for col in ["units_sold", "gmv"]:
        monthly[col] = monthly[col].fillna(0.0)
    for col in ["orders", "buyers"]:
        monthly[col] = monthly[col].fillna(0).astype("Int64")

    monthly = ensure_period(monthly, "YearMonth", "period")
    monthly = calc_aov(
        monthly,
        net_sales_col="net_sales",
        orders_col="orders",
        target_col="aov",
    )
    monthly = calc_return_units(
        monthly,
        returns_units_col="return_units_abs",
        base_units_col="units_sold",
    )
    monthly = calc_return_rate_value(monthly)
    monthly["gross_margin_pct"] = safe_div(
        monthly["gp_net"], monthly["net_sales"])

    if not dim.empty:
        monthly = monthly.merge(dim, on="StockCode", how="left")
    else:
        monthly["description_mode"] = pd.NA

    # Orden por SKU y periodo + MoM por SKU
    monthly = monthly.sort_values(
        ["StockCode", "period"]).reset_index(drop=True)
    monthly["net_sales_mom"] = monthly.groupby("StockCode")["net_sales"].pct_change()
    monthly["net_sales_mom"] = safe_div(monthly["net_sales_mom"], 1.0)

    cols = [
        "period",
        "YearMonth",
        "StockCode",
        "description_mode",
        "units_sold",
        "gmv",
        "returns_value",
        "return_units_abs",
        "net_sales",
        "cogs_net",
        "gp_net",
        "orders",
        "buyers",
        "aov",
        "gross_margin_pct",
        "return_rate_units",
        "return_rate_value",
        "net_sales_mom",
    ]
    monthly = monthly[cols]

    monthly = product_monthly_kpis_schema.validate(monthly, lazy=True)

    money_cols = [
        "gmv",
        "returns_value",
        "net_sales",
        "cogs_net",
        "gp_net",
        "aov",
    ]
    monthly[money_cols] = monthly[money_cols].round(2)
    pct_cols = ["gross_margin_pct", "return_rate_units",
                "return_rate_value", "net_sales_mom"]
    monthly[pct_cols] = monthly[pct_cols].round(4).fillna(0.0)
    monthly[["units_sold", "return_units_abs"]] = monthly[
        ["units_sold", "return_units_abs"]
    ].round(2)
    monthly["aov"] = monthly["aov"].fillna(0.0)

    return monthly


def build_product_snapshot(tx: pd.DataFrame, monthly: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    sales = tx[~tx["IsReturn"]].copy()
    returns = tx[tx["IsReturn"]].copy()

    base = tx.groupby("StockCode", dropna=False).agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    sales_agg = sales.groupby("StockCode", dropna=False).agg(
        units_sold=("Quantity", "sum"),
        gmv=("Sales", "sum"),
        orders=("InvoiceNo", "nunique"),
        buyers=("CustomerID", lambda s: s.dropna().nunique()),
    ).reset_index()

    returns_agg = returns.groupby("StockCode", dropna=False).agg(
        returns_value=("Sales", lambda s: np.abs(s.sum())),
        return_units_abs=("Quantity", lambda q: np.abs(q.sum())),
    ).reset_index()

    snap = base.merge(sales_agg, on="StockCode", how="left")
    snap = snap.merge(returns_agg, on="StockCode", how="left")

    snap[["units_sold", "gmv", "orders", "buyers", "returns_value", "return_units_abs"]] = snap[
        ["units_sold", "gmv", "orders", "buyers",
            "returns_value", "return_units_abs"]
    ].fillna(0)
    snap["orders"] = snap["orders"].astype("Int64")
    snap["buyers"] = snap["buyers"].astype("Int64")

    first_last = monthly.groupby("StockCode").agg(
        first_period=("YearMonth", "min"),
        last_period=("YearMonth", "max"),
    ).reset_index()
    snap = snap.merge(first_last, on="StockCode", how="left")

    if not dim.empty:
        snap = snap.merge(dim, on="StockCode", how="left")

    snap["gross_margin_pct"] = safe_div(snap["gp_net"], snap["net_sales"])
    snap["return_rate_units"] = safe_div(
        snap["return_units_abs"], snap["units_sold"])
    snap["return_rate_value"] = safe_div(snap["returns_value"], snap["gmv"])

    money_cols = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    snap[money_cols] = snap[money_cols].round(2)
    snap[["units_sold", "return_units_abs"]] = snap[[
        "units_sold", "return_units_abs"]].round(2)
    snap[["gross_margin_pct", "return_rate_units", "return_rate_value"]] = (
        snap[["gross_margin_pct", "return_rate_units", "return_rate_value"]]
        .round(4)
        .fillna(0.0)
    )

    cols = [
        "StockCode",
        "description_mode",
        "first_period",
        "last_period",
        "orders",
        "buyers",
        "units_sold",
        "return_units_abs",
        "gmv",
        "returns_value",
        "net_sales",
        "cogs_net",
        "gp_net",
        "gross_margin_pct",
        "return_rate_units",
        "return_rate_value",
    ]
    snap = snap[cols].sort_values(
        "net_sales", ascending=False).reset_index(drop=True)
    return snap


def build_product_abc(snapshot: pd.DataFrame) -> pd.DataFrame:
    df = snapshot.copy().sort_values(
        "net_sales", ascending=False).reset_index(drop=True)
    total = float(df["net_sales"].clip(lower=0).sum()) or 1.0
    df["cum_share_net_sales"] = df["net_sales"].clip(lower=0).cumsum() / total

    def label(percent: float) -> str:
        if percent <= 0.80:
            return "A"
        if percent <= 0.95:
            return "B"
        return "C"

    df["ABC"] = df["cum_share_net_sales"].map(label)
    return df[["StockCode", "ABC", "cum_share_net_sales"]]


def build_product_tables() -> dict[str, pd.DataFrame]:
    tx = load_transactions()
    dim = _load_dim_product()
    monthly = build_product_monthly(tx, dim)
    snapshot = build_product_snapshot(tx, monthly, dim)
    abc = build_product_abc(snapshot)
    return {
        "product_monthly_kpis": monthly,
        "product_kpis": snapshot,
        "product_abc": abc,
    }


def main() -> dict[str, pd.DataFrame]:
    outputs = build_product_tables()
    write_parquet(outputs["product_monthly_kpis"], PRODUCT_MONTHLY_PATH)
    write_parquet(outputs["product_kpis"], PRODUCT_SNAPSHOT_PATH)
    write_parquet(outputs["product_abc"], PRODUCT_ABC_PATH)
    logger.info(
        "product_monthly_kpis rows=%s | product_kpis rows=%s | product_abc rows=%s",
        len(outputs["product_monthly_kpis"]),
        len(outputs["product_kpis"]),
        len(outputs["product_abc"]),
    )
    return outputs


if __name__ == "__main__":
    main()
