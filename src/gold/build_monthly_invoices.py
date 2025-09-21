"""Generate GOLD revenue monthly KPIs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features.metrics import ensure_period, safe_div
from utils.data import load_transactions
from utils.io import get_paths, logger, write_parquet

PATHS = get_paths()
DEFAULT_INPUT = PATHS.silver / "transactions_base.csv"
DEFAULT_OUTPUT = PATHS.gold / "revenue_monthly.parquet"


def build_revenue_monthly(df: pd.DataFrame | None = None) -> pd.DataFrame:
    tx = df.copy() if df is not None else load_transactions()

    if "InvoiceDate" not in tx.columns:
        raise ValueError("transactions_base must include 'InvoiceDate'.")

    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"], errors="raise")
    tx["YearMonth"] = tx["InvoiceDate"].dt.to_period("M").astype(str)

    if "IsReturn" not in tx.columns:
        tx["IsReturn"] = tx["Quantity"] < 0

    sales = tx[~tx["IsReturn"]].copy()
    returns = tx[tx["IsReturn"]].copy()

    sales["sales_gross_line"] = sales["Quantity"] * sales["UnitPrice"]
    sales["cogs_line"] = sales["Quantity"] * sales["UnitCost"]
    returns["returns_gross_ln"] = np.abs(returns["Quantity"] * returns["UnitPrice"])
    returns["cogs_ret_ln"] = np.abs(returns["Quantity"] * returns["UnitCost"])

    sales_agg = sales.groupby("YearMonth").agg(
        sales_gross=("sales_gross_line", "sum"),
        cogs_sales=("cogs_line", "sum"),
        orders=("InvoiceNo", "nunique"),
    )

    returns_agg = returns.groupby("YearMonth").agg(
        returns_gross=("returns_gross_ln", "sum"),
        cogs_returns=("cogs_ret_ln", "sum"),
        credit_notes=("InvoiceNo", "nunique"),
    )

    monthly = (
        sales_agg.join(returns_agg, how="outer")
        .fillna(0.0)
        .reset_index()
        .sort_values("YearMonth")
    )

    monthly["net_sales"] = monthly["sales_gross"] - monthly["returns_gross"]
    monthly["net_cogs"] = monthly["cogs_sales"] - monthly["cogs_returns"]
    monthly["gross_profit"] = monthly["net_sales"] - monthly["net_cogs"]
    monthly["margin_pct"] = safe_div(monthly["gross_profit"], monthly["net_sales"])
    monthly["return_rate_pct"] = safe_div(monthly["returns_gross"], monthly["sales_gross"])

    monthly = ensure_period(monthly, "YearMonth", "period")

    money_cols = [
        "sales_gross",
        "returns_gross",
        "net_sales",
        "cogs_sales",
        "cogs_returns",
        "net_cogs",
        "gross_profit",
    ]
    monthly[money_cols] = monthly[money_cols].round(2)
    monthly[["margin_pct", "return_rate_pct"]] = monthly[["margin_pct", "return_rate_pct"]].round(4)

    col_order = [
        "period",
        "YearMonth",
        "sales_gross",
        "returns_gross",
        "net_sales",
        "cogs_sales",
        "cogs_returns",
        "net_cogs",
        "gross_profit",
        "margin_pct",
        "return_rate_pct",
        "orders",
        "credit_notes",
    ]
    for col in ["orders", "credit_notes"]:
        if col not in monthly.columns:
            monthly[col] = 0
        monthly[col] = monthly[col].astype("Int64")
    monthly = monthly[col_order]

    return monthly


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Build revenue monthly KPIs")
    parser.add_argument("--inp", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    if str(DEFAULT_INPUT) == args.inp:
        monthly = build_revenue_monthly()
    else:
        monthly = build_revenue_monthly(pd.read_csv(args.inp, parse_dates=["InvoiceDate"]))

    out_path = Path(args.out)
    write_parquet(monthly, out_path)
    logger.info("revenue_monthly rows=%s", len(monthly))
    return monthly


if __name__ == "__main__":
    main()
