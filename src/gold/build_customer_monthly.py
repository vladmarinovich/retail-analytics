"""Build GOLD customer monthly KPIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.metrics import calc_aov, ensure_period
from utils.data import load_transactions
from utils.io import get_paths, logger, write_parquet
from utils.schemas import customer_monthly_kpis_schema

PATHS = get_paths()
OUTPUT_PATH = PATHS.gold / "customer_monthly_kpis.parquet"


def build_customer_monthly() -> pd.DataFrame:
    df = load_transactions()
    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    grouped_all = df.groupby(["CustomerID", "YearMonth"], dropna=False)
    grouped_sales = sales.groupby(["CustomerID", "YearMonth"], dropna=False)
    grouped_returns = returns.groupby(["CustomerID", "YearMonth"], dropna=False)

    monthly = grouped_all.agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    monthly = monthly.merge(
        grouped_sales.agg(
            orders=("InvoiceNo", "nunique"),
            items_sold=("Quantity", "sum"),
            gmv=("Sales", "sum"),
        ).reset_index(),
        on=["CustomerID", "YearMonth"],
        how="left",
    )

    returns_metrics = grouped_returns.agg(
        returns_value=("Sales", lambda s: np.abs(s.sum())),
    ).reset_index()
    monthly = monthly.merge(
        returns_metrics,
        on=["CustomerID", "YearMonth"],
        how="left",
    )

    monthly[["orders"]] = monthly[["orders"]].fillna(0)
    monthly["orders"] = monthly["orders"].astype("Int64")
    for col in ["items_sold", "gmv", "returns_value"]:
        monthly[col] = monthly[col].fillna(0.0)

    monthly = ensure_period(monthly, "YearMonth", "period")
    monthly = calc_aov(
        monthly,
        net_sales_col="net_sales",
        orders_col="orders",
        target_col="aov",
    )

    monthly = monthly.rename(columns={"CustomerID": "customer_id"})
    monthly = monthly.sort_values(["customer_id", "period"]).reset_index(drop=True)

    cols = [
        "period",
        "YearMonth",
        "customer_id",
        "orders",
        "items_sold",
        "gmv",
        "returns_value",
        "net_sales",
        "cogs_net",
        "gp_net",
        "aov",
    ]
    monthly = monthly[cols]

    monthly = customer_monthly_kpis_schema.validate(monthly, lazy=True)

    money_cols = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net", "aov"]
    monthly[money_cols] = monthly[money_cols].round(2)
    monthly["items_sold"] = monthly["items_sold"].round(2)

    return monthly


def main() -> pd.DataFrame:
    monthly = build_customer_monthly()
    write_parquet(monthly, OUTPUT_PATH)
    logger.info("customer_monthly_kpis rows=%s", len(monthly))
    return monthly


if __name__ == "__main__":
    main()
