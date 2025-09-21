"""Build GOLD company-level monthly KPIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.metrics import calc_aov, ensure_period, safe_div
from utils.data import load_transactions
from utils.io import get_paths, logger, write_parquet
from utils.schemas import company_monthly_kpis_schema

PATHS = get_paths()
OUTPUT_PATH = PATHS.gold / "company_monthly_kpis.parquet"


def build_company_monthly() -> pd.DataFrame:
    df = load_transactions()
    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    grouped_all = df.groupby("YearMonth", dropna=False)
    grouped_sales = sales.groupby("YearMonth", dropna=False)
    grouped_returns = returns.groupby("YearMonth", dropna=False)

    # Base (ventas netas/costos/margen)
    monthly = grouped_all.agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    # Solo ventas (no devoluciones)
    monthly = monthly.merge(
        grouped_sales.agg(
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", lambda s: s.dropna().nunique()),
            items_sold=("Quantity", "sum"),
            gmv=("Sales", "sum"),
        ).reset_index(),
        on="YearMonth",
        how="left",
    )

    # Solo devoluciones (valor)
    returns_metrics = grouped_returns.agg(
        returns_value=("Sales", lambda s: np.abs(s.sum())),
    ).reset_index()
    monthly = monthly.merge(returns_metrics, on="YearMonth", how="left")

    # NAs y tipos
    for col in ["orders", "customers"]:
        monthly[col] = monthly[col].fillna(0).astype("Int64")
    monthly[["items_sold", "gmv", "returns_value"]] = monthly[
        ["items_sold", "gmv", "returns_value"]
    ].fillna(0.0)

    # Period (DATE) y métricas derivadas
    monthly = ensure_period(monthly, "YearMonth", "period")
    monthly = calc_aov(monthly)  # aov = net_sales / orders (safe)
    monthly["gross_margin_pct"] = safe_div(
        monthly["gp_net"], monthly["net_sales"])
    monthly["return_rate_value"] = safe_div(
        monthly["returns_value"], monthly["gmv"])

    # MoM global
    monthly = monthly.sort_values("period").reset_index(drop=True)
    monthly["net_sales_mom"] = monthly["net_sales"].pct_change()
    monthly["net_sales_mom"] = safe_div(monthly["net_sales_mom"], 1.0)

    # Columnas finales
    cols = [
        "period",
        "YearMonth",
        "orders",
        "customers",
        "items_sold",
        "gmv",
        "returns_value",
        "return_rate_value",
        "net_sales",
        "cogs_net",
        "gp_net",
        "gross_margin_pct",
        "net_sales_mom",
        "aov",
    ]
    monthly = monthly[cols]

    # Validación + redondeos
    monthly = company_monthly_kpis_schema.validate(monthly, lazy=True)

    money_cols = [
        "gmv",
        "returns_value",
        "net_sales",
        "cogs_net",
        "gp_net",
        "aov",
    ]
    monthly[money_cols] = monthly[money_cols].round(2)
    pct_cols = ["gross_margin_pct", "net_sales_mom", "return_rate_value"]
    monthly[pct_cols] = monthly[pct_cols].round(4)
    monthly[pct_cols] = monthly[pct_cols].fillna(0.0)
    monthly["items_sold"] = monthly["items_sold"].round(2)
    monthly["aov"] = monthly["aov"].fillna(0.0)

    return monthly


def main() -> pd.DataFrame:
    monthly = build_company_monthly()
    write_parquet(monthly, OUTPUT_PATH)
    logger.info("company_monthly_kpis rows=%s", len(monthly))
    return monthly


if __name__ == "__main__":
    main()
