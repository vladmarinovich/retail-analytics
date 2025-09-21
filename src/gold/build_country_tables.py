"""Build GOLD country-level KPI tables (monthly + snapshot)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# ⬇️ utilidades que ya tienes en el repo
from features.metrics import (
    calc_aov,                 # aov = net_sales / orders (safe)
    calc_return_rate_value,   # returns_value / gmv (safe)
    calc_return_units,        # añade return_units_abs y return_rate_units
    ensure_period,            # añade period (DATE) desde YearMonth
    safe_div,                 # división segura
)
from utils.data import load_transactions
from utils.io import get_paths, logger, write_parquet
from utils.schemas import country_monthly_kpis_schema

PATHS = get_paths()
COUNTRY_MONTHLY_PATH = PATHS.gold / "country_monthly_kpis.parquet"
COUNTRY_SNAPSHOT_PATH = PATHS.gold / "country_kpis.parquet"


def _normalize_country(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza país nulo
    out = df.copy()
    out["Country"] = out["Country"].fillna("Unspecified")
    return out


def build_country_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_country(df)

    # Split ventas vs devoluciones
    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    # Agrupaciones
    g_all = df.groupby(["Country", "YearMonth"], dropna=False)
    g_sales = sales.groupby(["Country", "YearMonth"], dropna=False)
    g_ret = returns.groupby(["Country", "YearMonth"], dropna=False)

    # Base neta (net_sales/cogs/gp incluyen signo)
    monthly = g_all.agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    # Solo ventas (sin devoluciones)
    monthly = monthly.merge(
        g_sales.agg(
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", lambda s: s.dropna().nunique()),
            items_sold=("Quantity", "sum"),
            gmv=("Sales", "sum"),
        ).reset_index(),
        on=["Country", "YearMonth"],
        how="left",
    )

    # Solo devoluciones (valor y unidades)
    ret_metrics = g_ret.agg(
        returns_value=("Sales", lambda s: np.abs(s.sum())),
        return_units_abs=("Quantity", lambda q: np.abs(q.sum())),
    ).reset_index()

    monthly = monthly.merge(
        ret_metrics, on=["Country", "YearMonth"], how="left")

    # NAs y tipos
    monthly[["returns_value", "return_units_abs"]] = monthly[
        ["returns_value", "return_units_abs"]
    ].fillna(0.0)

    for col in ["orders", "customers"]:
        monthly[col] = monthly[col].fillna(0).astype("Int64")
    for col in ["items_sold", "gmv"]:
        monthly[col] = monthly[col].fillna(0.0).astype(float)

    # period (DATE) + métricas derivadas
    monthly = ensure_period(monthly, "YearMonth",
                            "period")        # YYYY-MM-01 (date)
    monthly = calc_aov(monthly)                                    # aov
    # returns_value / gmv
    monthly = calc_return_rate_value(monthly)
    # return_units_abs + return_rate_units
    monthly = calc_return_units(monthly)

    # Margen %
    monthly["gross_margin_pct"] = safe_div(
        monthly["gp_net"], monthly["net_sales"]
    )

    # MoM por país
    monthly = monthly.sort_values(["Country", "period"]).reset_index(drop=True)
    monthly["net_sales_mom"] = monthly.groupby("Country")["net_sales"].pct_change()
    monthly["net_sales_mom"] = safe_div(monthly["net_sales_mom"], 1.0)

    # Share por mes (evita /0)
    total_ns_m = monthly.groupby("YearMonth")["net_sales"].transform("sum")
    monthly["net_sales_share"] = safe_div(monthly["net_sales"], total_ns_m)

    # Orden de columnas
    cols = [
        "period", "YearMonth", "Country",
        "orders", "customers", "items_sold",
        "gmv", "returns_value", "return_units_abs",
        "net_sales", "cogs_net", "gp_net",
        "gross_margin_pct", "net_sales_share", "net_sales_mom",
        "aov", "return_rate_value", "return_rate_units",
    ]
    monthly = monthly[cols]

    # Validación + redondeos
    monthly = country_monthly_kpis_schema.validate(monthly, lazy=True)

    money_cols = ["gmv", "returns_value",
                  "net_sales", "cogs_net", "gp_net", "aov"]
    monthly[money_cols] = monthly[money_cols].round(2)
    pct_cols = ["gross_margin_pct", "net_sales_share", "net_sales_mom",
                "return_rate_value", "return_rate_units"]
    monthly[pct_cols] = monthly[pct_cols].round(4).fillna(0.0)
    monthly["return_units_abs"] = monthly["return_units_abs"].round(2)
    monthly["items_sold"] = monthly["items_sold"].round(2)
    monthly["aov"] = monthly["aov"].fillna(0.0)

    return monthly


def build_country_snapshot(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_country(df)

    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    # Base neta (lifetime)
    base = df.groupby("Country", dropna=False).agg(
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()

    # Solo ventas
    sales_agg = sales.groupby("Country", dropna=False).agg(
        gmv=("Sales", "sum"),
        items_sold=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
        buyers=("CustomerID", lambda s: s.dropna().nunique()),
    ).reset_index()

    # Solo devoluciones
    ret_agg = returns.groupby("Country", dropna=False).agg(
        returns_value=("Sales", lambda s: np.abs(s.sum())),
        return_units_abs=("Quantity", lambda q: np.abs(q.sum())),
    ).reset_index()

    snap = base.merge(sales_agg, on="Country", how="left").merge(
        ret_agg, on="Country", how="left")

    # NAs
    snap[["gmv", "items_sold", "orders", "buyers", "returns_value", "return_units_abs"]] = snap[
        ["gmv", "items_sold", "orders", "buyers",
            "returns_value", "return_units_abs"]
    ].fillna(0)
    snap["orders"] = snap["orders"].astype("Int64")
    snap["buyers"] = snap["buyers"].astype("Int64")

    # Rango de meses
    first_last = monthly.groupby("Country", dropna=False).agg(
        first_period=("YearMonth", "min"),
        last_period=("YearMonth", "max"),
    ).reset_index()
    snap = snap.merge(first_last, on="Country", how="left")

    # Margen % y share total
    snap["gross_margin_pct"] = safe_div(snap["gp_net"], snap["net_sales"])
    total_ns = snap["net_sales"].clip(lower=0).sum() or 1.0
    snap["net_sales_share_total"] = safe_div(
        snap["net_sales"].clip(lower=0), total_ns)

    # Redondeos + orden
    money_cols = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    snap[money_cols] = snap[money_cols].round(2)
    snap[["items_sold", "return_units_abs"]] = snap[[
        "items_sold", "return_units_abs"]].round(2)
    snap[["gross_margin_pct", "net_sales_share_total"]] = (
        snap[["gross_margin_pct", "net_sales_share_total"]]
        .round(4)
        .fillna(0.0)
    )

    cols = [
        "Country", "first_period", "last_period",
        "orders", "buyers", "items_sold", "return_units_abs",
        "gmv", "returns_value", "net_sales", "cogs_net", "gp_net",
        "gross_margin_pct", "net_sales_share_total",
    ]
    snap = snap[cols].sort_values(
        "net_sales", ascending=False).reset_index(drop=True)
    return snap


def build_country_tables() -> dict[str, pd.DataFrame]:
    tx = load_transactions()
    monthly = build_country_monthly(tx)
    snapshot = build_country_snapshot(tx, monthly)
    return {"country_monthly_kpis": monthly, "country_kpis": snapshot}


def main() -> dict[str, pd.DataFrame]:
    out = build_country_tables()
    write_parquet(out["country_monthly_kpis"], COUNTRY_MONTHLY_PATH)
    write_parquet(out["country_kpis"], COUNTRY_SNAPSHOT_PATH)
    logger.info(
        "country_monthly_kpis rows=%s | country_kpis rows=%s",
        len(out["country_monthly_kpis"]), len(out["country_kpis"])
    )
    return out


if __name__ == "__main__":
    main()
