"""Build GOLD returns KPI tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data import load_transactions
from utils.io import get_paths, logger, read_csv, write_parquet

PATHS = get_paths()
DEFAULT_INPUT = PATHS.silver / "transactions_base.csv"
GOLD_DIR = PATHS.gold


def load_tx(path: str | None = None) -> pd.DataFrame:
    if path is None or Path(path) == DEFAULT_INPUT:
        df = load_transactions()
    else:
        df = read_csv(path, parse_dates=["InvoiceDate"])
        if "IsReturn" not in df.columns:
            df["IsReturn"] = df["Quantity"] < 0
        if "YearMonth" not in df.columns:
            df["YearMonth"] = pd.to_datetime(df["InvoiceDate"], errors="raise").dt.to_period("M").astype(str)
    df["StockCode"] = df["StockCode"].astype("string").str.strip().str.upper()
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype("string").str.strip()
    return df


def kpis_returns(df: pd.DataFrame):
    """Construye outputs GOLD de devoluciones."""
    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    returns["return_units_abs"] = np.abs(returns["Quantity"])
    returns["returns_value"] = np.abs(returns["Sales"])
    returns["returns_cogs"] = np.abs(returns["COGS"])

    inv = (
        returns.groupby("InvoiceNo").agg(
            InvoiceDate=("InvoiceDate", "min"),
            CustomerID=("CustomerID", "first"),
            Country=("Country", "first"),
            items_distinct=("StockCode", "nunique"),
            return_units_abs=("return_units_abs", "sum"),
            returns_value=("returns_value", "sum"),
            returns_cogs=("returns_cogs", "sum"),
        ).reset_index()
    )
    inv["period"] = pd.to_datetime(inv["InvoiceDate"]).dt.to_period("M").to_timestamp()

    denom_prod = sales.groupby("StockCode").agg(
        units_sold=("Quantity", "sum"),
        gmv=("Sales", "sum"),
    )
    ret_prod = returns.groupby("StockCode").agg(
        return_units_abs=("return_units_abs", "sum"),
        returns_value=("returns_value", "sum"),
        returns_cogs=("returns_cogs", "sum"),
        first_return=("InvoiceDate", "min"),
        last_return=("InvoiceDate", "max"),
    )
    prod = denom_prod.join(ret_prod, how="outer").fillna(0).reset_index()
    try:
        desc_mode = (
            df.groupby("StockCode")["Description"].agg(
                lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA
            )
        )
        prod = prod.merge(desc_mode.rename("Description"), on="StockCode", how="left")
    except Exception:
        prod["Description"] = pd.NA
    prod["return_rate_units"] = np.where(prod["units_sold"] > 0, prod["return_units_abs"] / prod["units_sold"], np.nan)
    prod["return_rate_value"] = np.where(prod["gmv"] > 0, prod["returns_value"] / prod["gmv"], np.nan)

    denom_ctry = sales.groupby("Country").agg(
        units_sold=("Quantity", "sum"),
        gmv=("Sales", "sum"),
        orders=("InvoiceNo", "nunique"),
        buyers=("CustomerID", lambda s: s.dropna().nunique()),
    )
    ret_ctry = returns.groupby("Country").agg(
        return_units_abs=("return_units_abs", "sum"),
        returns_value=("returns_value", "sum"),
        returns_cogs=("returns_cogs", "sum"),
        credit_notes=("InvoiceNo", "nunique"),
    )
    ctry = denom_ctry.join(ret_ctry, how="outer").fillna(0).reset_index()
    ctry["return_rate_units"] = np.where(ctry["units_sold"] > 0, ctry["return_units_abs"] / ctry["units_sold"], np.nan)
    ctry["return_rate_value"] = np.where(ctry["gmv"] > 0, ctry["returns_value"] / ctry["gmv"], np.nan)

    sales["YearMonth"] = sales["YearMonth"].astype(str)
    returns["YearMonth"] = returns["YearMonth"].astype(str)
    denom_m = sales.groupby("YearMonth").agg(
        units_sold=("Quantity", "sum"),
        gmv=("Sales", "sum"),
        orders=("InvoiceNo", "nunique"),
    )
    ret_m = returns.groupby("YearMonth").agg(
        return_units_abs=("return_units_abs", "sum"),
        returns_value=("returns_value", "sum"),
        returns_cogs=("returns_cogs", "sum"),
        credit_notes=("InvoiceNo", "nunique"),
    )
    monthly = (
        denom_m.join(ret_m, how="outer").fillna(0).reset_index().rename(columns={"YearMonth": "period"})
    )
    monthly["period"] = pd.to_datetime(monthly["period"] + "-01")
    monthly["return_rate_units"] = np.where(monthly["units_sold"] > 0, monthly["return_units_abs"] / monthly["units_sold"], np.nan)
    monthly["return_rate_value"] = np.where(monthly["gmv"] > 0, monthly["returns_value"] / monthly["gmv"], np.nan)

    money_cols = ["gmv", "returns_value", "returns_cogs"]
    for df_out in (prod, ctry, monthly):
        df_out[money_cols] = df_out[money_cols].round(2)
    prod[["return_rate_units", "return_rate_value"]] = prod[["return_rate_units", "return_rate_value"]].round(4)
    ctry[["return_rate_units", "return_rate_value"]] = ctry[["return_rate_units", "return_rate_value"]].round(4)
    monthly[["return_rate_units", "return_rate_value"]] = monthly[["return_rate_units", "return_rate_value"]].round(4)

    return inv, prod, ctry, monthly


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(GOLD_DIR))
    args = parser.parse_args(argv)

    df = load_tx(args.inp)
    inv, prod, ctry, monthly = kpis_returns(df)

    outdir = Path(args.outdir)
    write_parquet(inv, outdir / "returns_invoices.parquet")
    write_parquet(prod, outdir / "returns_by_product.parquet")
    write_parquet(ctry, outdir / "returns_by_country.parquet")
    write_parquet(monthly, outdir / "returns_monthly.parquet")

    logger.info(
        "returns_invoices rows=%s | returns_by_product rows=%s | returns_by_country rows=%s | returns_monthly rows=%s",
        len(inv),
        len(prod),
        len(ctry),
        len(monthly),
    )


if __name__ == "__main__":
    main()
