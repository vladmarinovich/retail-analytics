# src/gold/build_returns_tables.py
# --------------------------------
# GOLD Returns: KPIs de devoluciones por factura, producto, país y mes
# Entrada: data/silver/transactions_base.csv
# Salidas:
#   data/gold/returns_invoices.csv
#   data/gold/returns_by_product.csv
#   data/gold/returns_by_country.csv
#   data/gold/returns_monthly.csv

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SILVER = BASE / "data" / "silver" / "transactions_base.csv"
GOLD_DIR = BASE / "data" / "gold"


def ensure_dirs():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)


def load_tx(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    df = pd.read_csv(path)
    # Supone Silver curado. Solo aseguramos campos mínimos:
    if "IsReturn" not in df.columns:
        df["IsReturn"] = df["Quantity"] < 0
    if "YearMonth" not in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="raise")
        df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    # Normaliza llaves
    df["StockCode"] = df["StockCode"].astype("string").str.strip().str.upper()
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype("string").str.strip()
    return df


def kpis_returns(df: pd.DataFrame):
    """Construye outputs GOLD de devoluciones."""
    sales = df[~df["IsReturn"]].copy()
    returns = df[df["IsReturn"]].copy()

    # Magnitudes positivas para devoluciones
    returns["return_units_abs"] = np.abs(returns["Quantity"])
    returns["returns_value"] = np.abs(returns["Sales"])
    returns["returns_cogs"] = np.abs(returns["COGS"])

    # -------- 1) Por FACTURA de devolución (nota crédito) --------
    inv = (returns.groupby("InvoiceNo").agg(
        InvoiceDate=("InvoiceDate", "min"),
        CustomerID=("CustomerID", "first"),
        Country=("Country", "first"),
        items_distinct=("StockCode", "nunique"),
        return_units_abs=("return_units_abs", "sum"),
        returns_value=("returns_value", "sum"),
        returns_cogs=("returns_cogs", "sum"),
    ).reset_index())
    inv["YearMonth"] = pd.to_datetime(
        inv["InvoiceDate"]).dt.to_period("M").astype(str)

    # -------- 2) Por PRODUCTO (totales lifetime + tasas) --------
    # Denominadores desde ventas
    denom_prod = (sales.groupby("StockCode").agg(
        units_sold=("Quantity", "sum"),
        gmv=("Sales", "sum"),
    ))
    ret_prod = (returns.groupby(["StockCode"]).agg(
        return_units_abs=("return_units_abs", "sum"),
        returns_value=("returns_value", "sum"),
        returns_cogs=("returns_cogs", "sum"),
        first_return=("InvoiceDate", "min"),
        last_return=("InvoiceDate", "max"),
    ))
    prod = denom_prod.join(ret_prod, how="outer").fillna(0).reset_index()
    # Descripción dominante (opcional)
    try:
        desc_mode = (df.groupby("StockCode")["Description"]
                       .agg(lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA))
        prod = prod.merge(desc_mode.rename("Description"),
                          on="StockCode", how="left")
    except Exception:
        prod["Description"] = pd.NA
    prod["return_rate_units"] = np.where(prod["units_sold"] > 0,
                                         prod["return_units_abs"] / prod["units_sold"], np.nan)
    prod["return_rate_value"] = np.where(prod["gmv"] > 0,
                                         prod["returns_value"] / prod["gmv"], np.nan)
    # Orden y redondeos
    money_cols = ["gmv", "returns_value", "returns_cogs"]
    prod[money_cols] = prod[money_cols].round(2)
    prod["return_rate_units"] = prod["return_rate_units"].round(4)
    prod["return_rate_value"] = prod["return_rate_value"].round(4)
    prod = prod[["StockCode", "Description", "units_sold", "gmv",
                 "return_units_abs", "returns_value", "returns_cogs",
                 "return_rate_units", "return_rate_value", "first_return", "last_return"]]

    # -------- 3) Por PAÍS (totales + tasas) --------
    denom_ctry = (sales.groupby("Country").agg(
        units_sold=("Quantity", "sum"),
        gmv=("Sales", "sum"),
        orders=("InvoiceNo", "nunique"),
        buyers=("CustomerID", lambda s: s.dropna().nunique()),
    ))
    ret_ctry = (returns.groupby("Country").agg(
        return_units_abs=("return_units_abs", "sum"),
        returns_value=("returns_value", "sum"),
        returns_cogs=("returns_cogs", "sum"),
        credit_notes=("InvoiceNo", "nunique"),
    ))
    ctry = denom_ctry.join(ret_ctry, how="outer").fillna(0).reset_index()
    ctry["return_rate_units"] = np.where(ctry["units_sold"] > 0,
                                         ctry["return_units_abs"] / ctry["units_sold"], np.nan)
    ctry["return_rate_value"] = np.where(ctry["gmv"] > 0,
                                         ctry["returns_value"] / ctry["gmv"], np.nan)
    ctry[money_cols] = ctry[money_cols].round(2)
    ctry["return_rate_units"] = ctry["return_rate_units"].round(4)
    ctry["return_rate_value"] = ctry["return_rate_value"].round(4)
    ctry = ctry[["Country", "orders", "buyers", "units_sold", "gmv",
                 "return_units_abs", "returns_value", "returns_cogs",
                 "credit_notes", "return_rate_units", "return_rate_value"]]

    # -------- 4) Por MES (totales + tasas) --------
    # Denominadores por mes desde ventas
    sales["YearMonth"] = sales["YearMonth"].astype(str)
    returns["YearMonth"] = returns["YearMonth"].astype(str)
    denom_m = (sales.groupby("YearMonth").agg(
        units_sold=("Quantity", "sum"),
        gmv=("Sales", "sum"),
        orders=("InvoiceNo", "nunique"),
    ))
    ret_m = (returns.groupby("YearMonth").agg(
        return_units_abs=("return_units_abs", "sum"),
        returns_value=("returns_value", "sum"),
        returns_cogs=("returns_cogs", "sum"),
        credit_notes=("InvoiceNo", "nunique"),
    ))
    monthly = (denom_m.join(ret_m, how="outer").fillna(0).reset_index()
               .rename(columns={"YearMonth": "period"}))
    monthly["return_rate_units"] = np.where(monthly["units_sold"] > 0,
                                            monthly["return_units_abs"]/monthly["units_sold"], np.nan)
    monthly["return_rate_value"] = np.where(monthly["gmv"] > 0,
                                            monthly["returns_value"]/monthly["gmv"], np.nan)
    monthly[money_cols] = monthly[money_cols].round(2)
    monthly["return_rate_units"] = monthly["return_rate_units"].round(4)
    monthly["return_rate_value"] = monthly["return_rate_value"].round(4)
    monthly = monthly[["period", "orders", "units_sold", "gmv",
                       "return_units_abs", "returns_value", "returns_cogs",
                       "credit_notes", "return_rate_units", "return_rate_value"]]

    return inv, prod, ctry, monthly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(SILVER))
    ap.add_argument("--outdir", default=str(GOLD_DIR))
    args = ap.parse_args()

    df = load_tx(Path(args.inp))
    ensure_dirs()
    inv, prod, ctry, monthly = kpis_returns(df)

    outdir = Path(args.outdir)
    inv.to_csv(outdir / "returns_invoices.csv",
               index=False, date_format="%Y-%m-%d")
    prod.to_csv(outdir / "returns_by_product.csv",
                index=False, date_format="%Y-%m-%d")
    ctry.to_csv(outdir / "returns_by_country.csv", index=False)
    monthly.to_csv(outdir / "returns_monthly.csv", index=False)

    print(f"[OK] returns_invoices.csv   -> {len(inv):,} filas")
    print(f"[OK] returns_by_product.csv -> {len(prod):,} filas")
    print(f"[OK] returns_by_country.csv -> {len(ctry):,} filas")
    print(f"[OK] returns_monthly.csv    -> {len(monthly):,} filas")


if __name__ == "__main__":
    main()
