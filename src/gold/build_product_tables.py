# src/gold/build_product_tables.py
# --------------------------------
# GOLD Products: KPIs lifetime por SKU, serie mensual y clasificación ABC.
# Entrada principal: data/silver/product_monthly.csv
# Fallback (si no existe): data/silver/transactions_base.csv

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SILVER = BASE / "data" / "silver"
GOLD = BASE / "data" / "gold"

PM_PATH = SILVER / "product_monthly.csv"
TX_PATH = SILVER / "transactions_base.csv"
DIM_PROD = SILVER / "dim_product.csv"

OUT_SNAPSHOT = GOLD / "product_kpis.csv"
OUT_MONTHLY = GOLD / "product_monthly_kpis.csv"
OUT_ABC = GOLD / "product_abc.csv"


def ensure_dirs():
    GOLD.mkdir(parents=True, exist_ok=True)


def load_product_monthly():
    """Lee product_monthly de Silver; si no existe, lo construye desde transactions_base."""
    if PM_PATH.exists():
        pm = pd.read_csv(PM_PATH)
        # Normalizaciones suaves (evita recasteos caros si ya viene bien)
        if "YearMonth" in pm.columns and not pm["YearMonth"].dtype.name.startswith("period"):
            # no convertimos a datetime; dejamos 'YYYY-MM' como string para rendimiento
            pass
        pm["StockCode"] = pm["StockCode"].astype(
            "string").str.strip().str.upper()
        return pm

    # Fallback: construir rápido desde transactions_base
    if not TX_PATH.exists():
        raise FileNotFoundError(f"No encuentro {PM_PATH} ni {TX_PATH}")

    # evitamos parse_dates para no recastear; ya viene curado desde Silver
    t = pd.read_csv(TX_PATH)
    if "IsReturn" not in t.columns:
        t["IsReturn"] = t["Quantity"] < 0

    # YearMonth: si no existe, derivamos 'YYYY-MM' sin cast a datetime
    if "YearMonth" not in t.columns:
        t["YearMonth"] = pd.to_datetime(
            t["InvoiceDate"]).dt.to_period("M").astype(str)

    is_sale = ~t["IsReturn"]
    is_ret = t["IsReturn"]

    g = t.groupby(["StockCode", "YearMonth"], dropna=False)
    pm = g.agg(
        units_sold=("Quantity", lambda q: q[is_sale.loc[q.index]].sum()),
        gmv=("Sales", lambda s: s[is_sale.loc[s.index]].sum()),
        return_units_abs=("Quantity", lambda q: np.abs(
            q[is_ret.loc[q.index]].sum())),
        returns_value=("Sales", lambda s: np.abs(
            s[is_ret.loc[s.index]].sum())),
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
        orders=("InvoiceNo", lambda x: x[is_sale.loc[x.index]].nunique()),
        buyers=("CustomerID", lambda c: c[is_sale.loc[c.index]].nunique()),
    ).reset_index()

    # KPI auxiliares
    pm["gross_margin_pct"] = np.where(
        pm["net_sales"] != 0, pm["gp_net"] / pm["net_sales"], np.nan)
    pm["return_rate_units"] = np.where(
        pm["units_sold"] > 0, pm["return_units_abs"] / pm["units_sold"], np.nan)

    pm["StockCode"] = pm["StockCode"].astype("string").str.strip().str.upper()
    return pm


def attach_description(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Intenta añadir 'Description' desde dim_product si existe."""
    if not DIM_PROD.exists():
        snapshot["Description"] = pd.NA
        return snapshot
    dim = pd.read_csv(DIM_PROD)
    dim["StockCode"] = dim["StockCode"].astype(
        "string").str.strip().str.upper()
    keep = dim[["StockCode", "description_mode"]].drop_duplicates()
    return snapshot.merge(keep, on="StockCode", how="left").rename(columns={"description_mode": "Description"})


def build_snapshot(pm: pd.DataFrame) -> pd.DataFrame:
    """Acumula por SKU a lo largo de todo el período."""
    g = pm.groupby("StockCode", dropna=False)
    snap = g.agg(
        first_period=("YearMonth", "min"),
        last_period=("YearMonth", "max"),
        units_sold=("units_sold", "sum"),
        orders=("orders", "sum"),
        buyers=("buyers", "sum"),
        gmv=("gmv", "sum"),
        returns_value=("returns_value", "sum"),
        net_sales=("net_sales", "sum"),
        cogs_net=("cogs_net", "sum"),
        gp_net=("gp_net", "sum"),
        return_units_abs=("return_units_abs", "sum"),
    ).reset_index()

    snap["gross_margin_pct"] = np.where(
        snap["net_sales"] != 0, snap["gp_net"] / snap["net_sales"], np.nan)
    snap["return_rate_units"] = np.where(
        snap["units_sold"] > 0, snap["return_units_abs"] / snap["units_sold"], np.nan)

    # Orden/round
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    snap[money] = snap[money].round(2)
    snap["gross_margin_pct"] = snap["gross_margin_pct"].round(4)
    snap["return_rate_units"] = snap["return_rate_units"].round(4)

    # Adjunta descripción si la tenemos en dim_product
    snap = attach_description(snap)

    cols = ["StockCode", "Description", "first_period", "last_period",
            "units_sold", "orders", "buyers",
            "gmv", "returns_value", "net_sales", "cogs_net", "gp_net",
            "gross_margin_pct", "return_rate_units"]
    return snap[cols].sort_values("net_sales", ascending=False)


def build_abc(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Clasifica productos A/B/C por contribución acumulada a ventas netas."""
    df = snapshot.copy().sort_values(
        "net_sales", ascending=False).reset_index(drop=True)
    total = float(df["net_sales"].clip(lower=0).sum())
    total = total if total > 0 else 1.0
    df["cum_share_net_sales"] = (
        df["net_sales"].clip(lower=0).cumsum() / total)

    def label(p):
        if p <= 0.80:
            return "A"
        if p <= 0.95:
            return "B"
        return "C"

    df["ABC"] = df["cum_share_net_sales"].map(label)
    return df[["StockCode", "ABC", "cum_share_net_sales"]]


def main():
    ensure_dirs()

    pm = load_product_monthly()
    snapshot = build_snapshot(pm)

    # Serie mensual (ya está en pm). Redondeos suaves para presentación
    pm_out = pm.copy()
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    for c in money:
        if c in pm_out.columns:
            pm_out[c] = pm_out[c].round(2)
    for c in ["gross_margin_pct", "return_rate_units"]:
        if c in pm_out.columns:
            pm_out[c] = pm_out[c].round(4)

    abc = build_abc(snapshot)

    snapshot.to_csv(OUT_SNAPSHOT, index=False)
    pm_out.to_csv(OUT_MONTHLY, index=False)
    abc.to_csv(OUT_ABC, index=False)

    print(f"[OK] {OUT_SNAPSHOT}         -> {len(snapshot):,} filas")
    print(f"[OK] {OUT_MONTHLY}          -> {len(pm_out):,} filas")
    print(f"[OK] {OUT_ABC}              -> {len(abc):,} filas")


if __name__ == "__main__":
    main()
