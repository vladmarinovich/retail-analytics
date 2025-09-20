# src/gold/build_country_tables.py
# --------------------------------
# GOLD Country: KPIs por país (snapshot lifetime) y serie mensual.
# Entradas preferidas: data/silver/country_monthly.csv
# Fallback: data/silver/transactions_base.csv
# Salidas:
#   data/gold/country_kpis.csv
#   data/gold/country_monthly_kpis.csv

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SILVER = BASE / "data" / "silver"
GOLD = BASE / "data" / "gold"

CM_PATH = SILVER / "country_monthly.csv"
TX_PATH = SILVER / "transactions_base.csv"

OUT_SNAPSHOT = GOLD / "country_kpis.csv"
OUT_MONTHLY = GOLD / "country_monthly_kpis.csv"


def ensure_dirs():
    GOLD.mkdir(parents=True, exist_ok=True)


def load_country_monthly() -> pd.DataFrame:
    """
    Carga country_monthly desde Silver; si no existe, lo construye rápido desde transactions_base.
    Incluye parches:
      - Renombrar sinónimos (buyers->customers, gp/gross_profit(_net)->gp_net).
      - Derivar cogs_net si falta (net_sales - gp_net) o traerlo desde transactions_base.
    """
    if CM_PATH.exists():
        cm = pd.read_csv(CM_PATH)

        # Normalizaciones suaves
        cm["Country"] = cm["Country"].astype("string").str.strip()
        cm["YearMonth"] = cm["YearMonth"].astype(str)

        # --- Renombra sinónimos esperados ---
        rename_map = {
            "buyers": "customers",
            "gp": "gp_net",
            "gross_profit_net": "gp_net",
            "gross_profit": "gp_net",
        }
        cm = cm.rename(
            columns={k: v for k, v in rename_map.items() if k in cm.columns})

        # --- Si falta cogs_net, intentar derivarlo ---
        if "cogs_net" not in cm.columns:
            if {"net_sales", "gp_net"}.issubset(cm.columns):
                cm["cogs_net"] = cm["net_sales"] - cm["gp_net"]
            else:
                # Fallback: traer cogs_net desde transactions_base
                if not TX_PATH.exists():
                    raise ValueError(
                        "country_monthly no tiene cogs_net ni gp_net y no existe transactions_base para reconstruir."
                    )
                t = pd.read_csv(TX_PATH)
                if "YearMonth" not in t.columns:
                    t["YearMonth"] = pd.to_datetime(
                        t["InvoiceDate"]).dt.to_period("M").astype(str)
                # cogs_net en silver ya es COGS con signo (ventas +, devoluciones -)
                cogs_net_by_cty = (
                    t.groupby(["Country", "YearMonth"])["COGS"].sum()
                    .rename("cogs_net")
                    .reset_index()
                )
                cm = cm.merge(cogs_net_by_cty, on=[
                              "Country", "YearMonth"], how="left")

        # Validación mínima
        needed = {
            "Country", "YearMonth", "gmv", "returns_value", "net_sales", "cogs_net", "gp_net",
            "orders", "customers", "items_sold"
        }
        missing = needed - set(cm.columns)
        if missing:
            raise ValueError(
                f"country_monthly.csv le faltan columnas (tras normalizar): {missing}")

        return cm

    # ---- Fallback: construir desde transactions_base ----
    if not TX_PATH.exists():
        raise FileNotFoundError(f"No encuentro {CM_PATH} ni {TX_PATH}")

    t = pd.read_csv(TX_PATH)
    if "IsReturn" not in t.columns:
        t["IsReturn"] = t["Quantity"] < 0
    if "YearMonth" not in t.columns:
        t["YearMonth"] = pd.to_datetime(
            t["InvoiceDate"]).dt.to_period("M").astype(str)

    t["Country"] = t["Country"].astype("string").str.strip()
    is_sale = ~t["IsReturn"]
    is_ret = t["IsReturn"]

    g = t.groupby(["Country", "YearMonth"], dropna=False)
    cm = g.agg(
        gmv=("Sales", lambda s: s[is_sale.loc[s.index]].sum()),
        returns_value=("Sales", lambda s: np.abs(
            s[is_ret.loc[s.index]].sum())),
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
        orders=("InvoiceNo", lambda x: x[is_sale.loc[x.index]].nunique()),
        customers=("CustomerID", lambda c: c[is_sale.loc[c.index]].nunique()),
        items_sold=("Quantity", lambda q: q[is_sale.loc[q.index]].sum()),
    ).reset_index()

    # métricas derivadas
    cm["gross_margin_pct"] = np.where(
        cm["net_sales"] != 0, cm["gp_net"] / cm["net_sales"], np.nan)
    # participación por mes
    total_mes = cm.groupby("YearMonth")["net_sales"].transform("sum")
    cm["net_sales_share"] = np.where(
        total_mes > 0, cm["net_sales"] / total_mes, np.nan)

    return cm


def build_snapshot(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Snapshot lifetime por país sumando todos los meses.
    """
    g = cm.groupby("Country", dropna=False)
    snap = g.agg(
        first_period=("YearMonth", "min"),
        last_period=("YearMonth", "max"),
        orders=("orders", "sum"),
        buyers=("customers", "sum"),
        items_sold=("items_sold", "sum"),
        gmv=("gmv", "sum"),
        returns_value=("returns_value", "sum"),
        net_sales=("net_sales", "sum"),
        cogs_net=("cogs_net", "sum"),
        gp_net=("gp_net", "sum"),
    ).reset_index()

    snap["gross_margin_pct"] = np.where(
        snap["net_sales"] != 0, snap["gp_net"] / snap["net_sales"], np.nan
    )
    # participación total sobre el período completo
    total_ns = float(snap["net_sales"].clip(lower=0).sum()) or 1.0
    snap["net_sales_share_total"] = snap["net_sales"].clip(lower=0) / total_ns

    # redondeos
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    snap[money] = snap[money].round(2)
    snap["gross_margin_pct"] = snap["gross_margin_pct"].round(4)
    snap["net_sales_share_total"] = snap["net_sales_share_total"].round(4)

    # ordenar por importancia
    snap = snap.sort_values("net_sales", ascending=False)
    cols = [
        "Country", "first_period", "last_period", "orders", "buyers", "items_sold",
        "gmv", "returns_value", "net_sales", "cogs_net", "gp_net",
        "gross_margin_pct", "net_sales_share_total"
    ]
    return snap[cols]


def prepare_monthly(cm: pd.DataFrame) -> pd.DataFrame:
    """
    Serie mensual por país con ratios y crecimiento.
    """
    out = cm.copy()

    # redondeos
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    out[money] = out[money].round(2)
    for c in ["gross_margin_pct", "net_sales_share"]:
        if c in out.columns:
            out[c] = out[c].round(4)

    # crecimiento MoM de ventas netas por país
    out = out.sort_values(["Country", "YearMonth"])
    out["net_sales_mom"] = (
        out.groupby("Country")["net_sales"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .round(4)
    )

    cols = [
        "Country", "YearMonth", "orders", "customers", "items_sold",
        "gmv", "returns_value", "net_sales", "cogs_net", "gp_net",
        "gross_margin_pct", "net_sales_share", "net_sales_mom"
    ]
    return out[cols]


def main():
    ensure_dirs()
    cm = load_country_monthly()
    snap = build_snapshot(cm)
    monthly = prepare_monthly(cm)

    snap.to_csv(OUT_SNAPSHOT, index=False)
    monthly.to_csv(OUT_MONTHLY, index=False)

    print(f"[OK] {OUT_SNAPSHOT}         -> {len(snap):,} filas")
    print(f"[OK] {OUT_MONTHLY}          -> {len(monthly):,} filas")


if __name__ == "__main__":
    main()
