# src/gold/build_customer_tables.py
# ---------------------------------
# Lee tablas SILVER de clientes y produce tablas GOLD de KPIs y Retención.
#
# Entradas (silver):
#   data/silver/customers_snapshot.csv   -> métricas lifetime por cliente
#   data/silver/customers_monthly.csv    -> métricas por cliente y mes (YYYY-MM)
#
# Salidas (gold):
#   data/gold/customer_kpis.csv
#   data/gold/customer_retention_monthly.csv

from pathlib import Path
import os
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SILVER_DIR = BASE / "data" / "silver"
GOLD_DIR = BASE / "data" / "gold"

SNAP_PATH = SILVER_DIR / "customers_snapshot.csv"
MONTHLY_PATH = SILVER_DIR / "customers_monthly.csv"


def ensure_dirs():
    GOLD_DIR.mkdir(parents=True, exist_ok=True)


def read_silver():
    if not SNAP_PATH.exists():
        raise FileNotFoundError(f"No encuentro {SNAP_PATH}")
    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(f"No encuentro {MONTHLY_PATH}")

    snap = pd.read_csv(
        SNAP_PATH,
        parse_dates=["first_purchase", "last_purchase"],
        dayfirst=False
    )

    monthly = pd.read_csv(
        MONTHLY_PATH,
        parse_dates=["last_purchase"],
        dayfirst=False
    )
    # Normalizaciones suaves
    snap["CustomerID"] = snap["CustomerID"].astype("string").str.strip()
    monthly["CustomerID"] = monthly["CustomerID"].astype("string").str.strip()
    return snap, monthly

# ---------- GOLD: KPIs por cliente (RFM, CLV, churn risk) ----------


def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula scores R, F, M (1..5). R invertido (1 peor recency, 5 mejor)."""
    out = df.copy()

    # Evitar colisiones por empates en qcut
    def safe_qcut(s, q, labels):
        # Si hay pocos valores únicos, degradamos a rank percentil
        if s.nunique(dropna=True) < q:
            ranks = s.rank(method="average", pct=True)
            bins = pd.qcut(ranks, q, labels=labels, duplicates="drop")
            return bins.astype(int)
        return pd.qcut(s, q, labels=labels, duplicates="drop").astype(int)

    # Recency: menos días -> mejor score
    # Creamos una versión "invertida": mayor = mejor
    # Para qcut usamos el negativo para que 5 = más reciente
    out["R_score"] = safe_qcut(-out["recency_days"].fillna(out["recency_days"].max()),
                               5, labels=[1, 2, 3, 4, 5])
    out["F_score"] = safe_qcut(
        out["frequency"].fillna(0), 5, labels=[1, 2, 3, 4, 5])
    out["M_score"] = safe_qcut(
        out["monetary"].fillna(0), 5, labels=[1, 2, 3, 4, 5])

    out["RFM_score"] = out["R_score"]*100 + out["F_score"]*10 + out["M_score"]

    # Segmento bonito básico
    def segment_row(r, f, m):
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal"
        if r >= 3 and m >= 4:
            return "Big Spenders"
        if r <= 2 and f <= 2:
            return "At Risk"
        if r >= 4 and f <= 2:
            return "Potential Loyalist"
        return "Regular"

    out["segment"] = [segment_row(r, f, m) for r, f, m in zip(
        out["R_score"], out["F_score"], out["M_score"])]
    return out


def estimate_clv(monthly: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """
    CLV simple: promedio de ventas netas de los últimos 3 meses * horizonte.
    Si un cliente no tiene 3 meses, se promedia lo disponible.
    """
    m = monthly.copy()
    # Orden temporal
    m["YearMonth"] = pd.to_datetime(m["YearMonth"] + "-01")
    m = m.sort_values(["CustomerID", "YearMonth"])

    # Últimos 3 meses por cliente
    m["net_sales_last3m_avg"] = (
        m.groupby("CustomerID")["net_sales"]
         .rolling(3, min_periods=1)
         .mean()
         .reset_index(level=0, drop=True)
    )
    clv = (
        m.groupby("CustomerID")
         .tail(1)[["CustomerID", "net_sales_last3m_avg"]]
         .rename(columns={"net_sales_last3m_avg": "clv_monthly_avg"})
    )
    clv["clv_12m_est"] = clv["clv_monthly_avg"].clip(lower=0) * horizon_months
    return clv


def churn_risk_from_recency(recency_days: pd.Series) -> pd.Series:
    """Clasifica riesgo de churn según recency (ajusta umbrales si quieres)."""
    bins = [-np.inf, 30, 90, 180, np.inf]
    labels = ["Low", "Medium", "High", "Very High"]
    return pd.cut(recency_days, bins=bins, labels=labels)


def build_customer_kpis(snapshot: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    snap = snapshot.copy()

    # Scores RFM y segmento
    snap = rfm_scores(snap)

    # CLV estimado desde el monthly
    clv = estimate_clv(monthly, horizon_months=12)
    out = snap.merge(clv, on="CustomerID", how="left")

    # Riesgo de churn
    out["churn_risk"] = churn_risk_from_recency(out["recency_days"])

    # Columnas ordenadas
    cols = [
        "CustomerID", "first_purchase", "last_purchase",
        "recency_days", "recency_bucket", "churn_risk",
        "gmv", "returns_value", "net_sales", "cogs_net", "gross_profit_net",
        "orders", "items_sold", "aov",
        "frequency", "monetary",
        "gross_margin_pct",
        "R_score", "F_score", "M_score", "RFM_score", "segment",
        "clv_monthly_avg", "clv_12m_est",
    ]
    # Asegurar columnas ausentes con NaN
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]

# ---------- GOLD: Retención mensual ----------


def monthly_retention_table(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula por mes:
      - active_customers  : clientes con actividad ese mes
      - new_customers     : primera compra ocurre ese mes
      - retained          : también estuvieron activos el mes previo
      - reactivated       : activos este mes, no el mes previo, pero sí en meses anteriores
      - churned           : activos mes previo y NO activos este mes
    """
    m = monthly.copy()
    m["YearMonth"] = pd.PeriodIndex(m["YearMonth"], freq="M").to_timestamp()
    m = m.sort_values(["CustomerID", "YearMonth"])

    # Primer mes de cada cliente (para detectar "new")
    first_month = m.groupby("CustomerID")[
        "YearMonth"].min().rename("first_month")
    m = m.merge(first_month, on="CustomerID", how="left")

    # Activo por cliente/mes
    active = m.groupby(["YearMonth", "CustomerID"]
                       ).size().reset_index(name="active")
    active["active"] = 1

    # Tabla pivote de actividad (0/1) por cliente x mes
    pivot = active.pivot(index="CustomerID", columns="YearMonth",
                         values="active").fillna(0).astype(int)
    months = list(pivot.columns)

    rows = []
    for i, month in enumerate(months):
        current = pivot[month] == 1
        active_customers = int(current.sum())

        # Nuevos: primera actividad de ese cliente justo en este mes
        # (comparando con primera aparición en la tabla m)
        is_new = (first_month == month).reindex(pivot.index).fillna(False)
        new_customers = int((current & is_new).sum())

        # Retenidos: activos este mes y también en el inmediato anterior
        if i > 0:
            prev_month = months[i-1]
            retained = int(((pivot[prev_month] == 1) & current).sum())
            churned = int(((pivot[prev_month] == 1) & (~current)).sum())
        else:
            retained = 0
            churned = 0

        # Reactivados: activos este mes, NO activos el mes anterior,
        # pero con algún 1 en meses anteriores (histórico)
        if i > 0:
            any_past = (pivot.iloc[:, :i].sum(axis=1) > 0)
            reactivated = int(
                (current & (pivot[months[i-1]] == 0) & any_past).sum())
        else:
            reactivated = 0

        rows.append({
            "YearMonth": month.strftime("%Y-%m"),
            "active_customers": active_customers,
            "new_customers": new_customers,
            "retained": retained,
            "reactivated": reactivated,
            "churned": churned
        })

    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    snapshot, monthly = read_silver()

    kpis = build_customer_kpis(snapshot, monthly)
    retention = monthly_retention_table(monthly)

    kpis.to_csv(GOLD_DIR / "customer_kpis.csv",
                index=False, date_format="%Y-%m-%d")
    retention.to_csv(GOLD_DIR / "customer_retention_monthly.csv", index=False)

    print(f"[OK] customer_kpis.csv               -> {len(kpis):,} filas")
    print(f"[OK] customer_retention_monthly.csv  -> {len(retention):,} filas")


if __name__ == "__main__":
    main()
