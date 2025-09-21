"""Build GOLD customer KPI and retention tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.io import get_paths, read_csv, write_parquet, logger

PATHS = get_paths()
SILVER_DIR = PATHS.silver
GOLD_DIR = PATHS.gold

SNAP_PATH = SILVER_DIR / "customers_snapshot.csv"
MONTHLY_PATH = SILVER_DIR / "customers_monthly.csv"


# ---------- GOLD: KPIs por cliente (RFM, CLV, churn risk) ----------


def read_silver():
    if not SNAP_PATH.exists():
        raise FileNotFoundError(f"No encuentro {SNAP_PATH}")
    if not MONTHLY_PATH.exists():
        raise FileNotFoundError(f"No encuentro {MONTHLY_PATH}")

    snap = read_csv(
        SNAP_PATH,
        parse_dates=["first_purchase", "last_purchase"],
        dayfirst=False,
    )

    monthly = read_csv(
        MONTHLY_PATH,
        parse_dates=["last_purchase"],
        dayfirst=False,
    )
    # Normalizaciones suaves
    snap["CustomerID"] = snap["CustomerID"].astype("string").str.strip()
    monthly["CustomerID"] = monthly["CustomerID"].astype("string").str.strip()
    return snap, monthly


def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula scores R, F, M (1..5). R invertido (1 peor recency, 5 mejor)."""
    out = df.copy()

    def safe_qcut(s, q, labels):
        if s.nunique(dropna=True) < q:
            ranks = s.rank(method="average", pct=True)
            bins = pd.qcut(ranks, q, labels=labels, duplicates="drop")
            return bins.astype(int)
        return pd.qcut(s, q, labels=labels, duplicates="drop").astype(int)

    out["R_score"] = safe_qcut(
        -out["recency_days"].fillna(out["recency_days"].max()), 5, labels=[1, 2, 3, 4, 5]
    )
    out["F_score"] = safe_qcut(out["frequency"].fillna(0), 5, labels=[1, 2, 3, 4, 5])
    out["M_score"] = safe_qcut(out["monetary"].fillna(0), 5, labels=[1, 2, 3, 4, 5])

    out["RFM_score"] = out["R_score"] * 100 + out["F_score"] * 10 + out["M_score"]

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

    out["segment"] = [segment_row(r, f, m) for r, f, m in zip(out["R_score"], out["F_score"], out["M_score"])]
    return out


def estimate_clv(monthly: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    """CLV simple: promedio de ventas netas de los últimos 3 meses * horizonte."""
    m = monthly.copy()
    m["YearMonth"] = pd.to_datetime(m["YearMonth"] + "-01")
    m = m.sort_values(["CustomerID", "YearMonth"])
    m["net_sales_last3m_avg"] = (
        m.groupby("CustomerID")["net_sales"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    clv = (
        m.groupby("CustomerID")
        .tail(1)[["CustomerID", "net_sales_last3m_avg"]]
        .rename(columns={"net_sales_last3m_avg": "clv_monthly_avg"})
    )
    clv["clv_12m_est"] = clv["clv_monthly_avg"].clip(lower=0) * horizon_months
    return clv


def churn_risk_from_recency(recency_days: pd.Series) -> pd.Series:
    """Clasifica riesgo de churn según recency."""
    bins = [-np.inf, 30, 90, 180, np.inf]
    labels = ["Low", "Medium", "High", "Very High"]
    return pd.cut(recency_days, bins=bins, labels=labels)


def build_customer_kpis(snapshot: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    snap = snapshot.copy()
    snap = rfm_scores(snap)
    clv = estimate_clv(monthly, horizon_months=12)
    out = snap.merge(clv, on="CustomerID", how="left")
    out["churn_risk"] = churn_risk_from_recency(out["recency_days"])

    cols = [
        "CustomerID",
        "first_purchase",
        "last_purchase",
        "recency_days",
        "recency_bucket",
        "churn_risk",
        "gmv",
        "returns_value",
        "net_sales",
        "cogs_net",
        "gross_profit_net",
        "orders",
        "items_sold",
        "aov",
        "frequency",
        "monetary",
        "gross_margin_pct",
        "R_score",
        "F_score",
        "M_score",
        "RFM_score",
        "segment",
        "clv_monthly_avg",
        "clv_12m_est",
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


# ---------- GOLD: Retención mensual ----------


def monthly_retention_table(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()
    m["YearMonth"] = pd.PeriodIndex(m["YearMonth"], freq="M").to_timestamp()
    m = m.sort_values(["CustomerID", "YearMonth"])

    first_month = m.groupby("CustomerID")["YearMonth"].min().rename("first_month")
    m = m.merge(first_month, on="CustomerID", how="left")

    active = m.groupby(["YearMonth", "CustomerID"]).size().reset_index(name="active")
    active["active"] = 1

    pivot = active.pivot(index="CustomerID", columns="YearMonth", values="active").fillna(0).astype(int)
    months = list(pivot.columns)

    rows = []
    for i, month in enumerate(months):
        current = pivot[month] == 1
        active_customers = int(current.sum())
        is_new = (first_month == month).reindex(pivot.index).fillna(False)
        new_customers = int(is_new[current].sum())

        if i == 0:
            retained = 0
            reactivated = 0
            churned = 0
        else:
            prev_month = months[i - 1]
            prev = pivot[prev_month] == 1
            retained = int((current & prev).sum())
            reactivated = int(current & ~prev & (pivot.iloc[:, :i].sum(axis=1) > 0))
            churned = int(prev & ~current)

        rows.append(
            {
                "period": month,
                "active_customers": active_customers,
                "new_customers": new_customers,
                "retained": retained,
                "reactivated": reactivated,
                "churned": churned,
            }
        )

    return pd.DataFrame(rows)


def build_customer_tables() -> dict[str, pd.DataFrame]:
    snapshot, monthly = read_silver()
    kpis = build_customer_kpis(snapshot, monthly)
    retention = monthly_retention_table(monthly)
    return {
        "customer_kpis": kpis,
        "customer_retention_monthly": retention,
    }


def main() -> dict[str, pd.DataFrame]:
    outputs = build_customer_tables()
    kpis_path = GOLD_DIR / "customer_kpis.parquet"
    retention_path = GOLD_DIR / "customer_retention_monthly.parquet"
    write_parquet(outputs["customer_kpis"], kpis_path)
    write_parquet(outputs["customer_retention_monthly"], retention_path)
    logger.info(
        "customer_kpis rows=%s | customer_retention_monthly rows=%s",
        len(outputs["customer_kpis"]),
        len(outputs["customer_retention_monthly"]),
    )
    return outputs


if __name__ == "__main__":
    main()
