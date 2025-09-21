"""Generate executive summary snapshot from company monthly KPIs."""
from __future__ import annotations

import pandas as pd

from features.metrics import safe_div
from utils.io import get_paths, logger, write_parquet

PATHS = get_paths()
COMPANY_MONTHLY = PATHS.gold / "company_monthly_kpis.parquet"
OUTPUT_PATH = PATHS.gold / "executive_summary.parquet"


def build_executive_summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    company = df.copy() if df is not None else pd.read_parquet(COMPANY_MONTHLY)
    if company.empty:
        raise ValueError("company_monthly_kpis está vacío; ejecuta build_company_monthly_kpis primero.")
    company["YearMonth"] = company["YearMonth"].astype(str)
    summary = pd.DataFrame(
        {
            "first_period": [company["YearMonth"].min()],
            "last_period": [company["YearMonth"].max()],
            "months": [company["YearMonth"].nunique()],
            "orders": [company["orders"].sum()],
            "customers": [company["customers"].sum()],
            "items_sold": [company["items_sold"].sum()],
            "gmv": [company["gmv"].sum()],
            "returns_value": [company["returns_value"].sum()],
            "net_sales": [company["net_sales"].sum()],
            "cogs_net": [company["cogs_net"].sum()],
            "gp_net": [company["gp_net"].sum()],
        }
    )
    summary["gross_margin_pct"] = safe_div(summary["gp_net"], summary["net_sales"])
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    summary[money] = summary[money].round(2)
    summary["gross_margin_pct"] = summary["gross_margin_pct"].round(4)
    return summary


def main() -> pd.DataFrame:
    summary = build_executive_summary()
    write_parquet(summary, OUTPUT_PATH)
    logger.info("executive_summary rows=%s", len(summary))
    return summary


if __name__ == "__main__":
    main()
