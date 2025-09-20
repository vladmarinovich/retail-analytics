from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
CM = BASE / "data" / "gold" / "company_monthly_kpis.csv"
OUT = BASE / "data" / "gold" / "executive_summary.csv"


def main():
    if not CM.exists():
        raise FileNotFoundError(
            f"No encuentro {CM}. Corre build_company_monthly_kpis primero.")
    m = pd.read_csv(CM)
    m["YearMonth"] = m["YearMonth"].astype(str)
    summary = pd.DataFrame({
        "first_period": [m["YearMonth"].min()],
        "last_period":  [m["YearMonth"].max()],
        "months":       [m["YearMonth"].nunique()],
        "orders":       [m["orders"].sum()],
        # clientes-actividad (no únicos globales)
        "customers":    [m["customers"].sum()],
        "items_sold":   [m["items_sold"].sum()],
        "gmv":          [m["gmv"].sum()],
        "returns_value": [m["returns_value"].sum()],
        "net_sales":    [m["net_sales"].sum()],
        "cogs_net":     [m["cogs_net"].sum()],
        "gp_net":       [m["gp_net"].sum()],
    })
    summary["gross_margin_pct"] = np.where(summary["net_sales"] != 0,
                                           summary["gp_net"]/summary["net_sales"], np.nan)
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    summary[money] = summary[money].round(2)
    summary["gross_margin_pct"] = summary["gross_margin_pct"].round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(f"[OK] {OUT} -> 1 fila")


if __name__ == "__main__":
    main()
