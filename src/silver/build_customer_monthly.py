# src/silver/build_customer_monthly.py
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "data/silver/transactions_base.csv"
OUT = BASE / "data/silver/customer_monthly.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"No encuentro {SRC}")
    t = pd.read_csv(SRC, parse_dates=["InvoiceDate"])

    # Flags por si no vienen
    if "IsReturn" not in t.columns:
        t["IsReturn"] = t["Quantity"] < 0

    g = t.groupby(["CustomerID", "YearMonth"], dropna=False)

    cm = g.agg(
        gmv=("Sales", lambda s: s[t.loc[s.index, "IsReturn"] == False].sum()),
        returns_value=("Sales", lambda s: np.abs(
            s[t.loc[s.index, "IsReturn"] == True]).sum()),
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
        orders=("InvoiceNo",
                lambda x: x[t.loc[x.index, "IsReturn"] == False].nunique()),
        items_sold=(
            "Quantity", lambda q: q[t.loc[q.index, "IsReturn"] == False].sum()),
        last_purchase=("InvoiceDate", "max"),
        country_mode=("Country", lambda c: c.mode(
        ).iat[0] if not c.mode().empty else pd.NA),
    ).reset_index()

    cm["aov"] = np.where(cm["orders"] > 0, cm["net_sales"] / cm["orders"], 0.0)
    cm["gross_margin_pct"] = np.where(
        cm["net_sales"] != 0, cm["gp_net"] / cm["net_sales"], np.nan)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cm.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"[OK] {OUT} -> {len(cm):,} filas")


if __name__ == "__main__":
    main()
