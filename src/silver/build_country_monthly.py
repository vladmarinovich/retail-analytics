# src/silver/build_country_monthly.py
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "data/silver/transactions_base.csv"
OUT = BASE / "data/silver/country_monthly.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"No encuentro {SRC}")
    t = pd.read_csv(SRC, parse_dates=["InvoiceDate"])

    if "IsReturn" not in t.columns:
        t["IsReturn"] = t["Quantity"] < 0

    g = t.groupby(["Country", "YearMonth"], dropna=False)

    cmtry = g.agg(
        gmv=("Sales", lambda s: s[t.loc[s.index, "IsReturn"] == False].sum()),
        returns_value=("Sales", lambda s: np.abs(
            s[t.loc[s.index, "IsReturn"] == True]).sum()),
        net_sales=("Sales", "sum"),
        gp_net=("GrossProfit", "sum"),
        orders=("InvoiceNo",
                lambda x: x[t.loc[x.index, "IsReturn"] == False].nunique()),
        customers=(
            "CustomerID", lambda c: c[t.loc[c.index, "IsReturn"] == False].nunique()),
        items_sold=(
            "Quantity", lambda q: q[t.loc[q.index, "IsReturn"] == False].sum()),
    ).reset_index()

    cmtry["aov"] = np.where(cmtry["orders"] > 0,
                            cmtry["net_sales"] / cmtry["orders"], 0.0)
    cmtry["gross_margin_pct"] = np.where(
        cmtry["net_sales"] != 0, cmtry["gp_net"] / cmtry["net_sales"], np.nan)

    # Share mensual (penetración por país dentro del mes)
    total_mes = cmtry.groupby("YearMonth")["net_sales"].transform("sum")
    cmtry["net_sales_share"] = np.where(
        total_mes > 0, cmtry["net_sales"] / total_mes, np.nan)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmtry.to_csv(OUT, index=False)
    print(f"[OK] {OUT} -> {len(cmtry):,} filas")


if __name__ == "__main__":
    main()
