from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SILVER = BASE / "data" / "silver" / "transactions_base.csv"
OUT = BASE / "data" / "gold" / "company_monthly_kpis.csv"


def main():
    df = pd.read_csv(SILVER)
    if "YearMonth" not in df.columns:
        df["YearMonth"] = pd.to_datetime(
            df["InvoiceDate"]).dt.to_period("M").astype(str)
    is_sale = ~(df.get("IsReturn", df["Quantity"] < 0))
    g = df.groupby("YearMonth", dropna=False)
    out = g.agg(
        orders=("InvoiceNo", lambda x: x[is_sale.loc[x.index]].nunique()),
        customers=("CustomerID", lambda c: c[is_sale.loc[c.index]].nunique()),
        items_sold=("Quantity", lambda q: q[is_sale.loc[q.index]].sum()),
        gmv=("Sales", lambda s: s[is_sale.loc[s.index]].sum()),
        returns_value=("Sales", lambda s: np.abs(
            s[~is_sale.loc[s.index]].sum())),
        net_sales=("Sales", "sum"),
        cogs_net=("COGS", "sum"),
        gp_net=("GrossProfit", "sum"),
    ).reset_index()
    out["gross_margin_pct"] = np.where(
        out["net_sales"] != 0, out["gp_net"]/out["net_sales"], np.nan)
    out = out.sort_values("YearMonth")
    out["net_sales_mom"] = (
        out["net_sales"].pct_change().replace(
            [np.inf, -np.inf], np.nan).round(4)
    )
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net"]
    out[money] = out[money].round(2)
    out["gross_margin_pct"] = out["gross_margin_pct"].round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[OK] {OUT} -> {len(out)} filas")


if __name__ == "__main__":
    main()
