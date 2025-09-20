from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "data/silver/transactions_base.csv"
OUT = BASE / "data/silver/dim_customer.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"No encuentro {SRC}")
    t = pd.read_csv(SRC, parse_dates=["InvoiceDate"])

    # Flags
    t["IsReturn"] = t.get("IsReturn", (t["Quantity"] < 0))

    # Métricas por cliente (solo ventas para órdenes; netos para $)
    is_sale = ~t["IsReturn"]
    g = t.groupby("CustomerID", dropna=False)

    dim = pd.DataFrame({
        "CustomerID": g.size().index
    })
    dim["first_purchase"] = g["InvoiceDate"].min().values
    dim["last_purchase"] = g["InvoiceDate"].max().values
    dim["country_mode"] = g["Country"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA).values

    # Totales netos
    net_sales = g["Sales"].sum()
    cogs_net = g["COGS"].sum()
    gp_net = g["GrossProfit"].sum()
    orders_sale = g["InvoiceNo"].agg(
        lambda x: x[is_sale.loc[x.index]].nunique())

    dim["net_sales_total"] = dim["CustomerID"].map(net_sales).round(2)
    dim["cogs_total"] = dim["CustomerID"].map(cogs_net).round(2)
    dim["gp_total"] = dim["CustomerID"].map(gp_net).round(2)
    dim["orders_total"] = dim["CustomerID"].map(
        orders_sale).fillna(0).astype(int)

    # AOV lifetime y margen
    dim["aov_lifetime"] = (dim["net_sales_total"] / dim["orders_total"]
                           ).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)
    dim["gross_margin_pct"] = np.where(
        dim["net_sales_total"] != 0, dim["gp_total"]/dim["net_sales_total"], np.nan).round(4)

    # Antigüedad (días) y “recency”
    dim["tenure_days"] = (dim["last_purchase"] - dim["first_purchase"]).dt.days
    max_date = t["InvoiceDate"].max()
    dim["recency_days"] = (max_date - dim["last_purchase"]).dt.days

    dim = dim.sort_values("CustomerID")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dim.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"[OK] {OUT} -> {len(dim):,} filas")


if __name__ == "__main__":
    main()
