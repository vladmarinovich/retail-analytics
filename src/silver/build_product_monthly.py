# src/silver/build_product_monthly.py
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "data/silver/transactions_base.csv"
OUT = BASE / "data/silver/product_monthly.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"No encuentro {SRC}")
    t = pd.read_csv(SRC, parse_dates=["InvoiceDate"])

    if "IsReturn" not in t.columns:
        t["IsReturn"] = t["Quantity"] < 0

    # Solo para ventas / devoluciones según flag
    is_sale = ~t["IsReturn"]
    is_ret = t["IsReturn"]

    # Métricas por producto y mes
    g = t.groupby(["StockCode", "YearMonth"], dropna=False)

    pm = g.agg(
        # ventas
        units_sold=("Quantity", lambda q: q[is_sale.loc[q.index]].sum()),
        gmv=("Sales", lambda s: s[is_sale.loc[s.index]].sum()),
        # devoluciones (magnitud)
        return_units_abs=("Quantity", lambda q: np.abs(
            q[is_ret.loc[q.index]].sum())),
        returns_value=("Sales", lambda s: np.abs(
            s[is_ret.loc[s.index]].sum())),
        # netos
        net_sales=("Sales", "sum"),
        cogs_net=("COGS",  "sum"),
        gp_net=("GrossProfit", "sum"),
        # demanda
        orders=("InvoiceNo", lambda x: x[is_sale.loc[x.index]].nunique()),
        buyers=("CustomerID", lambda c: c[is_sale.loc[c.index]].nunique()),
        # descripción dominante (para rotular)
        description_mode=("Description", lambda d: d.mode(
        ).iat[0] if not d.mode().empty else pd.NA),
    ).reset_index()

    # Ratios útiles
    pm["return_rate_units"] = np.where((pm["units_sold"] > 0),
                                       pm["return_units_abs"] / pm["units_sold"], np.nan)
    pm["gross_margin_pct"] = np.where(
        pm["net_sales"] != 0, pm["gp_net"]/pm["net_sales"], np.nan)
    pm["aov"] = np.where(pm["orders"] > 0, pm["net_sales"]/pm["orders"], 0.0)

    # Orden/round
    money = ["gmv", "returns_value", "net_sales", "cogs_net", "gp_net", "aov"]
    pm[money] = pm[money].round(2)
    pm["gross_margin_pct"] = pm["gross_margin_pct"].round(4)
    pm["return_rate_units"] = pm["return_rate_units"].round(4)
    pm = pm.sort_values(["StockCode", "YearMonth"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pm.to_csv(OUT, index=False)
    print(f"[OK] {OUT} -> {len(pm):,} filas")


if __name__ == "__main__":
    main()
