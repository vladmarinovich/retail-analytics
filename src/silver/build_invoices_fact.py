# src/silver/build_invoices_fact.py
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "data/silver/transactions_base.csv"
OUT = BASE / "data/silver/invoices_fact.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"No encuentro {SRC}")
    df = pd.read_csv(SRC, parse_dates=["InvoiceDate"])

    sales = df[~df["IsReturn"]].copy()  # solo ventas

    inv = (sales.groupby("InvoiceNo").agg(
        InvoiceDate=("InvoiceDate", "min"),
        CustomerID=("CustomerID", "first"),
        Country=("Country", "first"),
        items_count=("StockCode", "nunique"),
        qty_total=("Quantity", "sum"),
        sales_total=("Sales", "sum"),
        cogs_total=("COGS", "sum"),
        gross_profit_total=("GrossProfit", "sum"),
    ).reset_index())

    inv["YearMonth"] = inv["InvoiceDate"].dt.to_period("M").astype(str)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    inv.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"[OK] {OUT} -> {len(inv):,} filas")


if __name__ == "__main__":
    main()
