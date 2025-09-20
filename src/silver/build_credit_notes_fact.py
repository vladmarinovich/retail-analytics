# src/silver/build_credit_notes_fact.py
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "data/silver/transactions_base.csv"
OUT = BASE / "data/silver/credit_notes_fact.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"No encuentro {SRC}")
    df = pd.read_csv(SRC, parse_dates=["InvoiceDate"])

    returns = df[df["IsReturn"]].copy()  # solo devoluciones

    cn = (returns.groupby("InvoiceNo").agg(
        InvoiceDate=("InvoiceDate", "min"),
        CustomerID=("CustomerID", "first"),
        Country=("Country", "first"),
        items_count=("StockCode", "nunique"),
        qty_total=("Quantity", "sum"),           # suele ser negativo
        sales_total=("Sales", "sum"),            # negativo
        cogs_total=("COGS", "sum"),              # negativo
        gross_profit_total=("GrossProfit", "sum")  # negativo
    ).reset_index())

    cn["YearMonth"] = cn["InvoiceDate"].dt.to_period("M").astype(str)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cn.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"[OK] {OUT} -> {len(cn):,} filas")


if __name__ == "__main__":
    main()
