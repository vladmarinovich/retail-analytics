# src/silver/bronze_to_transactions.py
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
BRONZE = BASE / "data/bronze/online_retail_enriched.csv"
OUT = BASE / "data/silver/transactions_base.csv"


def main():
    # Lee Bronze
    df = pd.read_csv(BRONZE, parse_dates=["InvoiceDate"])

    # --- Normalización de claves/textos ---
    # Evita duplicados por casing/espacios en producto
    if "StockCode" in df.columns:
        df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str).str.strip()
    if "CustomerID" in df.columns:
        # Mantener nulos reales; evitar "nan" como texto
        df["CustomerID"] = (
            df["CustomerID"].where(df["CustomerID"].notna())
            .astype("string")
            .str.strip()
        )

    # Flags venta/devolución
    df["IsReturn"] = df["Quantity"] < 0
    df["is_sale"] = ~df["IsReturn"]

    # Recalcular por seguridad (conserva signos)
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    df["COGS"] = df["Quantity"] * df["UnitCost"]
    df["GrossProfit"] = df["Sales"] - df["COGS"]

    # --- Filtros Silver (mínimos y seguros) ---
    # Solo ventas inválidas (precio<=0 o costo<0). Devoluciones se mantienen.
    invalid_sales = (df["is_sale"]) & (
        (df["UnitPrice"] <= 0) | (df["UnitCost"] < 0))
    # Quitar cantidades nulas o 0
    invalid_qty = df["Quantity"].isna() | (df["Quantity"] == 0)
    keep = ~invalid_sales & ~invalid_qty

    trans = df.loc[keep].copy()

    # Derivados de fecha
    trans["YearMonth"] = trans["InvoiceDate"].dt.to_period("M").astype(str)
    trans["Date"] = trans["InvoiceDate"].dt.date

    # Export
    OUT.parent.mkdir(parents=True, exist_ok=True)
    trans.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(
        f"[OK] {OUT} -> {len(trans):,} filas (removidas inválidas: {int(invalid_sales.sum())})")


if __name__ == "__main__":
    main()
