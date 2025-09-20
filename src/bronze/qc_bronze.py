# src/bronze/qc_bronze.py
from pathlib import Path
import os
import pandas as pd

BASE = Path(__file__).resolve().parents[2]  # carpeta raíz del repo
BRONZE = BASE / "data/bronze/online_retail_enriched.csv"
OUTDIR = BASE / "reports/bronze_qc"
OUTCSV = OUTDIR / "bronze_profile.csv"


def main():
    if not BRONZE.exists():
        raise FileNotFoundError(f"No encuentro: {BRONZE}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BRONZE, parse_dates=["InvoiceDate"])

    qc = {
        "rows_total": [len(df)],
        "nulls_InvoiceDate": [df["InvoiceDate"].isna().sum()],
        "qty_negatives": [(df["Quantity"] < 0).sum()],
        "prices_le_0": [(df["UnitPrice"] <= 0).sum()],
        "costs_lt_0": [(df["UnitCost"] < 0).sum()],
        "nulls_CustomerID": [df["CustomerID"].isna().sum()],
        "min_date": [df["InvoiceDate"].min()],
        "max_date": [df["InvoiceDate"].max()],
        "grossprofit_nans": [df["GrossProfit"].isna().sum()],
    }

    pd.DataFrame(qc).to_csv(OUTCSV, index=False)
    print(f"QC exportado → {OUTCSV}")

    # Muestra rápida en consola
    print("\nResumen rápido:")
    print(df[["Sales", "COGS", "GrossProfit"]].describe().round(2))
    print("Fechas:", df["InvoiceDate"].min().date(),
          "→", df["InvoiceDate"].max().date())


if __name__ == "__main__":
    main()
