# src/bronze/enrich_bronze.py
# Agrega derivados contables a Bronze: Sales, COGS, GrossProfit, IsReturn, GrossMarginPct.

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DEF_INP = Path("data/bronze/online_retail_enriched.csv")
# sobrescribe por defecto
DEF_OUT = Path("data/bronze/online_retail_enriched.csv")

REQ_COLS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country", "MarginPct", "UnitCost"
]


def main(inp: Path, outp: Path):
    if not inp.exists():
        raise FileNotFoundError(f"No encuentro el archivo de entrada: {inp}")

    df = pd.read_csv(inp)

    # Tipos seguros
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").round(2)
    df["UnitCost"] = pd.to_numeric(df["UnitCost"], errors="coerce").round(2)
    df["MarginPct"] = pd.to_numeric(df["MarginPct"], errors="coerce").round(2)

    # Derivados contables (conservan signo de Quantity; retorna negativos)
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    df["COGS"] = df["Quantity"] * df["UnitCost"]
    df["GrossProfit"] = df["Sales"] - df["COGS"]
    df["IsReturn"] = df["Quantity"] < 0

    # % margen sobre la línea (puede ser NaN si Sales==0)
    df["GrossMarginPct"] = np.where(
        df["Sales"] != 0, df["GrossProfit"] / df["Sales"], np.nan)

    # Orden de columnas (original + nuevos al final si existen)
    cols_orig = [c for c in REQ_COLS if c in df.columns]
    cols_new = ["Sales", "COGS", "GrossProfit", "IsReturn", "GrossMarginPct"]
    cols_out = cols_orig + [c for c in cols_new if c in df.columns]

    # Exporta (fecha como YYYY-MM-DD)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df[cols_out].to_csv(outp, index=False,
                        date_format="%Y-%m-%d", encoding="utf-8")
    print(f"[OK] Bronze enriquecido → {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Enriquecer Bronze con métricas contables.")
    ap.add_argument("--in",  dest="inp",  default=str(DEF_INP),
                    help="CSV de entrada (bronze).")
    ap.add_argument("--out", dest="outp", default=str(DEF_OUT),
                    help="CSV de salida.")
    args = ap.parse_args()
    main(Path(args.inp), Path(args.outp))
