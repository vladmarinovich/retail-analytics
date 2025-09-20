# src/gold/build_monthly_invoices.py
"""
Genera data/gold/revenue_monthly.csv desde data/silver/transactions_base.csv

Métricas por mes (YYYY-MM):
- sales_gross          : ventas brutas ($, solo Quantity>0)
- returns_gross        : devoluciones brutas ($, absoluto de Quantity<0)
- net_sales            : sales_gross - returns_gross
- cogs_sales           : COGS de ventas (Quantity>0)
- cogs_returns         : COGS de devoluciones (absoluto de Quantity<0)
- net_cogs             : cogs_sales - cogs_returns
- gross_profit         : net_sales - net_cogs
- margin_pct           : gross_profit / net_sales (si net_sales>0)
- orders               : # facturas de venta únicas (InvoiceNo con Quantity>0)
- credit_notes         : # notas de crédito únicas (InvoiceNo con Quantity<0)
- return_rate_pct      : returns_gross / sales_gross (si sales_gross>0)
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def build_revenue_monthly(inp: Path, outp: Path):
    df = pd.read_csv(inp)

    # Asegurar tipos
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # Por si no existen estas columnas exactas, calcula on-the-fly
    if "UnitCost" not in df.columns:
        raise ValueError(
            "transactions_base debe contener la columna 'UnitCost'.")
    if "UnitPrice" not in df.columns or "Quantity" not in df.columns:
        raise ValueError("Faltan 'UnitPrice' o 'Quantity'.")

    df["month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Ventas (Quantity > 0) y devoluciones (Quantity < 0)
    sales = df[df["Quantity"] > 0].copy()
    returns = df[df["Quantity"] < 0].copy()

    # Valores monetarios
    sales["sales_gross_line"] = sales["UnitPrice"] * sales["Quantity"]
    sales["cogs_line"] = sales["UnitCost"] * sales["Quantity"]
    returns["returns_gross_ln"] = - \
        (returns["UnitPrice"] * returns["Quantity"])  # abs$
    returns["cogs_ret_ln"] = - \
        (returns["UnitCost"] * returns["Quantity"])  # abs$

    # Agregaciones por mes
    agg_sales = sales.groupby("month").agg(
        sales_gross=("sales_gross_line", "sum"),
        cogs_sales=("cogs_line", "sum"),
        orders=("InvoiceNo", "nunique"),
    )

    agg_returns = returns.groupby("month").agg(
        returns_gross=("returns_gross_ln", "sum"),
        cogs_returns=("cogs_ret_ln", "sum"),
        credit_notes=("InvoiceNo", "nunique"),
    )

    # Unir y calcular KPIs
    monthly = (
        agg_sales.join(agg_returns, how="outer")
        .fillna(0.0)
        .reset_index()
        .rename(columns={"month": "period"})
        .sort_values("period")
    )

    monthly["net_sales"] = monthly["sales_gross"] - monthly["returns_gross"]
    monthly["net_cogs"] = monthly["cogs_sales"] - monthly["cogs_returns"]
    monthly["gross_profit"] = monthly["net_sales"] - monthly["net_cogs"]

    # % margen y % devoluciones
    monthly["margin_pct"] = np.where(
        monthly["net_sales"] > 0, monthly["gross_profit"] /
        monthly["net_sales"], np.nan
    )
    monthly["return_rate_pct"] = np.where(
        monthly["sales_gross"] > 0, monthly["returns_gross"] /
        monthly["sales_gross"], 0.0
    )

    # Redondeos amigables
    money_cols = [
        "sales_gross", "returns_gross", "net_sales",
        "cogs_sales", "cogs_returns", "net_cogs", "gross_profit"
    ]
    monthly[money_cols] = monthly[money_cols].round(2)
    monthly[["margin_pct", "return_rate_pct"]] = monthly[[
        "margin_pct", "return_rate_pct"]].round(4)

    # Guardar
    outp.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(outp, index=False)
    print(f"OK → {outp} ({len(monthly)} filas)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inp",
        default="data/silver/transactions_base.csv",
        help="Ruta del CSV base en silver",
    )
    parser.add_argument(
        "--outp",
        default="data/gold/revenue_monthly.csv",
        help="Ruta de salida para la tabla gold",
    )
    args = parser.parse_args()
    build_revenue_monthly(Path(args.inp), Path(args.outp))


if __name__ == "__main__":
    main()
