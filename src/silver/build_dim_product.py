from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
SRC = BASE / "data/silver/transactions_base.csv"
OUT = BASE / "data/silver/dim_product.csv"


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"No encuentro {SRC}")
    t = pd.read_csv(SRC, parse_dates=["InvoiceDate"])

    # Precio unitario efectivo por fila (por si hay cambios con el tiempo)
    t["UnitPrice"] = t["UnitPrice"].astype(float)

    g = t.groupby("StockCode", dropna=False)
    dim = g.agg(
        description_mode=("Description", lambda s: s.mode(
        ).iat[0] if not s.mode().empty else pd.NA),
        first_sold=("InvoiceDate", "min"),
        last_sold=("InvoiceDate", "max"),
        median_price=("UnitPrice", "median"),
        p95_price=("UnitPrice", lambda s: np.percentile(s, 95)),
        buyer_count_total=("CustomerID", lambda s: s.dropna().nunique()),
        orders_total=("InvoiceNo", "nunique")
    ).reset_index()

    dim = dim.sort_values("StockCode")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dim.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"[OK] {OUT} -> {len(dim):,} filas")


if __name__ == "__main__":
    main()
