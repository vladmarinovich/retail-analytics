"""Build GOLD customer monthly KPIs using silver customers_monthly.csv."""
from __future__ import annotations

from utils.data import load_transactions  # backward compatibility for tests
from utils.io import get_paths, logger, write_parquet
from utils.schemas import customer_monthly_kpis_schema

from gold.build_customer_tables import (
    MONTHLY_PATH,
    SNAP_PATH,
    _monthly_from_transactions,
    build_customer_monthly_kpis,
    read_silver,
)

PATHS = get_paths()
OUTPUT_PATH = PATHS.gold / "customer_monthly_kpis.parquet"


def build_customer_monthly() -> object:
    if SNAP_PATH.exists() and MONTHLY_PATH.exists():
        _snapshot, monthly_silver = read_silver()
    else:
        monthly_silver = _monthly_from_transactions(load_transactions())
    monthly = build_customer_monthly_kpis(monthly_silver)
    monthly = customer_monthly_kpis_schema.validate(monthly, lazy=True)
    return monthly


def main() -> object:
    monthly = build_customer_monthly()
    write_parquet(monthly, OUTPUT_PATH)
    logger.info("customer_monthly_kpis rows=%s", len(monthly))
    return monthly


if __name__ == "__main__":
    main()
