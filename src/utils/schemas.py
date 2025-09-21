"""DataFrame contracts for retail analytics artifacts."""
from __future__ import annotations

try:
    import pandera as pa  # type: ignore
    from pandera import Column, DataFrameSchema, Check  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without pandera
    class _FallbackCheck:
        @staticmethod
        def ge(_min):
            return None

    class Column:  # type: ignore
        def __init__(
            self,
            *_args,
            required: bool = True,
            **_kwargs,
        ) -> None:
            self.required = required

    class DataFrameSchema:  # type: ignore
        def __init__(self, columns, **_kwargs) -> None:
            self.columns = columns

        def validate(self, df, lazy: bool | None = None):
            missing = [
                name
                for name, col in self.columns.items()
                if getattr(col, "required", True) and name not in df.columns
            ]
            if missing:
                raise KeyError(f"Missing columns: {missing}")
            return df

    class _FallbackPandera:
        String = object()
        Float = object()
        Int64 = object()
        DateTime = object()
        Bool = object()

    pa = _FallbackPandera()  # type: ignore
    Check = _FallbackCheck()  # type: ignore

# ---------------------------------------------------------------------------
# SILVER
# ---------------------------------------------------------------------------

transactions_base_schema = DataFrameSchema(
    {
        "InvoiceNo": Column(pa.String, required=True),
        "StockCode": Column(pa.String, required=True),
        "Description": Column(pa.String, required=False, coerce=True, nullable=True),
        "Quantity": Column(pa.Float, required=True),
        "InvoiceDate": Column(pa.DateTime, required=True),
        "UnitPrice": Column(pa.Float, required=True),
        "UnitCost": Column(pa.Float, required=True),
        "CustomerID": Column(pa.String, required=False, nullable=True),
        "Country": Column(pa.String, required=False, nullable=True),
        "Sales": Column(pa.Float, required=True),
        "COGS": Column(pa.Float, required=True),
        "GrossProfit": Column(pa.Float, required=True),
        "IsReturn": Column(pa.Bool, required=True),
        "YearMonth": Column(pa.String, required=True),
    },
    coerce=True,
    strict=False,
)

# ---------------------------------------------------------------------------
# GOLD MONTHLY KPI TABLES
# ---------------------------------------------------------------------------

company_monthly_kpis_schema = DataFrameSchema(
    {
        "period": Column(pa.DateTime, required=True),
        "YearMonth": Column(pa.String, required=True),
        "orders": Column(pa.Int64, Check.ge(0), required=True),
        "customers": Column(pa.Int64, Check.ge(0), required=True),
        "items_sold": Column(pa.Float, required=True),
        "gmv": Column(pa.Float, required=True),
        "returns_value": Column(pa.Float, required=True),
        "net_sales": Column(pa.Float, required=True),
        "cogs_net": Column(pa.Float, required=True),
        "gp_net": Column(pa.Float, required=True),
        "gross_margin_pct": Column(pa.Float, required=True, nullable=True),
        "net_sales_mom": Column(pa.Float, required=True, nullable=True),
        "aov": Column(pa.Float, required=True, nullable=True),
    },
    coerce=True,
    strict=False,
)

country_monthly_kpis_schema = DataFrameSchema(
    {
        "period": Column(pa.DateTime, required=True),
        "YearMonth": Column(pa.String, required=True),
        "Country": Column(pa.String, required=True),
        "orders": Column(pa.Int64, Check.ge(0), required=True),
        "customers": Column(pa.Int64, Check.ge(0), required=True),
        "items_sold": Column(pa.Float, required=True),
        "gmv": Column(pa.Float, required=True),
        "returns_value": Column(pa.Float, required=True),
        "net_sales": Column(pa.Float, required=True),
        "cogs_net": Column(pa.Float, required=True),
        "gp_net": Column(pa.Float, required=True),
        "gross_margin_pct": Column(pa.Float, required=True, nullable=True),
        "net_sales_share": Column(pa.Float, required=True, nullable=True),
        "net_sales_mom": Column(pa.Float, required=True, nullable=True),
        "aov": Column(pa.Float, required=True, nullable=True),
        "return_units_abs": Column(pa.Float, required=True),
        "return_rate_units": Column(pa.Float, required=True, nullable=True),
        "return_rate_value": Column(pa.Float, required=True, nullable=True),
    },
    coerce=True,
    strict=False,
)

product_monthly_kpis_schema = DataFrameSchema(
    {
        "period": Column(pa.DateTime, required=True),
        "YearMonth": Column(pa.String, required=True),
        "StockCode": Column(pa.String, required=True),
        "description_mode": Column(pa.String, required=False, nullable=True),
        "units_sold": Column(pa.Float, required=True),
        "gmv": Column(pa.Float, required=True),
        "returns_value": Column(pa.Float, required=True),
        "net_sales": Column(pa.Float, required=True),
        "cogs_net": Column(pa.Float, required=True),
        "gp_net": Column(pa.Float, required=True),
        "orders": Column(pa.Int64, Check.ge(0), required=True),
        "buyers": Column(pa.Int64, Check.ge(0), required=True),
        "aov": Column(pa.Float, required=True, nullable=True),
        "return_units_abs": Column(pa.Float, required=True),
        "return_rate_units": Column(pa.Float, required=True, nullable=True),
        "return_rate_value": Column(pa.Float, required=True, nullable=True),
    },
    coerce=True,
    strict=False,
)

customer_monthly_kpis_schema = DataFrameSchema(
    {
        "period": Column(pa.DateTime, required=True),
        "YearMonth": Column(pa.String, required=True),
        "customer_id": Column(pa.String, required=True),
        "orders": Column(pa.Int64, Check.ge(0), required=True),
        "items_sold": Column(pa.Float, required=True),
        "gmv": Column(pa.Float, required=True),
        "returns_value": Column(pa.Float, required=True),
        "net_sales": Column(pa.Float, required=True),
        "cogs_net": Column(pa.Float, required=True),
        "gp_net": Column(pa.Float, required=True),
        "aov": Column(pa.Float, required=True, nullable=True),
    },
    coerce=True,
    strict=False,
)


__all__ = [
    "transactions_base_schema",
    "company_monthly_kpis_schema",
    "country_monthly_kpis_schema",
    "product_monthly_kpis_schema",
    "customer_monthly_kpis_schema",
]
