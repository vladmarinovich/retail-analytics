from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from gold.build_country_tables import build_country_monthly, build_country_snapshot
from gold.build_product_tables import build_product_monthly, build_product_snapshot
from gold.build_customer_monthly import build_customer_monthly
from unittest.mock import patch


def _sample_transactions() -> pd.DataFrame:
    rows = [
        {
            "InvoiceNo": "100",
            "StockCode": "A",
            "Description": "Alpha",
            "Quantity": 5,
            "InvoiceDate": datetime(2021, 1, 5),
            "UnitPrice": 20.0,
            "UnitCost": 12.0,
            "CustomerID": "C1",
            "Country": "UK",
        },
        {
            "InvoiceNo": "101",
            "StockCode": "A",
            "Description": "Alpha",
            "Quantity": -2,
            "InvoiceDate": datetime(2021, 2, 2),
            "UnitPrice": 20.0,
            "UnitCost": 12.0,
            "CustomerID": "C1",
            "Country": "UK",
        },
        {
            "InvoiceNo": "102",
            "StockCode": "A",
            "Description": "Alpha",
            "Quantity": 2,
            "InvoiceDate": datetime(2021, 2, 10),
            "UnitPrice": 20.0,
            "UnitCost": 12.0,
            "CustomerID": "C1",
            "Country": "UK",
        },
        {
            "InvoiceNo": "103",
            "StockCode": "B",
            "Description": "Beta",
            "Quantity": 3,
            "InvoiceDate": datetime(2021, 2, 11),
            "UnitPrice": 30.0,
            "UnitCost": 15.0,
            "CustomerID": "C3",
            "Country": "UK",
        },
        {
            "InvoiceNo": "104",
            "StockCode": "B",
            "Description": "Beta",
            "Quantity": 1,
            "InvoiceDate": datetime(2021, 1, 15),
            "UnitPrice": 30.0,
            "UnitCost": 15.0,
            "CustomerID": "C2",
            "Country": "France",
        },
        {
            "InvoiceNo": "105",
            "StockCode": "C",
            "Description": "Gamma",
            "Quantity": 4,
            "InvoiceDate": datetime(2021, 1, 20),
            "UnitPrice": 25.0,
            "UnitCost": 10.0,
            "CustomerID": None,
            "Country": None,
        },
    ]
    df = pd.DataFrame(rows)
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    df["COGS"] = df["Quantity"] * df["UnitCost"]
    df["GrossProfit"] = df["Sales"] - df["COGS"]
    df["IsReturn"] = df["Quantity"] < 0
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df


class GoldMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tx = _sample_transactions()
        self.tx["Country"] = self.tx["Country"].fillna("Unspecified")
        self.dim = pd.DataFrame(
            {
                "StockCode": ["A", "B", "C"],
                "description_mode": ["Alpha", "Beta", "Gamma"],
            }
        )

    def test_country_snapshot_buyers_distinct(self):
        monthly = build_country_monthly(self.tx)
        snapshot = build_country_snapshot(self.tx, monthly)
        buyers_uk = snapshot.loc[snapshot["Country"] == "UK", "buyers"].iat[0]
        self.assertEqual(buyers_uk, 2)

    def test_country_monthly_return_units(self):
        monthly = build_country_monthly(self.tx)
        feb_uk = monthly[
            (monthly["Country"] == "UK") & (monthly["YearMonth"] == "2021-02")
        ].iloc[0]
        self.assertAlmostEqual(feb_uk["return_units_abs"], 2.0)
        # Sales quantities for Feb UK = 2 + 3 = 5, return units = 2 -> rate 0.4
        self.assertAlmostEqual(feb_uk["return_rate_units"], 0.4)

    def test_product_snapshot_buyers_distinct(self):
        monthly = build_product_monthly(self.tx, self.dim)
        snapshot = build_product_snapshot(self.tx, monthly, self.dim)
        buyers_a = snapshot.loc[snapshot["StockCode"] == "A", "buyers"].iat[0]
        # Only customer C1 purchased product A
        self.assertEqual(buyers_a, 1)

    @patch("gold.build_customer_monthly.load_transactions")
    def test_customer_monthly_excludes_returns_from_orders(self, mock_load):
        mock_load.return_value = self.tx.copy()
        cm = build_customer_monthly()
        feb_c1 = cm[(cm["customer_id"] == "C1") & (cm["YearMonth"] == "2021-02")].iloc[0]
        self.assertEqual(feb_c1["orders"], 1)
        self.assertAlmostEqual(feb_c1["items_sold"], 2)
        self.assertAlmostEqual(feb_c1["returns_value"], 40)


if __name__ == "__main__":
    unittest.main()
