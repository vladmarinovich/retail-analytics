from pathlib import Path
import pandas as pd

GOLD = Path("data/gold")
for csv in GOLD.glob("*.csv"):
    df = pd.read_csv(csv)
    parquet = csv.with_suffix(".parquet")
    df.to_parquet(parquet, index=False)
    print("OK ->", parquet)
