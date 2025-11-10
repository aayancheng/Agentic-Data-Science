"""
Create small sample slices under data/sample/ for quick commits / demos.
"""
import pandas as pd, pathlib

BASE = pathlib.Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
SAMPLE = BASE / "data" / "sample"
SAMPLE.mkdir(exist_ok=True, parents=True)

# Fraud (if available)
fraud_csvs = list((RAW / "credit_card_fraud").glob("*.csv"))
if fraud_csvs:
    df = pd.read_csv(fraud_csvs[0])
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)
    df_sample.to_csv(SAMPLE / "creditcard_sample.csv", index=False)

# NYC taxi (if available)
pqs = list((RAW / "nyc_taxi").glob("*.parquet"))
if pqs:
    df = pd.read_parquet(pqs[0])
    df = df.sample(n=min(20000, len(df)), random_state=42)
    df.to_parquet(SAMPLE / "yellow_taxi_sample.parquet")

print("Sample slices created (where source data was present).")
