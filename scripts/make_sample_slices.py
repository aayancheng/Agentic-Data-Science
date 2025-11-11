"""
Create small sample slices under data/sample/ for quick commits / demos.
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

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
    try:
        df = pd.read_parquet(pqs[0])
        df = df.sample(n=min(20000, len(df)), random_state=42)
        df.to_parquet(SAMPLE / "yellow_taxi_sample.parquet")
    except Exception as exc:  # pragma: no cover - informational
        print(f"Skipping taxi sample generation (missing parquet engine): {exc}")

# Fraud threshold demo (deterministic synthetic sample for UI)
def build_fraud_threshold_demo(n_rows: int = 50) -> None:
    rng = np.random.RandomState(7)
    merchants = [
        "Urban Threads", "Skyline Electronics", "Northwind Market",
        "Summit Outdoors", "Blue River Travel", "Metro Grocery",
        "Aurora Jewelers", "Tempo Fitness", "Orbit RideShare", "Verde Organics"
    ]
    countries = ["US", "CA", "UK", "DE", "FR", "SG", "AU", "BR"]
    card_networks = ["Visa", "Mastercard", "Amex", "Discover", "UnionPay"]
    channels = ["In-Store", "E-Commerce", "In-App", "Call Center"]

    fraud_flags = np.array([1] * (n_rows // 2) + [0] * (n_rows - n_rows // 2))
    rng.shuffle(fraud_flags)

    records: List[dict] = []
    for idx in range(n_rows):
        txn_id = f"TX-{idx+1:03d}"
        is_fraud = int(fraud_flags[idx])
        amount = float(np.round(rng.gamma(shape=3.5, scale=40.0) + 5, 2))
        country = rng.choice(countries, p=[0.45, 0.05, 0.1, 0.08, 0.07, 0.1, 0.08, 0.07])
        merchant = rng.choice(merchants)
        network = rng.choice(card_networks, p=[0.48, 0.3, 0.12, 0.08, 0.02])
        channel = rng.choice(channels, p=[0.35, 0.4, 0.2, 0.05])

        if is_fraud:
            fraud_prob = float(np.clip(rng.normal(loc=0.78, scale=0.12), 0.35, 0.99))
        else:
            fraud_prob = float(np.clip(rng.normal(loc=0.22, scale=0.10), 0.01, 0.65))

        records.append({
            "txn_id": txn_id,
            "amount": amount,
            "merchant": merchant,
            "card_network": network,
            "channel": channel,
            "country": country,
            "is_fraud_actual": bool(is_fraud),
            "fraud_prob": round(fraud_prob, 4),
        })

    df = pd.DataFrame(records).sort_values("txn_id").reset_index(drop=True)
    parquet_path = SAMPLE / "fraud_threshold_demo.parquet"
    json_path = SAMPLE / "fraud_threshold_demo.json"
    parquet_written = False
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:  # pragma: no cover - informational
        print(f"Skipping parquet export for fraud demo (missing engine): {exc}")
    df.to_json(json_path, orient="records", indent=2)
    metadata = {
        "sample_size": len(df),
        "fraud_cases": int(df["is_fraud_actual"].sum()),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": "synthetic_sample_v1",
    }
    (SAMPLE / "fraud_threshold_demo.meta.json").write_text(json.dumps(metadata, indent=2))
    target_desc = f"{json_path.name}"
    if parquet_written:
        target_desc = f"{parquet_path.name} + {json_path.name}"
    print(f"Fraud threshold demo sample written to {target_desc}")

build_fraud_threshold_demo()

print("Sample slices created (where source data was present).")
