"""
FastAPI backend powering the Agentic Data Science demos.

Endpoints
---------
- POST /api/fraud/score: placeholder CSV scoring hook (to be replaced with trained model).
- GET /api/fraud/demo: returns the cached 50-row fraud-threshold demo payload used by the React explorer.
- POST /api/sentiment/predict: toy sentiment example for the sentiment tab.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io

app = FastAPI(title="AgenticDataScience API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentRequest(BaseModel):
    text: str

BASE_DIR = Path(__file__).resolve().parents[3]
DEMO_JSON = BASE_DIR / "data" / "sample" / "fraud_threshold_demo.json"
DEMO_META = BASE_DIR / "data" / "sample" / "fraud_threshold_demo.meta.json"


def load_demo_payload() -> Dict[str, Any]:
    if not DEMO_JSON.exists():
        return {"transactions": [], "generated_at": None, "metadata": {}}

    with DEMO_JSON.open() as f:
        transactions: List[Dict[str, Any]] = json.load(f)

    generated_at = None
    if DEMO_META.exists():
        try:
            generated_at = json.loads(DEMO_META.read_text()).get("generated_at")
        except json.JSONDecodeError:
            generated_at = None
    if not generated_at:
        generated_at = datetime.utcfromtimestamp(DEMO_JSON.stat().st_mtime).isoformat() + "Z"

    fraud_cases = sum(1 for row in transactions if row.get("is_fraud_actual"))
    sample_size = len(transactions)
    metadata = {
        "model_version": "demo-calibrated-v1",
        "sample_size": sample_size,
        "class_balance": {"fraud": fraud_cases, "legit": sample_size - fraud_cases},
        "avg_prob": round(sum(row.get("fraud_prob", 0.0) for row in transactions) / sample_size, 4)
        if sample_size else 0.0,
    }
    return {"transactions": transactions, "generated_at": generated_at, "metadata": metadata}


FRAUD_THRESHOLD_DEMO = load_demo_payload()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/sentiment/predict")
def sentiment_predict(req: SentimentRequest):
    text = req.text.lower()
    score = 1.0 if "good" in text or "great" in text else 0.0
    return {"label": "positive" if score > 0.5 else "negative", "score": score}

@app.post("/api/fraud/score")
async def fraud_score(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    # TODO: replace with trained model + SHAP
    df["fraud_score"] = 0.01
    df["is_fraud"] = df["fraud_score"] > 0.5
    return {"n_rows": int(df.shape[0]), "n_flagged": int(df["is_fraud"].sum())}


@app.get("/api/fraud/demo")
def fraud_demo():
    if not FRAUD_THRESHOLD_DEMO["transactions"]:
        raise HTTPException(status_code=500, detail="Fraud demo sample is missing. Please regenerate data/sample slice.")
    return FRAUD_THRESHOLD_DEMO
