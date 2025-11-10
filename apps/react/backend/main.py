from fastapi import FastAPI, UploadFile, File
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
