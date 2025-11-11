# AgenticDataScience

A starter workspace for **agentic data science** powered by **OpenAI Codex** + **VS Code**.

## Datasets (3 diverse areas)
1) **Credit Card Fraud (tabular, imbalanced)** — Kaggle `mlg-ulb/creditcardfraud` (requires Kaggle account).
2) **NYC Taxi Trips (time series / geospatial)** — NYC TLC Trip Record Data.
3) **IMDb Movie Reviews (NLP sentiment)** — Stanford Large Movie Review Dataset.

## Repo layout
```
AgenticDataScience/
  apps/react/            # React + FastAPI demo app
  codex/                 # Reusable prompts + cloud config
  data/                  # raw/ (ignored by git), sample/ (small checked-in slices)
  docs/                  # MRM-grade documentation + evidence
  notebooks/             # EDA + modeling
  scripts/               # Data downloads and utilities
```

## Quickstart
```bash
# from your Documents folder on macOS/Linux
cp -R AgenticDataScience ~/Documents/
cd ~/Documents/AgenticDataScience

# (optional) Python venv
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r apps/react/backend/requirements.txt

# Run the backend API (FastAPI)
uvicorn apps.react.backend.main:app --reload

# Run the React frontend (in another terminal)
cd apps/react/frontend
npm install
npm run dev
```

## Fraud Threshold Explorer UI
- Demo data lives in `data/sample/fraud_threshold_demo.json` (50 curated transactions). Regenerate anytime via `python3 scripts/make_sample_slices.py`.
- The FastAPI backend exposes `GET /api/fraud/demo`, serving the cached sample along with metadata.
- The React tab "Fraud" now loads an interactive threshold explorer with precision/recall cards, a chart, and an outcome-colored table.
- Update the frontend locally with:
  ```bash
  cd apps/react/frontend
  npm install   # first run
  npm run dev   # serve UI
  npm test      # runs vitest suite for fraud metrics helpers
  ```
