You are Codex. Implement the React + FastAPI app skeleton into this repo.

Backend:
- Expand apps/react/backend/main.py to load trained models from `models/` if present.
- Implement /api/fraud/score (CSV upload → probability + flags + top features via SHAP).
- Implement /api/taxi/forecast (params: date range, zone → predicted hourly counts).
- Implement /api/sentiment/predict (text → sentiment prob).

Frontend:
- Flesh out the three pages with simple forms and result tables.
- Add a small client util for API base URL.
- Create a README in the frontend with `npm run dev` instructions.

Ensure CORS is okay for local dev. Add minimal tests for API (pytest).
