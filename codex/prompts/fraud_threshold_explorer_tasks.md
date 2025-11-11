# Fraud Threshold Explorer — Codex Task List

Use this sequence when implementing the React experience described in `docs/fraud_threshold_explorer_plan.md`.

## 1. Prep & Data
1. [x] Generate 50-row synthetic (25 fraud / 25 legit) sample with friendly metadata via `scripts/make_sample_slices.py` (`data/sample/fraud_threshold_demo.json`).
2. [x] Attach modeled probabilities mimicking realistic calibration (fraud mean ≈0.78, legit ≈0.22) and clamp to [0.01, 0.99].
3. [x] Persist deterministic ordering plus JSON artifact under `data/sample/` (parquet export auto-skipped if pyarrow unavailable).
4. [x] Documented the generation logic inside `scripts/make_sample_slices.py` so the slice can be reproduced anytime.

## 2. Backend API
1. [x] Added FastAPI route `GET /api/fraud/demo` in `apps/react/backend/main.py`.
2. [x] Demo payload loads once at startup from `data/sample/fraud_threshold_demo.json` with cached dict response.
3. [x] Response now includes `metadata` (model version, sample size, class balance, avg prob) plus `generated_at`.
4. [x] Documented the endpoint inline in `apps/react/backend/main.py` module docstring.

## 3. Frontend Architecture
1. [x] Created `FraudThresholdExplorer.jsx` and wired it into the Fraud tab (replacing the simple uploader page).
2. Built subcomponents outlined in the plan:
   - [x] `ThresholdControls`
   - [x] `MetricsSummary`
   - [x] `PrecisionRecallChart`
   - [x] `TransactionTable`
   - [x] Utility module `lib/fraudMetrics.js`
3. [x] Fetches demo data via `useEffect` with loading/error states.
4. [x] Derived metrics (counts, precision/recall, filters) memoized with `useMemo`.

## 4. UI Behavior
1. [x] Threshold slider spans 0.05–0.95 with 0.01 steps plus preset chips (0.2 / 0.5 / 0.8).
2. [x] Metrics summary cards show flagged counts + recall/precision/FPR + avg ticket alongside confusion matrix.
3. [x] Chart renders precision/recall trends vs threshold and highlights the active point.
4. [x] Transaction table lists 50 rows with outcome colors + filter pills for All/Flagged/FP/Missed.
5. [x] Added inline loading + retry UX for fetch failures.

## 5. Testing & Polish
1. [x] Added Vitest unit tests for `lib/fraudMetrics.js` covering scoring, metrics, series building, and filters.
2. [ ] Frontend tests pending — `npm install`/`vitest` run blocked until dependencies are installed in this environment.
3. [x] README now documents the fraud explorer workflow and commands.
4. [ ] Screenshots/GIF optional task not yet done.
