# Fraud Threshold Explorer — Codex Task List

Use this sequence when implementing the React experience described in `docs/fraud_threshold_explorer_plan.md`.

## 1. Prep & Data
1. Pull 25 fraud + 25 legit rows from the Kaggle credit-card dataset sample (or reuse existing slice) and enrich with friendly metadata (`merchant`, `country`, `card_network`, `amount`).
2. Attach modeled probabilities (`fraud_prob`) using the latest calibrated model artifact or mock values that roughly mirror real recall/precision tradeoffs (mean ≈ 0.35 for legit, 0.75 for fraud).
3. Save to `data/sample/fraud_threshold_demo.parquet` (and optionally a JSON copy for quick diffing). Commit-friendly deterministic ordering by `id`.
4. Document schema + generation script in `scripts/make_sample_slices.py` so samples can be reproduced.

## 2. Backend API
1. Add a FastAPI route `GET /api/fraud/demo` in `apps/react/backend/main.py`.
2. Load the parquet/JSON once at startup (module scope) to avoid repeated disk I/O; respond with JSON payload `{ "transactions": [...], "generated_at": ... }`.
3. Include model metadata (`model_version`, `sample_size`, `class_balance`) in the response to display in the UI.
4. Update backend README / docstring to explain the new endpoint.

## 3. Frontend Architecture
1. Create a dedicated page component `FraudThresholdExplorer.jsx` and route it from `App.jsx` (can replace current `Fraud` page).
2. Build subcomponents outlined in the plan:
   - `ThresholdControls`
   - `MetricsSummary`
   - `PrecisionRecallChart`
   - `TransactionTable`
   - Utility module `lib/fraudMetrics.js`
3. Fetch demo data via `useEffect` and store in state (`transactions`, `loading`, `error`).
4. Implement derived metrics memoized via `useMemo` (counts, metrics, filtered lists).

## 4. UI Behavior
1. Threshold slider: 0.05–0.95 range, step 0.01, label the current value and show preset chips (0.2, 0.5, 0.8).
2. Metrics summary: show flagged count, recall, precision, false-positive rate, avg ticket. Include confusion matrix styled cards.
3. Chart: render two lines (precision, recall) across stored thresholds; highlight the current threshold.
4. Transaction table: 50 rows, color-coded by outcome (TP/FP/FN/TN), filter pill group (All, Flagged, False Positives, Missed Fraud).
5. Empty/error states: friendly message + retry button if fetch fails.

## 5. Testing & Polish
1. Add unit tests for `lib/fraudMetrics.js` (e.g., via Vitest) covering edge cases (no frauds, all frauds, division by zero).
2. Run frontend lint/test scripts; ensure storybook-like snapshots aren’t required.
3. Include brief instructions in `README.md` (how to run backend + frontend to view the explorer).
4. Provide screenshots or GIF (optional) in `docs/validation/` once UI is ready.

