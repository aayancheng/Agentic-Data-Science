# Fraud Threshold Explorer — React App Plan

## 1. Purpose & Narrative
- **Goal:** Help risk stakeholders understand how the fraud model’s probability outputs translate into business tradeoffs by letting them interactively adjust the decision threshold.
- **Story:** A dashboard shows 50 representative transactions (blend of fraud/legit) plus modeled fraud probabilities. When users drag threshold or pick presets (“High Recall”, “Balanced”, “High Precision”), the UI instantly highlights which transactions get flagged and updates recall/precision summaries.

## 2. Sample Data & Model Assumptions
| Item | Plan |
| --- | --- |
| Dataset | `data/sample/fraud_threshold_demo.parquet` (50 rows, schema: `id`, `amount`, `merchant`, `card_network`, `country`, `is_fraud_actual`, `fraud_prob`). |
| Source | Pull 25 fraudulent + 25 legitimate rows from Kaggle credit-card dataset; attach metadata columns for storytelling. |
| Model outputs | Precomputed `fraud_prob` between 0-1 (calibrated). No live scoring needed for demo. |
| Backend API | `GET /api/fraud/demo` returns the 50 rows + metadata (cached JSON). |
| Frontend state | Fetch once at mount; all threshold math done client-side. |

## 3. UX Flow
1. **Load state:** Spinner until `/api/fraud/demo` resolves; default threshold 0.5 (Balanced preset).
2. **Hero insight cards:** Show aggregated stats at current threshold (flagged %, recall, precision, FPR, avg ticket).
3. **Threshold controls:**
   - Slider (0.01 increments) with live label.
   - Three preset buttons (0.2 / 0.5 / 0.8) to snap quickly.
   - Numeric input for accessibility.
4. **Performance visualization:**
   - Dual metric bars (Precision vs Recall) with color change as threshold shifts.
   - Mini confusion matrix (TP/FP/FN/TN counts over the 50 samples).
   - Optional line chart plotting Precision/Recall across stored thresholds (precomputed arrays) to teach PR curve concepts.
5. **Transaction table:**
   - Columns: `id`, `amount`, `merchant`, `country`, `fraud_prob`, `model_flag`, `actual`.
   - Row highlighting: green (TN/TP) vs red (FP/FN) depending on actual vs model flag.
   - Ability to filter to “Flagged only” or “False Positives only”.
   - Sticky header + inline legend explaining color meanings.

## 4. Component Architecture
| Component | Responsibility |
| --- | --- |
| `pages/FraudThresholdExplorer.jsx` | Fetch demo payload, own global state (`threshold`, `preset`, `filters`). |
| `components/fraud/ThresholdControls.jsx` | Slider, presets, numeric input, displays selected threshold. |
| `components/fraud/MetricsSummary.jsx` | Accepts derived metrics; renders stat cards + confusion matrix. |
| `components/fraud/PrecisionRecallChart.jsx` | Simple SVG/Canvas line chart showing precision & recall vs threshold. |
| `components/fraud/TransactionTable.jsx` | Paginated table of 50 rows with filter pills + legend. |
| `lib/fraudMetrics.js` | Pure helpers to compute counts, precision, recall, FPR from data + threshold. |

## 5. Calculations
- **Flag logic:** `model_flag = fraud_prob >= threshold`.
- **Counts:** `TP/FN/FP/TN` computed by comparing `model_flag` vs `is_fraud_actual`.
- **Metrics:** `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`, `fpr = FP / (FP + TN)`. Guard against division by zero by returning `0.0`.
- **Summary cards:** 
  1. Flagged transactions (count + % of total).
  2. Recall (% of true fraud caught).
  3. Precision (% of flags that are fraud).
  4. False Positive Rate.
  5. Average ticket size of flagged transactions (sums `amount`).

## 6. Visual & UX Notes
- Keep layout within 1200px width, flex layout stacking to column on mobile.
- Use a consistent color palette (e.g., Tailwind teal for true positives, orange for false positives, gray for neutral).
- Add copy blocks:
  - Short description near controls (“Drag the threshold to trade recall vs precision”).
  - Tooltip on metrics cards referencing formulas.
- Provide accessible labels for slider/input; ensure colorblind-friendly patterns (e.g., dashed borders for FP/FN).

## 7. Future Hooks
- Swap static sample with live scoring once backend model artifact lands.
- Allow uploading CSVs to replace the 50-row sample while reusing components.
- Persist slider selections in URL query params for shareable views.

