# Model Documentation (MRM-Grade) — Credit Card Fraud Detection

## Cover & Inventory
- Model Name: Credit Card Fraud Classifier (Tabular)
- Model ID: FRAUD_CC_V1
- Version: v1.0.0 (demo baseline)
- Owner: Risk Analytics
- Developers: Agentic Data Science Team
- First Use Date: 2025-11-10
- Business Unit: Cards Risk
- Use Cases: Real-time transaction review queue prioritization; post-authorization monitoring
- Model Tier/Materiality: High — financial loss prevention; customer impact for false positives
- Status: Development (demo); ready for validation readout
- Related Systems: FastAPI service, React threshold explorer UI, offline notebooks
- Repositories: apps/react/backend, apps/react/frontend, notebooks/fraud_eda.ipynb, scripts/

## 1. Executive Summary
- Objective: Detect fraudulent card transactions and route high-risk items to manual review to reduce chargebacks and losses while managing false-positive workload.
- Benefit: Model achieves ROC-AUC ≈ 0.991 on validation with a tuned operating point targeting ≥80% recall. Expected to intercept ~80% of fraud with precision ≈ 0.684 on the validation slice; configurable threshold supports business tradeoffs via UI.
- Key Risks: Severe class imbalance (~0.17% fraud rate), potential drift in spending patterns and time-of-day behavior, PCA-derived features limit semantic explainability, calibration sensitivity to threshold selection.
- Acceptance Criteria & Results (validation snapshot):
  - Fraud: ROC-AUC 0.9905; PR-AUC 0.7375; tuned threshold ≈ 0.998 yields Recall 0.8163, Precision 0.6838; Confusion matrix: [[56826, 37], [18, 80]].
- Go/No-Go: Proceed to independent validation with focus on calibration, threshold policy, and drift monitoring. Do not deploy without monitoring and reviewer workflow safeguards.
- Limitations: Dataset is anonymized PCA components (reduced interpretability), no demographics; findings may not generalize across geographies/merchants.

## 2. Scope & Governance
- Business Scope & Boundaries: Card-present, e-commerce, and in-app transactions; initial geography as per dataset scope; single-currency context.
- In/Out of Scope: Not a chargeback prediction or merchant risk model; excludes account-takeover detection; not for adverse action decisions.
- Roles & Responsibilities: Owner (Risk Analytics); Developers (Agentic Data Science); Independent Validators (Model Risk); Approvers (Risk Committee).
- Policies/Standards Mapping: Threshold policy aligned to loss vs. alert-cost matrix; data licensing documented in docs/DATA_SOURCES.md.
- Approvals & Sign-offs: See Section 10.

## 3. Data Governance & Lineage
- Source Registry: Kaggle Credit Card Fraud (284,807 rows, 31 columns; Time, Amount, V1–V28 PCA, Class label). See docs/DATA_SOURCES.md.
- Time Windows: 2-day observation window (dataset), stratified 80/20 train/validation split.
- Sampling: Stratified by Class to preserve base rate in splits; leakage avoided by simple holdout methodology.
- Data Quality & Controls: No missing values across columns; Amount 0–25,691.16; heavy class imbalance.
- Representativeness & Bias: No demographic attributes present; fairness analysis not applicable at this stage.
- Evidence:
  - Class balance: docs/images/fraud_eda_table_class_balance.png
  - Summary stats (Time & Amount): docs/images/fraud_eda_table_summary_stats.png
  - Imbalance & amount distributions chart: docs/images/fraud_eda_chart_01.png
  - Top hours by fraud rate: docs/images/fraud_eda_table_top_hours.png

## 4. Feature Engineering
- Target: Fraud label `Class ∈ {0,1}`.
- Features: 30 predictors — Time, Amount, and 28 anonymized PCA components (V1–V28).
- Transformations: StandardScaler in baseline pipelines; class weighting (balanced) during training.
- Interpretability: PCA features are obfuscated; rely on SHAP/global diagnostics post-training for policy review.

## 5. Modeling Approach
- Candidates: Logistic Regression (interpretable baseline), Random Forest (non-linear), with class_weight='balanced'.
- Training: 3-fold Stratified CV for baseline evaluation; final holdout computed on 20% validation.
- Hyperparameters: 
  - Logistic Regression: default liblinear/saga equivalent via scikit-learn pipeline with StandardScaler.
  - Random Forest: typical defaults; importances reported as orientation (see evidence image).
- Thresholding & Policy: Threshold tuned to achieve ≥80% recall with highest precision found by PR-curve scan; current tuned threshold ≈ 0.9976.
- Explainability: Coefficient magnitudes (LR) and RF feature importances captured; future SHAP planned.
- Evidence:
  - Feature importance snapshots: docs/images/fraud_eda_table_feature_importance.png

## 6. Performance & Validation Results
- Cross-Validation: Not shown here in detail; see notebook for per-fold scores.
- Validation (holdout):
  - ROC-AUC: 0.9905; PR-AUC: 0.7375
  - Tuned threshold (≥80% recall): ~0.9976
  - Precision: 0.6838; Recall: 0.8163
  - Confusion Matrix (TN, FP; FN, TP): [[56826, 37], [18, 80]] (docs/images/fraud_eda_table_confusion_matrix.png)
- Threshold Trade-off Snapshot: docs/images/fraud_eda_table_best_threshold.png
- Business Interpretation: At tuned threshold, majority of fraud is captured at the cost of modest false positives—appropriate for review queues but must be paired with reviewer capacity.

## 7. Implementation & Controls
- Serving: FastAPI `/api/fraud/demo` serves calibrated probabilities for demo; production service to load trained artifact and compute live scores.
- UI: React threshold explorer enables operations to choose/validate operating point.
- Security & Privacy: Dataset anonymized; no PII in repository; ensure compliance when integrating production data.
- CI & Change: Version models and metrics JSON; basic tests on metrics helpers included.
- Monitoring & Alerts: Track weekly fraud rate, precision@policy threshold, recall via labeled samples, transaction amount and hour-of-day drift; triggers for re-calibration/retraining.

## 8. Assumptions, Limitations, Risks
- Assumptions: Similar fraud/legit behavior between training and deployment; threshold policy regularly reviewed.
- Limitations: PCA obfuscation reduces interpretability; dataset time window narrow; no demographic attributes for fairness.
- Risks & Mitigations: Drift — implement monitoring; Mis-calibration — periodic threshold tuning; False positives — human-in-the-loop review and feedback loop.

## 9. Documentation, Artifacts & Evidence
- Notebook: notebooks/fraud_eda.ipynb
- Images: docs/images/* (see links above)
- UI Plan: docs/fraud_threshold_explorer_plan.md
- Task Tracking: codex/prompts/fraud_threshold_explorer_tasks.md

## 10. Sign-offs
| Role | Name | Date | Decision | Notes |
|---|---|---|---|---|
| Owner |  |  |  |  |
| Independent Validator |  |  |  |  |
| Risk/Compliance |  |  |  |  |
| Model Committee |  |  |  |  |

---

Appendix: Additional Evidence
- Top hours by fraud rate and distributions demonstrate temporal lift; see docs/images/fraud_eda_table_top_hours.png and docs/images/fraud_eda_chart_01.png.
- Feature importance suggests V14, V10, V4, V12 among top signals across RF, with Amount & V14 notable for LR.
