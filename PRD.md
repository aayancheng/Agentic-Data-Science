# PRD — Agentic Data Science (Codex)

## Objective
Create a repeatable **agentic data science** workflow using Codex that takes 3 popular public datasets to production-grade demos: notebooks + an interactive React app.

## Datasets
- **Credit Card Fraud** (Kaggle). Goal: high-recall classifier with calibrated probabilities and explainability.
- **NYC Taxi** (TLC). Goal: demand forecasting per hour/zone.
- **IMDb Reviews** (Stanford). Goal: binary sentiment classifier.

## Users
- Data Science Team (builders), Risk/Security (reviewers), Product/Exec (consumers).

## Success Metrics
- Fraud AUC ≥ 0.98 on stratified holdout; Recall@FPR=1% ≥ 0.60.
- Taxi MAPE ≤ 20% on last-month backtest; coverage of prediction intervals reported.
- IMDb Accuracy ≥ 90% on test; latency < 50ms per prediction (CPU).

## Scope (v1)
- Jupyter notebooks: EDA, features, baselines, model cards.
- FastAPI backend: 3 endpoints.
- React UI: 3 tabs with upload/input + results.
- CI: lint + API smoke test; per-dataset sample slices tracked in git.

## Out of Scope (v1)
- Full MLOps (feature store, deployment to cloud), advanced retraining pipelines, GPU serving.

## Constraints
- Keep raw data out of git; licensing compliance; no PII beyond datasets.

## Agentic Workflow (High-level)
1. **Plan**: Codex drafts a workplan and directory changes. **HITL** approve.
2. **Build**: Codex edits files, runs commands inside sandbox (`workspace-write`). **HITL** approve git commits.
3. **Evaluate**: Codex runs notebooks and prints metrics; **HITL** decide promotion thresholds.
4. **Ship**: Codex starts dev servers and opens ports; **HITL** does manual QA in UI.
5. **Document**: Codex generates READMEs + model cards + MRM pack.

## Task List (for Codex)
- Bootstrap repo (venv, pre-commit, CI).
- Implement download scripts; fetch data; create `data/sample` slices.
- Notebooks for Fraud, Taxi, IMDb with baseline models.
- Backend endpoints loading artifacts from `models/`.
- Frontend UI pages calling the API.
- CI smoke tests; Dockerfile (optional).

## Human-in-the-Loop (HITL)
- Confirm dataset licenses and responsible use.
- Approve any `danger-full-access` or network actions.
- Review PRs; enforce acceptance metrics before merging to `main`.
- Sign off on model cards and risk notes (bias, drift, failure modes).

## Risks & Mitigations
- **Data volume**: Use samples in git; document full-download steps.
- **Imbalance**: Use stratified splits; PR curves; threshold tuning.
- **Licensing**: Track sources and terms in `DATA_SOURCES.md`.

## Documentation Deliverables
- MRM-grade documentation (`docs/MRM_Documentation_Template.md`) completed for each model.
- Model cards for Fraud/Taxi/IMDb in `docs/model_cards/`.
- Evidence pack in `docs/validation/` (metrics, plots, backtests, fairness diagnostics).
- Monitoring plan with thresholds in `docs/monitoring/plan.md`.
- Data sources registry (`docs/DATA_SOURCES.md`) and data dictionary.
- Metrics summary autogen via `make docs` (pulls from `models/*/metrics.json`).

**HITL**: Approve documentation completeness and evidence sufficiency before submission.
