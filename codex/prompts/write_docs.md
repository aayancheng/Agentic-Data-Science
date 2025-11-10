You are Codex. Produce **MRM-grade documentation** for all three models.

Inputs & Evidence:
- Notebooks in `notebooks/` with EDA, metrics, and plots.
- Metrics JSON in `models/*/metrics.json` (produce if missing).
- Plots saved to `docs/images/`.

Deliverables:
1) Fill `docs/MRM_Documentation_Template.md` completely (replace placeholders).
2) Generate/refresh `docs/model_cards/*` with final numbers, and `docs/metrics_summary.md`.
3) Populate `docs/validation/` with exported confusion matrices, calibration curves, drift plots.
4) Update `docs/monitoring/plan.md` thresholds based on observed stability.
