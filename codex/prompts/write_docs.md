You are Codex. Produce **MRM-grade documentation** for all three models (fraud, sentiment, taxi).

## Core Objective

Generate comprehensive, challenge-ready model documentation that meets enterprise Model Risk Management (MRM) standards. Each deliverable must be defensible to senior stakeholders, regulators, and independent model reviewers.

---

## Documentation Sections & Rigorous Success Criteria

### 1. Executive Summary

**Purpose**: Communicate model impact and risk posture in 1–2 pages for non-technical decision makers.

**Success Criteria**:

- ✅ **Clear business value statement**: Quantify expected impact (e.g., "Reduces fraud losses by ~X% based on Y% fraud recall at Z% precision").
- ✅ **Risk rating**: Classify as Low/Medium/High based on decision consequence + model uncertainty.
- ✅ **Key limitations upfront**: Highlight class imbalance, data drift risks, or blind spots.
- ✅ **Recommended threshold**: Specify operating point with trade-off rationale (e.g., "80% recall chosen to minimize undetected fraud vs. 3% false alarm cost").
- ✅ **Approval readiness**: Include sign-off checklist (data governance, bias review, monitoring plan status).

**Concrete Example (Fraud Model)**:

> _This model detects fraudulent credit card transactions with 95.2% ROC-AUC on holdout data. Expected to flag ~80% of fraud cases (recall) with ~5% false positive rate. Imbalance risk (0.17% fraud rate in training) mitigated via class-weighted loss. Requires weekly drift monitoring on transaction amount and time-of-day patterns._

---

### 2. Objective of the Model

**Purpose**: Define scope, success metrics, and decision boundaries.

**Success Criteria**:

- ✅ **Explicit use case**: Who uses it, for what decision, and what problem does it solve?
- ✅ **Target metric & threshold**: E.g., "Maximize recall ≥80% while minimizing precision drop below 95%."
- ✅ **Out-of-scope clarifications**: State what the model does NOT do (e.g., "Does not account for insider fraud" or "Not designed for international transactions").
- ✅ **Performance constraints**: Document latency, compute, fairness, or explainability requirements.
- ✅ **Success definition**: Measurable goals (e.g., "Model deployed iff validation F1 ≥0.70 AND no significant demographic bias detected").

**Concrete Example (Fraud Model)**:

> _Objective: Flag high-risk transactions for real-time review to reduce fraud losses. Success: Achieve ≥80% fraud recall on holdout set while maintaining precision ≥85%. Constraint: Inference latency <100ms. Out-of-scope: Account takeover, chargeback prediction, merchant risk scoring._

---

### 3. Data Acquisition for Model Training

**Purpose**: Document training data provenance, characteristics, and limitations.

**Success Criteria**:

- ✅ **Source & timeframe**: Where, when, and how were labels generated? (e.g., "Kaggle CC Fraud dataset: 284,807 transactions from Sept–Dec 2013").
- ✅ **Size & composition**: Report total rows, feature count, class distribution with percentages. **Quantify imbalance**: "0.17% fraud (n=492) vs. 99.83% legitimate (n=284,315)."
- ✅ **Feature descriptions**: Document all 30 features, including PCA components. State if data is anonymized/synthetic.
- ✅ **Data quality checks**:
  - Missing values: _"Zero missing values across all columns."_
  - Outliers: _"Transaction amounts range 0–25,691.58; fraud median $88.26 vs. legitimate median $22.00."_
  - Temporal coverage: _"48-hour observation window; peak fraud hours 02:00–05:00 UTC (3.2% fraud rate vs. 0.1% baseline)."_
- ✅ **Train/validation split**: Report stratified split (e.g., "80/20 stratified by class; 227,845 train, 56,962 validation").
- ✅ **Known limitations**: Class imbalance, PCA obfuscation limiting interpretability, single-currency scope.
- ✅ **Bias & fairness baseline**: Document demographic information if available (e.g., "Dataset does not include cardholder demographics; fairness analysis deferred pending data availability").

**Concrete Example (Fraud Model)** from notebook:

> _Training data: Kaggle credit card dataset, 284,807 transactions over 2 days. Features: Time (seconds), Amount, 28 anonymized PCA components. Class distribution: 0.17% fraud (n=492), 99.83% legitimate. Training set: 227,845 rows, 20 features. Missing values: None. Amount range: $0–$25,691 (median $22). Key finding: Fraud transactions show ~3.2% rate in 02:00–05:00 UTC, vs. 0.1% baseline. Limitation: Single-currency, no demographic data; imbalance requires class-weighted loss or threshold tuning._

---

### 4. Model Methodology and Results

**Purpose**: Explain how the model works, why it was chosen, and demonstrate performance.

**Success Criteria**:

#### 4a. Methodology

- ✅ **Algorithm selection & justification**: Why Logistic Regression, Random Forest, Gradient Boosting, etc.? (e.g., "Logistic Regression chosen for interpretability + baseline; Random Forest for non-linear patterns").
- ✅ **Preprocessing pipeline**: Scaling, feature engineering, class balancing (e.g., "StandardScaler + class_weight='balanced' to mitigate imbalance").
- ✅ **Hyperparameter tuning**: Document search strategy, CV folds, final hyperparameters. Report CV mean ± std.
- ✅ **Cross-validation results**: Show 5-fold CV metrics (ROC-AUC, PR-AUC, precision, recall) with fold-level variation.

#### 4b. Results

- ✅ **Holdout test metrics**: Report on validation/test set at chosen threshold.
  - **Fraud model example**: "Validation ROC-AUC: 0.985 ± 0.003 (5-fold CV), Logistic Regression achieves 96.2% recall, 93.1% precision @ threshold 0.45."
  - Include confusion matrix: TN, FP, FN, TP counts + rates.
- ✅ **Threshold analysis**: Show precision-recall trade-off curve. Justify chosen threshold (e.g., "Threshold 0.45 selected to achieve ≥80% recall while maintaining precision >90%").
- ✅ **Feature importance**: Top 10 features by coefficient (LR) or importance (RF). Example from notebook:
  > _Top features: V4, V10, V3, V14, V17 (absolute coefficients 1.2–2.8; Log Reg). Random Forest top features: V4, V14, V10, V3, V12 (importances 0.08–0.15)._
- ✅ **Comparison to baseline**: Report performance vs. majority class, random model, or prior model.
- ✅ **Calibration**: Report calibration plots or Brier score to assess probability well-calibration.
- ✅ **Statistical significance**: Confidence intervals or bootstrap confidence bounds on key metrics.

**Concrete Example (Fraud Model)** from notebook results:

> _Algorithm: Logistic Regression (balanced class weight) vs. Random Forest (200 trees, max_depth=10). CV Results (5-fold, n=227,845 train):_
>
> _Logistic Regression:_ > _- ROC-AUC: 0.9525 ± 0.0045_ > _- PR-AUC: 0.8234 ± 0.0156_ > _- Recall@0.5: 0.8120 ± 0.0234_ > _- Precision@0.5: 0.8934 ± 0.0178_
>
> _Random Forest:_ > _- ROC-AUC: 0.9812 ± 0.0031_ > _- PR-AUC: 0.9143 ± 0.0089_ > _- Recall@0.5: 0.9234 ± 0.0128_ > _- Precision@0.5: 0.9012 ± 0.0145_
>
> _Holdout Validation (56,962 rows): Random Forest selected (superior ROC-AUC, recall, and feature stability). Threshold tuning at target 80% recall yields optimal threshold 0.42 (precision 92.1%, recall 81.3%). Confusion matrix: TP=405, FP=36, FN=98, TN=56,423._

---

### 5. Model Risk and Future Improvements

**Purpose**: Surface known limitations, risks, and mitigation strategies.

**Success Criteria**:

- ✅ **Data risks**:
  - Class imbalance: _"0.17% fraud → oversampling or class-weight used; residual recall bias remains."_
  - Distribution shift: _"Fraud patterns evolve; time-of-day rates vary by season; monitor monthly."_
  - Generalization: _"Trained on single currency, 2-day window; unvalidated on international, longer-horizon data."_
- ✅ **Model risks**:
  - Interpretability: _"PCA components anonymize feature meanings; SHAP analysis recommended for edge cases."_
  - False positives: _"3% false alarm rate → ~8,000 legitimate transactions flagged daily (at 1M transactions/day); customer friction cost to quantify."_
  - Threshold drift: _"Optimal threshold assumes 0.17% fraud baseline; monitor fraud rate; recalibrate if >0.3% observed."_
- ✅ **Fairness & bias**:
  - _"No demographic features available; fairness analysis deferred. Recommend collecting geography, customer tenure, card type for bias audit."_
  - _"Time-of-day patterns show 32x fraud rate variance (3.2% vs. 0.1%); investigate cardholder region, merchant type confounds."_
- ✅ **Operational risks**:
  - Latency: _"Inference <100ms required; monitor tail latencies, feature lookup time."_
  - Rollback: _"Prior model baseline ROC-AUC 0.92; new model 0.981; rollback trigger: if validation ROC drops <0.95."_
- ✅ **Future improvements** (prioritized):
  1. **Short-term (1–2 weeks)**:
     - [ ] Add SHAP global explanations + sample local explanations for top 20 flagged transactions.
     - [ ] Implement monthly drift monitoring on transaction amount, time-of-day, velocity features.
     - [ ] Create feedback loop to re-label false positives confirmed by investigators.
  2. **Medium-term (1–3 months)**:
     - [ ] Incorporate merchant category code, cardholder tenure, velocity features (# txns in past 24h).
     - [ ] Train ensemble (Logistic Regression + Random Forest) for improved robustness.
     - [ ] Conduct fairness audit by geography, card type; implement fairness constraints if bias detected.
  3. **Long-term (3–6 months)**:
     - [ ] Develop transaction graph model (cardholder relationships, merchant networks) for ensemble.
     - [ ] Implement active learning to prioritize labeling of uncertain cases.
     - [ ] Establish automated retraining pipeline with monthly data + bias monitoring.

---

### 6. Model Training Code Explanation

**Purpose**: Enable reproducibility and audit trail.

**Success Criteria**:

- ✅ **Environment & dependencies**: Python version, key packages (pandas, scikit-learn), versions. _"Python 3.9.5, scikit-learn 1.0.2, pandas 1.3.4."_
- ✅ **Data loading**: Exact path, data shape, preprocessing steps. Example:
  ```python
  # Load training data (284,807 rows, 31 columns)
  df = pd.read_csv('data/raw/credit_card_fraud/creditcard.csv')
  X_train = train_df[FEATURES].copy()  # 227,845 × 30 features
  y_train = train_df['Class'].copy()    # 227,845 × 1, 99.83% class 0
  ```
- ✅ **Feature engineering**: All transformations documented with rationale.
  ```python
  # StandardScaler applied to normalize PCA components (mean=0, std=1)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  ```
- ✅ **Model training code**: Full pipeline with hyperparameters.
  ```python
  rf_model = Pipeline([
      ('clf', RandomForestClassifier(
          n_estimators=200, max_depth=10, min_samples_leaf=10,
          class_weight='balanced', n_jobs=-1, random_state=42
      )),
  ])
  rf_model.fit(X_train, y_train)
  ```
- ✅ **Validation & evaluation code**: Metrics computation, threshold selection, reproducibility seeds.
  ```python
  # Cross-validation: 5-fold stratified on class
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  roc_auc = roc_auc_score(y_val, proba)
  precision, recall, thresholds = precision_recall_curve(y_val, proba)
  ```
- ✅ **Output artifacts**: Saved model, feature names, scaler state, metrics JSON.
  ```json
  {
    "model_type": "RandomForest",
    "roc_auc_validation": 0.9812,
    "pr_auc_validation": 0.9143,
    "recall_at_threshold_0.42": 0.813,
    "precision_at_threshold_0.42": 0.921,
    "confusion_matrix": { "tp": 405, "fp": 36, "fn": 98, "tn": 56423 }
  }
  ```
- ✅ **Reproducibility**: Document random seeds, data version, exact command to re-run. _"To reproduce: `python train_model.py --data-version 2025-11-01 --seed 42`"_

---

## Self-Review for Model Risk Management Effectiveness

**Before finalizing documentation, you MUST conduct a rigorous self-review against these criteria:**

### Completeness Checklist

- [ ] **Executive Summary**: Business impact quantified, risk rating assigned, threshold rationale explained?
- [ ] **Objective**: Use case, success metric, constraints, and out-of-scope items clearly defined?
- [ ] **Data**: Size, composition, class distribution, missing values, drift risks documented?
- [ ] **Methodology**: Algorithm choice justified, CV results reported with fold variation, hyperparameters specified?
- [ ] **Results**: Holdout metrics, confusion matrix, feature importance, threshold trade-off analysis included?
- [ ] **Risk**: Data risks, model risks, fairness gaps, operational risks, and mitigation plans listed?
- [ ] **Code**: Full pipeline reproducible from path, versions, seeds, and saved artifacts?

### Rigor Checklist (Challenge-Ready)

- [ ] **Quantification**: All claims backed by metrics (not "good" but "0.981 ROC-AUC")?
- [ ] **Limitations acknowledged**: Every success criterion paired with known limitation or caveat?
- [ ] **Fairness gaps stated**: If fairness analysis incomplete, why, and when will it be done?
- [ ] **Threshold justified**: Operating point explained in context of business cost matrix, not arbitrary?
- [ ] **Drift triggers defined**: What happens if fraud rate, transaction amount, or other KPI drifts beyond bounds?
- [ ] **Rollback plan**: What is the fallback if this model fails validation or production checks?

### MRM Challenge Questions (Anticipate & Answer)

1. **Data Governance**: Is the training data versioned, labeled by humans or ML, and subject to SLAs?
2. **Bias & Fairness**: Can this model discriminate by protected attributes (race, gender, age)? If not analyzed, why?
3. **Explainability**: For a declined transaction, can you explain why in <2 minutes to a customer?
4. **Monitoring**: What metrics will you track weekly? What thresholds trigger retraining?
5. **Compliance**: Does this model meet regulatory requirements (GDPR, Fair Lending, Consumer Protection)?
6. **Performance**: Is validation performance stable across demographic segments or time windows?

---

## Inputs & Evidence

- Notebooks in `notebooks/` with EDA, metrics, and plots.
- Metrics JSON in `models/*/metrics.json` (produce if missing).
- Plots saved to `docs/images/`.

## Deliverables

1. **Model Documentation** (`docs/MRM_Documentation_Template.md`): Fill completely with all 6 sections above, rigor checklist answers, and MRM challenge Q&A.
2. **Model Cards** (`docs/model_cards/{fraud,imdb,taxi}.md`): Structured summaries with metrics, risk rating, and threshold recommendations.
3. **Validation Evidence** (`docs/validation/`): Confusion matrices, calibration curves, drift plots, stratified performance breakdowns.
4. **Monitoring Plan** (`docs/monitoring/plan.md`): Weekly/monthly KPIs, drift detection thresholds, retraining triggers, rollback procedures.
5. **Metrics Summary** (`docs/metrics_summary.md`): Comparative table across all three models with key findings and recommendations.

---

## Success Outcome

Documentation that passes independent MRM review, enables confident model deployment, and provides clear audit trail for regulators or stakeholders.
