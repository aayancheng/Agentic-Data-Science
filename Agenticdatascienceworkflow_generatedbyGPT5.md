# Codex Data Science Workflow

A complete, VS Code–ready, **agentic data science** workflow powered by **OpenAI Codex**. It covers repository setup, dataset acquisition, notebooks, a demo app (FastAPI + React), governance-grade documentation for effective challenge, and end-to-end VS Code steps.

---

## 1) Overview & Goals

- **Objective**: Build a repeatable workflow that turns public datasets into:

  1. EDA + baseline models (notebooks), and
  2. A demo app (FastAPI API + Vite React UI).

- **Agentic approach**: Use Codex via CLI, VS Code extension, and headless execution to plan, code, test, document, and open PRs, with **Human-in-the-Loop (HITL)** gates for quality and compliance.

**Datasets (examples)**

- **Credit Card Fraud** (tabular, imbalanced) — Kaggle `mlg-ulb/creditcardfraud` (requires Kaggle auth)
- **NYC Taxi Trips** (time series / geo) — NYC TLC Trip Record Data
- **IMDb Movie Reviews** (NLP) — Stanford (aclImdb)

**Outcomes**

- Clear acceptance metrics (e.g., Fraud AUC & Recall@1% FPR, Taxi MAPE, IMDb Accuracy)
- Governance-ready docs (MRM template, model cards, validation evidence, monitoring plan)
- Working demo app to showcase value to stakeholders

---

## 2) Repository Structure (single workspace)

```
AgenticDataScience/
  apps/react/            # React + FastAPI demo app
  codex/                 # Reusable prompts + cloud config
  data/                  # raw/ (ignored by git), sample/ (small checked-in slices)
  docs/                  # MRM-grade documentation + evidence
  notebooks/             # EDA + modeling
  scripts/               # Data downloads and utilities
  .vscode/               # VS Code tasks/launch/recommendations
  .gitignore
  .gitattributes
  Makefile
  PRD.md
  README.md
```

---

## 3) Kick Off in VS Code — Step by Step

> macOS commands shown; Windows equivalents indicated.

1. **Install prerequisites**

   - VS Code, Git, Python 3.11+, Node.js (LTS)
   - VS Code extensions: _Python_, _Jupyter_, _(optional)_ ESLint/Prettier, **Codex VS Code**
   - Verify: `python3 --version && node -v && git --version`

2. **Open the project**

   - Place the folder at `~/Documents/AgenticDataScience/`
   - VS Code → **File → Open Folder…** → `AgenticDataScience` → **Trust**

3. **Create & select the Python venv**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

   - Command Palette → **Python: Select Interpreter** → choose `.venv`

4. **Install dependencies**

   - **Tasks: Run Task → Install Python deps** (or):

     ```bash
     pip install -r requirements.txt
     pip install -r apps/react/backend/requirements.txt
     ```

5. **Configure data access (Fraud)**

   - Kaggle: set `~/.kaggle/kaggle.json` **or** `.env` with `KAGGLE_USERNAME`/`KAGGLE_KEY`
   - For aayan macbook:
   - After download the kaggle.json from Kaggle API Website
   - pip install kaggle
   - mv /Users/aayan/Downloades/kaggle.json ~/.kaggle/
   - chmod 600 ~/.kaggle/kaggle.json

6. **Download data & create samples**

   - **Tasks: Run Task → Download data** (or `python scripts/download_data.py`)
   - **Tasks: Run Task → Make sample slices** to keep small artifacts under `data/sample/`

7. **Run notebooks**

   - Open `notebooks/*` → select Kernel **Python 3 (.venv)** → **Run All**
   - Use the **Codex extension** to elaborate cells (e.g., SHAP explanations)

8. **Start backend (FastAPI)**

   - **Run & Debug → FastAPI (Uvicorn)** → _Start Debugging_
   - Or terminal: `python -m uvicorn apps.react.backend.main:app --reload`
   - Health: `http://localhost:8000/health`

9. **Start frontend (React)**

   ```bash
   cd apps/react/frontend
   npm install
   npm run dev   # http://localhost:5173
   ```

   - Or use **Run & Debug → Vite Frontend (Browser)**

10. **Agentic build with Codex**

    - Use prompt pack in `codex/prompts/*`
    - Example (from repo root):

      ```bash
      codex exec --cd ~/Documents/AgenticDataScience --full-auto < codex/prompts/build_react_app.md
      ```

    - **HITL** gates: approve pushes/PRs, verify metrics & licensing

11. **Git & PRs**

    ```bash
    make init  # or: git init && git add . && git commit -m "bootstrap"
    git remote add origin <your_repo_url>
    git push -u origin main
    ```

    - Ask Codex (terminal) to branch, commit, and open PRs

12. **Build docs for effective challenge**

    - **Tasks: Run Task → Build docs** (or `make docs`) to produce `docs/metrics_summary.md`
    - Compile the MRM pack with Codex:

      ```bash
      codex exec --cd ~/Documents/AgenticDataScience --full-auto < codex/prompts/write_docs.md
      ```

---

## 4) Codex Agentic Capabilities (how we use it)

- **Planning** — `codex/prompts/plan_work.md` produces a DAG of tasks, milestones, metrics and acceptance criteria
- **Scaffolding** — `codex/prompts/bootstrap_repo.md` sets up venv, pre-commit, CI, `.gitignore`, and git init
- **Building** — `codex/prompts/build_react_app.md` implements API endpoints and UI stubs
- **Documentation** — `codex/prompts/write_docs.md` compiles MRM-grade docs and model cards
- **Headless runs** — `codex exec --full-auto` with approval gates for network/PR actions

**Human-in-the-Loop (HITL)**

- Approve any network/push/PR actions
- Validate licensing and data lineage
- Enforce PRD acceptance metrics before merge to `main`
- Sign off model cards, risk notes, and monitoring thresholds

---

## 5) Data Acquisition & Scripts

```bash
python scripts/download_data.py   # Fraud (Kaggle), NYC Taxi (Parquet), IMDb (tar.gz)
python scripts/make_sample_slices.py
```

- Keep **raw** data out of git (`data/raw/` is ignored)
- Commit only `data/sample/` small slices for demos

---

## 6) Notebooks (EDA & Baselines)

- `notebooks/fraud_eda.ipynb` — class imbalance, stratified split, baselines, SHAP
- `notebooks/taxi_eda.ipynb` — time-based splits, seasonality, baseline forecasting
- `notebooks/imdb_eda.ipynb` — TF‑IDF + Logistic Regression, confusion matrix, latency

**Acceptance targets (example)**

- Fraud: **AUC ≥ 0.98**, **Recall@1% FPR ≥ 0.60**, good calibration (ECE/Brier)
- Taxi: **MAPE ≤ 20%** on last-month backtest; interval coverage reported
- IMDb: **Accuracy ≥ 90%**; single-request latency < 50ms (CPU)

---

## 7) App (FastAPI + React)

**Backend**: `apps/react/backend/main.py` (CORS enabled)

- `POST /api/fraud/score` — CSV → probability/flags (+SHAP later)
- `POST /api/sentiment/predict` — text → sentiment
- `GET/POST /api/taxi/forecast` — (to implement) hourly counts for a zone

**Frontend**: `apps/react/frontend` (Vite)

- Tabs: **Fraud**, **Taxi**, **Sentiment**
- Simple forms, basic results display; extend with charts/tables

---

## 8) Documentation for Effective Challenge (MRM-grade)

**Folder**: `docs/`

- `MRM_Documentation_Template.md` — master report (scope/governance → sign-offs)
- `model_cards/` — per-model cards (`fraud_model_card.md`, `taxi_model_card.md`, `imdb_model_card.md`)
- `checklists/effective_challenge_checklist.md` — validator checklist
- `DATA_SOURCES.md` — URLs, licenses, cadence
- `monitoring/plan.md` — drift/perf/calibration/fairness thresholds & escalation
- `validation/` — evidence pack (metrics, plots, backtests, fairness diagnostics)
- `scripts/build_docs.py` + **Makefile** `docs` target → `docs/metrics_summary.md`

**Submission package**

- Completed `MRM_Documentation_Template.md`
- Final model cards
- Evidence pack in `docs/validation/`
- Monitoring plan
- Commit SHA + environment manifest

---

## 9) PRD Snapshot (in repo as `PRD.md`)

- Objective, datasets, users, success metrics, scope/out-of-scope, constraints
- Agentic workflow stages (Plan → Build → Evaluate → Ship → Document)
- Task list for Codex + HITL checkpoints
- Risks & mitigations (data volume, imbalance, licensing)
- Documentation deliverables and acceptance gates

---

## 10) Guardrails & Good Practices

- Keep `data/raw/` out of git; use LFS for large artifacts if needed
- Reproducibility: pin seeds, log versions, save model cards
- Document threshold trade-offs (e.g., recall vs FPR for fraud)
- Fairness & explainability with clear narratives
- CI for lint & smoke tests; PR reviews mandatory

---

## 11) Handy Commands

```bash
# Create & activate venv
python3 -m venv .venv && source .venv/bin/activate

# Install deps
pip install -r requirements.txt
pip install -r apps/react/backend/requirements.txt

# Run API & UI
uvicorn apps.react.backend.main:app --reload
cd apps/react/frontend && npm install && npm run dev

# Fetch data & make samples
python scripts/download_data.py
python scripts/make_sample_slices.py

# Build docs
make docs

# Codex agent runs
codex exec --cd ~/Documents/AgenticDataScience --full-auto < codex/prompts/bootstrap_repo.md
codex exec --cd ~/Documents/AgenticDataScience --full-auto < codex/prompts/plan_work.md
codex exec --cd ~/Documents/AgenticDataScience --full-auto < codex/prompts/build_react_app.md
codex exec --cd ~/Documents/AgenticDataScience --full-auto < codex/prompts/write_docs.md
```

---

## 12) Human‑in‑the‑Loop (HITL) Checklist — Full Version

Use this checklist to operationalize HITL gates across the Codex agentic data‑science workflow. It is designed for **effective challenge** and aligns with common MRM practices.

> **How to use**: Each gate should have a named approver and evidence (link to PRs, notebooks, reports, or dashboards). Keep this next to your PRD and MRM doc.

### A. Governance & Scope

- [ ] **Business objective** is clearly stated, measurable, and approved.
- [ ] **Intended use / out‑of‑scope use** documented; misuse scenarios considered.
- [ ] **Materiality/tier** assessed with rationale; risk appetite documented.
- [ ] **Roles & RACI** (Owner / Dev / Validator / Approver) named and reachable.
- [ ] **Policy mapping** (internal standards/regulations) referenced in PRD and docs.

### B. Data Sources, Licensing & Lineage

- [ ] **DATA_SOURCES.md** complete (URLs, license terms, cadence, retention limits).
- [ ] **Lineage** from raw → cleaned → features → train/serve diagrammed.
- [ ] **PII status** confirmed (none expected in public datasets); redaction policy if discovered.
- [ ] **Sampling/time windows** justified; leakage controls described.
- [ ] **Data dictionary** present with types, units, valid ranges.

### C. Data Quality & Representativeness

- [ ] Automated **schema & quality checks** (missingness, outliers, duplicates) executed; results attached.
- [ ] **Representativeness**: train vs expected production distributions compared; gaps noted.
- [ ] **Bias/fairness** prelim assessment performed (where applicable) with metrics selected.

### D. Modeling Methodology

- [ ] **Candidate algorithms** and chosen approach justified (pros/cons).
- [ ] **Cross‑validation** design appropriate (stratified K‑fold / time‑based split).
- [ ] **Imbalance handling** (fraud): class weights / sampling / cost‑aware thresholding chosen and justified.
- [ ] **Hyperparameter search** documented with seeds, budget, and search space.
- [ ] **Calibration** method (Platt/Isotonic) evaluated; ECE/Brier reported.
- [ ] **Explainability** approach fixed (global + local SHAP/PDP); examples curated.

### E. Performance, Backtesting & Acceptance

- [ ] **Primary metrics** achieved on holdout/OOT per PRD (Fraud AUC + Recall@1% FPR; Taxi MAPE; IMDb Accuracy).
- [ ] **Confidence intervals** or variability reported; statistical significance addressed.
- [ ] **Subgroup stability** checked; fairness metrics within tolerance.
- [ ] **Stress tests**: sensitivity to missing/corrupted features and seasonality.
- [ ] **Promotion decision** documented (Go/No‑Go) with rationale.

### F. Implementation & Controls

- [ ] **API contract** (request/response, error codes) documented and versioned.
- [ ] **Latency/throughput** benchmarks meet SLOs.
- [ ] **Fallbacks/safe defaults** defined for errors or unavailable features.
- [ ] **Security & secrets** handled via env vars/secret store; `.env.example` provided.
- [ ] **Reproducibility**: env manifest, seeds, data snapshot hashes recorded.

### G. Documentation & Evidence

- [ ] `docs/MRM_Documentation_Template.md` completed with hyperlinks to evidence.
- [ ] **Model cards** finalized for Fraud/Taxi/IMDb with metrics and limitations.
- [ ] **Validation evidence** (ROC/PR, calibration, confusion matrices, drift checks) exported to `docs/validation/`.
- [ ] **Monitoring plan** thresholds and escalation path set in `docs/monitoring/plan.md`.

### H. Change Management & CI/CD

- [ ] **Branching & PRs** enforced; peer review completed; risk/validator sign‑offs captured.
- [ ] **CI** runs lint + unit/smoke tests; results attached to PR.
- [ ] **Release notes** and version bump recorded (semver) with commit SHA.

### I. Deployment Readiness (Demo App)

- [ ] **FastAPI** health route verified; endpoints function against sample payloads.
- [ ] **React UI** manual QA completed (Fraud/Taxi/Sentiment tabs) with screenshots.
- [ ] **CORS**/network config safe for intended environment.

### J. Monitoring & Alerts (Post‑deployment)

- [ ] **Data drift** monitors (e.g., PSI/KS) operational with thresholds.
- [ ] **Performance** guardrails (e.g., Recall@1% FPR drop > 10%) create alerts.
- [ ] **Calibration** monitor (ECE) and **fairness** parity checks scheduled.
- [ ] **Alerting** wired to owners + ticketing; playbook documented.

### K. Post‑Incident & Model Lifecycle

- [ ] **Incident playbook** (rollback, safe mode, comms) accessible.
- [ ] **Retraining policy** (cadence/triggers) defined; challenger strategy documented.
- [ ] **Retirement criteria** and archival process written.

### L. Final Sign‑offs

| Role                  | Name | Date | Decision | Notes |
| --------------------- | ---- | ---- | -------- | ----- |
| Model Owner           |      |      |          |       |
| Independent Validator |      |      |          |       |
| Risk/Compliance       |      |      |          |       |
| Product/Business      |      |      |          |       |
| Model Committee       |      |      |          |       |

---

## 13) Glossary

- **HITL** — _Human‑in‑the‑Loop_: required human review/approval points that bound agent autonomy and ensure compliance, safety, and quality.
- **MRM** — _Model Risk Management_: policies and practices for governing model lifecycle, including development, validation, and monitoring.
