# Repository Guidelines

## Project Structure & Module Organization
- `apps/react/backend`: FastAPI entry `main.py`; add routers under `routers/` and keep shared utilities in `services/` modules.
- `apps/react/frontend`: React + Vite client; pages in `src/pages`, shared UI in `src/components`, and fraud helpers under `src/features/fraud`.
- `notebooks`: Exploratory analysis; name files `topic_owner.ipynb` and keep outputs light.
- `data`: `raw/` (gitignored) for large pulls, `sample/` for committed slices like `fraud_threshold_demo.json`.
- `scripts`: Operational helpers such as `make_sample_slices.py`; document CLI usage in the module docstring.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create an isolated Python env aligned with `requirements.txt`.
- `pip install -r requirements.txt` plus `pip install -r apps/react/backend/requirements.txt`: Install core DS and API deps.
- `uvicorn apps.react.backend.main:app --reload`: Serve the FastAPI backend with hot reload.
- `cd apps/react/frontend && npm install && npm run dev`: Install JS deps and start the Vite dev server.
- `npm test`: Run the Vitest suite for fraud metric helpers and UI logic.

## Coding Style & Naming Conventions
- Python: Follow PEP8 with 4-space indents and snake_case APIs. Run `python -m black apps/react/backend scripts notebooks` before committing.
- TypeScript/React: Favor functional components, hooks, PascalCase components, camelCase utilities, and colocated styles.
- Data assets and notebooks use descriptive lowercase names and start with a markdown cell summarizing scope.

## Testing Guidelines
- Frontend: Write Vitest specs beside the code (`foo.test.ts`) covering reducers, hooks, and critical rendering paths; mock network calls with `msw`.
- Backend: Add `pytest` suites under `apps/react/backend/tests`, asserting response schemas and edge cases. Use `coverage run -m pytest` on backend changes and keep touched modules ≥85% covered.
- Notebooks: When promoting logic into scripts, move validation checks into automated tests rather than keeping ad-hoc asserts in notebooks.

## Commit & Pull Request Guidelines
- Commits use short, imperative summaries similar to current history (`Update fraud cutoff doc`, `Add vitest coverage for helpers`). Group related changes; avoid "misc fixes".
- PRs include a problem statement, before/after screenshots for UI changes, test evidence (`npm test`, `pytest`, data checks), and linked issue IDs. Tag both data and frontend leads when changes span stacks.

## Security & Configuration Tips
- Never commit credentials or large datasets; `.env` values belong in `.env.example` with placeholders.
- Keep external pulls (Kaggle, NYC TLC) under `data/raw/` and describe preprocessing steps in the relevant script or notebook header.
