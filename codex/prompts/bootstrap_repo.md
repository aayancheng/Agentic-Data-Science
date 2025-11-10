You are Codex, a coding agent running in my terminal/IDE.

Goal: Bootstrap this repo for an agentic data science workflow.

Tasks:
- Ensure a Python 3.11 venv at .venv and install deps from requirements.txt.
- Create a .env.example with placeholders (KAGGLE_USERNAME, KAGGLE_KEY).
- Add a pre-commit config for black + ruff.
- Generate a .gitignore based on Python/Node/Jupyter/data (respect existing entries).
- Create a GitHub Actions workflow `.github/workflows/ci.yml` that runs lint and a smoke test for the FastAPI backend.
- Initialize git, make the first commit, and (after I paste the remote) set up origin and push.

Constraints:
- Ask for my approval before pushing or creating a public repo.
- Keep raw data out of git; only commit `data/sample` slices.
