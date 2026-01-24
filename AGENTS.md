# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core Python modules (`core/`, `ml/`, `llm/`, `production/`).
- `tests/`: pytest suites and fixtures.
- `notebooks/`: learning notebooks by week.
- `docs/`: user guides, system design docs, and Sphinx output (`docs/_build/`).
- `scripts/`: runnable utilities (benchmarks, training, pipelines).
- `app/`: app-level assets and entry points.
- `models/`: saved models and artifacts.
- `config/`: configuration files and environment defaults.
- `templates/`: reusable templates for docs or reports.
- `case_studies/`, `research/`, `benchmarks/`, `interviews/`, `sprints/`: supporting materials.

## Build, Test, and Development Commands
Use the Makefile targets for consistent workflows:
- `make install`: install dependencies from `requirements.txt`.
- `make test`: run the test suite (`pytest tests/ -v`).
- `make test-cov`: run tests with coverage report (`--cov=src --cov-report=html`).
- `make lint`: run `black --check`, `mypy`, and `flake8`.
- `make format`: format with `black` and `isort`.
- `make run`: start FastAPI in dev mode (`uvicorn src.production.api:app --reload`).
- `make docker-build` / `make docker-run`: build and run Docker images/compose.
- `make docs`: build Sphinx docs to `docs/_build/`.

## Coding Style & Naming Conventions
- Python 3.10+ only.
- Type hints required for all functions.
- Line length: 100 characters.
- Formatters: `black` (100 char), `isort`.
- Linters/type checks: `flake8`, `mypy`.
- Prefer descriptive module names and keep new code within existing subpackages.

## Testing Guidelines
- Framework: `pytest` with `pytest-cov`.
- Tests live in `tests/` and should mirror `src/` module names.
- Run locally with `make test` or `make test-cov`.
- Aim to maintain the existing high coverage target (95%+ badge).

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits seen in history: `feat: ...`, `docs: ...`, etc.
- PRs should:
  - Describe changes clearly and update `README.md` if the interface changes.
  - Update version numbers in examples/README if releasing.
  - Request a second reviewer before merge when possible.

## Security & Configuration Tips
- Store secrets in environment variables (see `config/` and `.env` usage).
- Avoid committing large model artifacts; place outputs under `models/` or `results/`.
