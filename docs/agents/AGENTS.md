# Repository Guidelines for AI Agents

This document provides essential guidelines and an overview of the repository's structure for AI agents to effectively understand, navigate, and interact with the AI-Mastery-2026 project.

## Project Structure & Module Organization

The repository follows a clean and logical structure designed for both human and AI readability:

*   **`src/`**: Contains core Python modules (`core/`, `ml/`, `llm/`, `production/`) for from-scratch implementations and production-ready components.
*   **`tests/`**: Pytest suites and fixtures, mirroring the `src/` module structure.
*   **`notebooks/`**: Learning notebooks organized by foundational stages (e.g., `01_foundations/`, `02_classical_ml/`).
*   **`docs/`**: The central hub for all project documentation, structured into numbered categories for clear navigation. Key subdirectories include:
    *   **`docs/00_introduction/`**: General project overview, user guide, and contribution guidelines.
    *   **`docs/01_learning_roadmap/`**: Detailed learning path and project management roadmaps.
    *   **`docs/02_core_concepts/`**: Foundational math, ML/DL fundamentals, and deep dives.
    *   **`docs/03_system_design/`**: Architectural patterns, deployment, and MLOps.
    *   **`docs/04_tutorials/`**: Practical guides, examples, and troubleshooting.
    *   **`docs/05_interview_prep/`**: Interview questions and preparation materials.
    *   **`docs/06_case_studies/`**: Real-world application examples.
    *   **`docs/agents/`**: Specific instructions and context for AI agents (like this file, `GEMINI.md`, `QWEN.md`).
    *   **`docs/assets/`**: Images and static resources used in documentation.
    *   **`docs/legacy_or_misc/`**: Files pending review or re-categorization.
    *   **`docs/reference/`**: API references, glossaries, and technical specifications.
    *   **`docs/reports/`**: Project status, completion reports, and learning logs.
*   **`research/`**: Houses independent research projects and sprints (e.g., `rag_engine/`, `week01_rag_production/`). The former `sprints/` directory has been merged here.
*   **`scripts/`**: Runnable utilities (benchmarks, training, pipelines).
*   **`app/`**: Application-level assets and entry points (e.g., FastAPI application).
*   **`models/`**: Saved models and artifacts.
*   **`config/`**: Configuration files and environment defaults.
*   **`templates/`**: Reusable templates for docs or reports.
*   **`benchmarks/`**: Performance and cost benchmarks.

## Build, Test, and Development Commands

Use the `Makefile` targets for consistent workflows. Agents should reference these commands for common operations:

*   **`make install`**: Install dependencies from `requirements.txt`.
*   **`make test`**: Run the test suite (`pytest tests/ -v`).
*   **`make test-cov`**: Run tests with coverage report (`--cov=src --cov-report=html`).
*   **`make lint`**: Run `black --check`, `mypy`, and `flake8`.
*   **`make format`**: Format with `black` and `isort`.
*   **`make run`**: Start FastAPI in dev mode (`uvicorn src.production.api:app --reload`).
*   **`make docker-build`** / **`make docker-run`**: Build and run Docker images/compose.
*   **`make docs`**: Build Sphinx docs to `docs/_build/`.

## Coding Style & Naming Conventions

*   **Python 3.10+ only.**
*   **Type hints** required for all functions.
*   **Line length**: 100 characters.
*   **Formatters**: `black` (100 char), `isort`.
*   **Linters/type checks**: `flake8`, `mypy`.
*   Prefer descriptive module names and keep new code within existing subpackages.

## Testing Guidelines

*   **Framework**: `pytest` with `pytest-cov`.
*   Tests live in `tests/` and should mirror `src/` module names.
*   Run locally with `make test` or `make test-cov`.
*   Aim to maintain the existing high coverage target (95%+ badge).

## Commit & Pull Request Guidelines

*   **Commit messages** follow Conventional Commits seen in history: `feat: ...`, `docs: ...`, etc.
*   **PRs should**:
    *   Describe changes clearly and update `README.md` if the interface changes.
    *   Update version numbers in examples/README if releasing.
    *   Request a second reviewer before merge when possible.

## Security & Configuration Tips

*   Store secrets in environment variables (see `config/` and `.env` usage).
*   Avoid committing large model artifacts; place outputs under `models/` or `results/`.
*   **Agent-Specific Context**: `docs/agents/GEMINI.md` and `docs/agents/QWEN.md` provide specialized instructions for their respective AI agents.

## Important Note for AI Agents

When interacting with this repository, prioritize information from the numbered `docs/` directories for the most up-to-date and structured guidance. Use the `docs/README.md` as your primary navigation hub.