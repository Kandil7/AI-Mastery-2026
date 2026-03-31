# Gemini Agent Context: AI-Mastery-2026 Project Overview

This document provides a concise overview of the `AI-Mastery-2026` project, its purpose, architecture, and key operational commands, specifically tailored for efficient processing by the Gemini AI agent.

---

## 1. Project Overview

*   **Purpose:** `AI-Mastery-2026` is a comprehensive AI Engineer Toolkit and learning repository. It emphasizes a "White-Box Approach" for deep understanding, covering mathematical foundations, classical ML, deep learning, LLM engineering (Transformers, RAG, fine-tuning), and production deployment.
*   **Architecture:** Modular structure with `src/` for core implementations, `notebooks/` for learning, `research/` for independent projects (e.g., RAG engine), `docs/` for all documentation, `models/` for artifacts, and `scripts/` for utilities.
*   **Key Entry Points:**
    *   **Main Project Information:** `README.md` (at project root)
    *   **Detailed Documentation:** `docs/README.md` (main entry point for documentation suite)
*   **Target Audience:** Caters to AI beginners, mid-level engineers, and senior AI architects.

---

## 2. Repository Structure

The repository is organized into clearly defined sections. For detailed information, consult the `docs/` directory.

### Core Modules:

*   **`src/`**: Core implementations (`src/core`, `src/ml`, `src/llm`, `src/production`).
*   **`notebooks/`**: Jupyter notebooks (`notebooks/01_foundations`, `notebooks/02_classical_ml`, `notebooks/03_deep_learning`).
*   **`research/`**: Independent projects and sprints (e.g., `research/rag_engine`).
*   **`models/`**: Saved model artifacts.
*   **`scripts/`**: Utility scripts.
*   **`app/`**: FastAPI application entry point.

### Documentation (`docs/` directory):

*   **`docs/00_introduction/`**: Getting Started, User Guide, Contribution.
*   **`docs/01_learning_roadmap/`**: Detailed learning path, project roadmaps.
*   **`docs/02_core_concepts/`**: Math, ML/DL fundamentals, deep dives.
*   **`docs/03_system_design/`**: Architecture, deployment, MLOps, security.
*   **`docs/04_tutorials/`**: Practical guides, examples, troubleshooting.
*   **`docs/05_interview_prep/`**: Interview questions and prep.
*   **`docs/06_case_studies/`**: Real-world application examples.
*   **`docs/agents/`**: Agent-specific instructions (this file, `AGENTS.md`, `QWEN.md`).
*   **`docs/reference/`**: API references, glossaries.
*   **`docs/reports/`**: Project status, completion reports.
*   **`docs/assets/`**: Static assets for documentation.
*   **`docs/legacy_or_misc/`**: Files pending review/re-categorization.

---

## 3. Key Operational Commands (using `Makefile`)

*   **`make install`**: Set up environment and install dependencies.
*   **`make test` / `make test-cov`**: Run tests and generate coverage reports.
*   **`make lint` / `make format`**: Enforce code quality and formatting.
*   **`make run` / `make run-prod`**: Run the FastAPI application in development or production mode.
*   **`make docker-build` / `make docker-run`**: Manage Dockerized services.
*   **`make docs`**: Build Sphinx documentation.

---

## 4. Development Conventions

*   **Python 3.10+**, type hints, 100 char line limit (`black`, `isort`, `flake8`, `mypy`).
*   **`pytest`** for testing, with tests mirroring `src/` structure.
*   **Conventional Commits** for version control.
*   **Secrets** via environment variables.

---

## 5. Specific Guidance for Gemini

*   **Prioritize Structured Docs:** When seeking information, always refer first to the numbered directories within `docs/` (e.g., `docs/01_learning_roadmap/`).
*   **Navigation Hub:** Use `docs/README.md` as the primary guide for navigating the entire documentation suite.
*   **Context for Tasks:** For specific coding, testing, or architectural tasks, consult `docs/02_core_concepts/`, `docs/03_system_design/`, and the `src/` code directly.
*   **Task Management:** Use `docs/01_learning_roadmap/project_roadmaps_overview.md` for project status, backlog, and week-by-week checklists.