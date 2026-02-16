# Repository Documentation

This document provides a complete, high-level guide to the AI-Mastery-2026 repository. It
covers project structure, key modules, entry points, configuration, and how to navigate the
codebase.

## Project structure

Top-level layout:

- `src/` -> core Python modules (math, ML, LLM, production)
- `tests/` -> pytest test suites and fixtures
- `notebooks/` -> weekly learning notebooks
- `docs/` -> guides, system design docs, and reference material
- `scripts/` -> runnable utilities and benchmarks
- `app/` -> app-level assets and entry points
- `models/` -> saved models and artifacts
- `config/` -> configuration files and defaults
- `templates/` -> reusable templates
- `case_studies/`, `research/`, `benchmarks/`, `interviews/`, `sprints/` -> supporting material

## Core modules (src)

- `src/core/` -> math and first-principles implementations
- `src/ml/` -> ML and DL implementations
- `src/llm/` -> transformer and LLM systems
- `src/production/` -> production engineering utilities

## Week5 backend (research/week5-backend/week5_backend)

This is a standalone FastAPI RAG backend with ingestion, retrieval, reranking, and evaluation.

Key entry points:
- `research/week5-backend/week5_backend/app/main.py` -> FastAPI application
- `research/week5-backend/week5_backend/run_api.py` -> local dev server entry
- `research/week5-backend/week5_backend/pipelines/online_query.py` -> online RAG pipeline
- `research/week5-backend/week5_backend/pipelines/offline_index.py` -> offline indexer

Full explanations:
- `research/week5-backend/week5_backend/docs/full_explanations/README.md` -> full function/class docs
- `research/week5-backend/week5_backend/docs/line_by_line/README.md` -> line-by-line file walkthroughs

## Configuration

Primary configuration examples:
- `research/week5-backend/week5_backend/config/settings.example.yaml`
- `config/` for repo-wide configuration defaults

Runtime env var:
- `WEEK5_BACKEND_CONFIG` -> overrides Week5 backend config path

## Running the Week5 backend

1) Set up your environment and dependencies.
2) Start the API:

```bash
python research/week5-backend/week5_backend/run_api.py
```

3) Index data:

```bash
python research/week5-backend/week5_backend/pipelines/offline_index.py --source <PATH>
```

4) Query the API using `/query`.

## Testing and linting

Use Makefile targets:

- `make test`
- `make test-cov`
- `make lint`
- `make format`

## Documentation map

- `docs/USER_GUIDE.md` -> overall usage guide
- `docs/system_design_solutions/` -> architecture solutions
- `docs/line_by_line/` -> line-by-line RAG pipeline explanations
- `docs/full_explanations/` -> full function/class explanations for Week5 backend

## How to extend

- Add new ingestion sources in `rag/ingestion.py`.
- Add new tools in `agents/tooling.py` and register in the pipeline.
- Add vector store backends in `storage/`.
- Add evaluation datasets in `evaluation/*.jsonl`.

## Conventions

- Python 3.10+
- Type hints for functions
- Black + isort formatting
- pytest for testing

