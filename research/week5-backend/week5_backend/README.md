# Week 5 Backend: RAG at Scale (Multi-Model + Agentic RAG)

This folder is a full backend reference implementation outline for a production-grade RAG system
with multiple model providers and vector store backends. It is designed as a learning scaffold:
the code is minimal but realistic, and the docs describe real-world tradeoffs, operations, and APIs.

## What this includes
- End-to-end RAG backend layout (ingestion, indexing, retrieval, reranking, answer synthesis)
- Multi-model routing (OpenAI, Anthropic, local vLLM) via a common provider interface
- Multi-vector-store support (pgvector, Qdrant, Weaviate) with a shared adapter pattern
- Agentic RAG pipeline (planner + tool selection + retriever + verifier loop)
- Evaluation harness and metrics for offline quality checks
- API and data model docs

## Structure
- `app/`: FastAPI entrypoints and routers
- `core/`: configuration, logging, telemetry
- `rag/`: ingestion, chunking, embeddings, retrieval, answer/citations
- `agents/`: planner, tools, executor, policies
- `providers/`: LLM + embeddings provider interfaces and implementations
- `storage/`: vector database interfaces and adapters
- `pipelines/`: orchestration for offline indexing and online querying
- `evaluation/`: metrics, datasets, harness
- `config/`: example settings for providers and vector stores

## Quickstart (dev)
1) Copy config example and edit secrets:
   - `research/week5-backend/week5_backend/config/settings.example.yaml` -> `research/week5-backend/week5_backend/config/settings.yaml`
   - set env vars for provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
   - or set `WEEK5_BACKEND_CONFIG` to a custom config path

2) Install deps (see repo Makefile):
   - `make install`

3) Run API:
   - `python research/week5-backend/week5_backend/run_api.py`

4) Ingest and index:
   - `python research/week5-backend/week5_backend/pipelines/offline_index.py --source ./data`

5) Run eval on sample dataset:
   - `POST /eval/run` with `dataset_id: "sample_dataset"`

## Real-case scenarios included
- Customer support RAG with multi-tenant isolation
- Compliance/Legal RAG with strict citation constraints
- Engineering knowledge base with code-aware chunking
- Hybrid retrieval (BM25 + vectors + re-ranker) and tool-augmented answers
- Agentic triage: planner selects tools (RAG, SQL, web search) and validates

See `case_studies.md` for full examples.

## Notes
- This is a learning scaffold, not a production-ready package.
- Replace stubs and TODOs with real providers in `providers/` and `storage/`.
