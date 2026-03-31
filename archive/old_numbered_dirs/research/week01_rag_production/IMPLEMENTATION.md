# From Notebook to Production: Week 01 RAG

This guide translates `plan.md` into concrete implementation steps inside
`sprints/week01_rag_production`. It shows how the notebook experiments become a
production-grade pipeline, with clear module boundaries and operational concerns.

## 1) What the notebooks prove

Notebook `notebooks/day1_hybrid_demo.ipynb` validates hybrid retrieval:
- Dense retrieval wins on conceptual queries.
- Sparse retrieval wins on exact keywords (IDs, error codes).
- Hybrid with RRF fuses the best of both.

Notebook `notebooks/day2_eval_pipeline.ipynb` validates evaluation:
- Context Recall checks if retrieval is correct.
- Faithfulness checks if generation is grounded in retrieved evidence.
- A CI-friendly RAGAS workflow is the target for production.

These are the core behaviors we must preserve in production: hybrid retrieval,
traceable answers, and measurable quality.

## 2) Production modules mapped to the plan

This sprint now uses a dedicated package under `sprints/week01_rag_production/src`.
The key runtime components align to the plan:

- Retrieval (Hybrid, RRF): `src/retrieval/retrieval.py`
  - Dense: embeddings via SentenceTransformers with optional Chroma persistence.
  - Sparse: TF-IDF for fast keyword matching.
  - Fusion: RRF (preferred) or weighted fusion.

- Orchestration: `src/pipeline.py`
  - Owns retrieval, prompt assembly, and generation.
  - Returns structured sources for UI and evaluation.

- API/UI surface: `api.py` and `ui.py`
  - API uses the sprint pipeline; Streamlit is a thin client.

What remains from the plan as next steps (notebook to production hardening):
- Ingestion & ETL (Unstructured/OCR): add `src/ingest/*` with connectors.
- Chunking strategies (semantic/agentic): add `src/chunking/*`.
- Reranking (cross-encoder or late interaction): add `src/rerank/*`.
- GraphRAG: add `src/graph/*` for entity/relationship retrieval.
- Agentic control (Self-RAG / CRAG): add `src/agents/*`.
- Evaluation & observability: add `src/eval/*` and `src/observability/*`.

## 3) Implementation steps (production path)

### Step A: Ingestion and data contracts
Goal: move from notebook lists to repeatable data pipelines.
- Implement `Document` schema (id/content/metadata).
- Add loaders for local files (PDF/MD/TXT) and a metadata envelope:
  - `source`, `ingested_at`, `document_type`, `acl`.
- Extract, normalize, and store raw text before chunking.

### Step B: Chunking with metadata
Goal: avoid context loss and preserve structure.
- Start with deterministic chunking (size + overlap) and document boundaries.
- Add semantic chunking (split on heading/paragraph) for legal/technical docs.
- Attach metadata to each chunk (page, section, offsets).

### Step C: Hybrid retrieval + fusion
Goal: consistent precision on IDs + semantics.
- Dense retrieval in Chroma (or in-memory fallback).
- Sparse retrieval via TF-IDF (upgrade to BM25 when needed).
- Fuse with RRF (stable across score scales).

### Step D: Reranking
Goal: improve top-k quality without raising recall cost.
- Lightweight reranker (lexical overlap) for MVP.
- Upgrade to cross-encoder or ColBERT if latency budget allows.

### Step E: Generation and guardrails
Goal: answers with explicit evidence.
- Prompt template that cites retrieved sources.
- Guardrail policies for refusal, PII, and out-of-scope requests.
- Add citation formatting for UI and API responses.

### Step F: Evaluation and observability
Goal: production feedback loop.
- Add batch eval to CI (RAGAS).
- Add tracing per request (query, retrieved ids, latency, tokens).
- Track metrics: recall@k, faithfulness, answer relevancy, p95 latency.

### Step G: Deployment and scaling
Goal: predictable performance in real traffic.
- Persist vector store to disk; warm indices on startup.
- Cache repeated queries (semantic cache).
- Add background indexing jobs and health checks.

## 4) How the current code aligns

Use `api.py` to run a minimal production skeleton:
- Index documents via `/index`.
- Query via `/query` to get answer + sources.

The notebook logic now maps cleanly:
- Hybrid retrieval code lives in `src/retrieval/retrieval.py`.
- The evaluation notebook can be re-pointed to the sprint API by collecting
  `question`, `contexts`, and `answer` from `/query`.

## 5) Suggested incremental roadmap (week01 scope)

1) Stabilize current pipeline
   - Add tests for retrieval fusion and ranking order.
   - Add persistent Chroma storage on disk.

2) Add evaluation hooks
   - Create a small eval set and add a script to run RAGAS offline.

3) Harden API
   - Add request validation, timeouts, and structured logging.

4) Add ingestion
   - Local file loader and PDF text extraction to seed index.

When these are done, the system matches the 2026 production RAG features
described in `plan.md` with a clear path to CRAG, GraphRAG, and advanced orchestration.
