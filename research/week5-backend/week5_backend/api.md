# API Spec (draft)

## Health
`GET /healthz`
- Response: `{ "status": "ok", "version": "0.1.0" }`

## Ingestion
`POST /ingest`
- Body:
  - `tenant_id: str`
  - `source_type: str` (file|web|pdf)
  - `uri: str`
  - `metadata: dict`
- Response:
  - `ingestion_id: str`
  - `status: str`

## Query
`POST /query`
- Body:
  - `tenant_id: str`
  - `question: str`
  - `filters: dict` (optional)
  - `top_k: int` (optional)
- `mode: str` (rag|agentic|hybrid)
- Note: `hybrid` uses BM25 + vector retrieval (requires BM25 index file).
- Response:
  - `answer: str`
  - `citations: list`
  - `trace_id: str`
  - `model: str`

## Evaluation
`POST /eval/run`
- Body:
  - `dataset_id: str`
  - `mode: str`
- Response:
  - `run_id: str`
  - `status: str`

## Feedback
`POST /feedback`
- Body:
  - `trace_id: str`
  - `rating: int` (1-5)
  - `notes: str`
- Response:
  - `status: str`
