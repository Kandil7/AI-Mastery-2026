# Data Model

## Core entities
- Document
  - `doc_id: str`
  - `tenant_id: str`
  - `source_type: str`
  - `uri: str`
  - `hash: str`
  - `metadata: dict`
  - `created_at: datetime`

- Chunk
  - `chunk_id: str`
  - `doc_id: str`
  - `text: str`
  - `embedding: list[float]`
  - `metadata: dict`

- QueryTrace
  - `trace_id: str`
  - `tenant_id: str`
  - `question: str`
  - `model: str`
  - `latency_ms: int`
  - `cost_usd: float`
  - `created_at: datetime`

- Feedback
  - `trace_id: str`
  - `rating: int`
  - `notes: str`
  - `created_at: datetime`

## Indexing strategy
- Vector index keyed by `chunk_id` with metadata filters.
- Secondary metadata store for fast document lookup.
- Optional BM25 index for lexical retrieval.
