# Data Model

This model aligns with the current Week 5 RAG pipeline implementation and the expanded
case studies. It focuses on what is stored, what is returned, and which fields drive
retrieval, ranking, verification, and governance.

## Core entities

### Document
- `doc_id: str` -> unique document identifier
- `tenant_id: str` -> tenant boundary for access control
- `source_type: str` -> `web` | `file` | `pdf`
- `uri: str` -> original source location
- `hash: str` -> content hash for dedup and caching
- `metadata: dict` -> domain-specific attributes (see "Metadata fields")
- `created_at: datetime`
- `updated_at: datetime`

### Chunk
- `chunk_id: str` -> unique chunk identifier (`{doc_id}:{index}`)
- `doc_id: str`
- `text: str`
- `embedding: list[float]` -> stored in the vector DB
- `metadata: dict` -> inherited + chunk-level fields

### BM25CorpusRecord
- `chunk_id: str`
- `doc_id: str`
- `text: str`

### RetrievalResult
- `chunk_id: str`
- `doc_id: str`
- `text: str`
- `score: float`
- `metadata: dict`

### Citation
- `chunk_id: str`
- `doc_id: str`
- `score: float`
- `snippet: str`

### QueryTrace
- `trace_id: str`
- `tenant_id: str`
- `question: str`
- `mode: str` -> `vector` | `hybrid` | `agentic`
- `retrieval_query: str` -> rewritten query if enabled
- `model: str`
- `latency_ms: int`
- `cost_usd: float`
- `created_at: datetime`

### VerificationResult
- `trace_id: str`
- `is_supported: bool`
- `reason: str`
- `created_at: datetime`

### ToolCall (agentic)
- `trace_id: str`
- `tool_name: str`
- `input: str`
- `output: str`
- `created_at: datetime`

### Feedback
- `trace_id: str`
- `rating: int`
- `notes: str`
- `created_at: datetime`

## Metadata fields (common)

Use these keys in `Document.metadata` and `Chunk.metadata` as needed:

- `tenant_id` -> enforced in retrieval filters
- `path` -> file path or URI for traceability
- `section` -> section title or heading (structured chunking)
- `plan_tier` -> e.g., `free` | `pro` | `enterprise`
- `language` -> `en`, `ar`, `fr`, etc.
- `tags` -> list of keywords or labels
- `effective_date` -> policy/version effective date
- `account_id` -> CRM/account scoped retrieval
- `doc_type` -> `policy`, `runbook`, `rfc`, `playbook`, etc.

## API payloads (logical shapes)

### Ingest request
```json
{
  "tenant_id": "tenant-123",
  "source_type": "web",
  "uri": "https://example.com/help",
  "metadata": { "doc_type": "help_center", "language": "en" }
}
```

### Query request
```json
{
  "tenant_id": "tenant-123",
  "question": "What is the on-call policy?",
  "filters": { "plan_tier": "enterprise" },
  "top_k": 8,
  "mode": "hybrid"
}
```

### Query response
```json
{
  "answer": "...",
  "citations": [
    { "chunk_id": "doc:3", "doc_id": "doc", "score": 0.88, "snippet": "..." }
  ],
  "trace_id": "uuid",
  "model": "openai"
}
```

## Indexing strategy
- Vector index keyed by `chunk_id` with metadata filters.
- Optional BM25 corpus stored in JSONL for hybrid retrieval.
- Secondary metadata store for fast document lookup.

## Notes for the expanded case studies
- Customer Support RAG relies on `tenant_id` and `plan_tier`.
- Compliance RAG relies on `effective_date` and strict verification.
- Engineering KB benefits from `path` + `doc_type` metadata for citations.
- Sales Enablement uses `account_id` to scope retrieval.
