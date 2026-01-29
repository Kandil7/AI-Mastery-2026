# 📚 Module Guide

> Detailed explanation of each module in the RAG Engine Mini project.

---

## 📁 src/core/

### `config.py`
**Purpose:** Centralized configuration using Pydantic Settings.

```python
from src.core import settings

# Access any config value
print(settings.openai_api_key)
print(settings.chunk_max_tokens)
```

**Key Points:**
- All settings from environment variables
- Type-safe with validation at startup
- Cached singleton via `@lru_cache`

### `logging.py`
**Purpose:** Structured logging with structlog.

```python
from src.core.logging import get_logger, bind_context

log = get_logger(__name__)
bind_context(request_id="abc-123")
log.info("processing", chunk_count=42)
```

### `bootstrap.py`
**Purpose:** Dependency Injection container.

```python
from src.core.bootstrap import get_container

container = get_container()
llm = container["llm"]
use_case = container["ask_hybrid_use_case"]
```

---

## 📁 src/domain/

### `entities.py`
**Purpose:** Pure domain objects with no external dependencies.

| Entity | Description |
|--------|-------------|
| `TenantId` | User/tenant identifier (value object) |
| `DocumentId` | Document identifier |
| `Chunk` | Text chunk with metadata |
| `Answer` | Generated answer with sources |
| `StoredFile` | File storage metadata |
| `DocumentStatus` | Document processing status |

### `errors.py`
**Purpose:** Domain-specific exceptions.

| Error | When Raised |
|-------|-------------|
| `DocumentNotFoundError` | Document lookup fails |
| `FileTooLargeError` | Upload exceeds limit |
| `TextExtractionError` | PDF/DOCX parsing fails |
| `LLMError` | OpenAI/Ollama fails |

---

## 📁 src/application/ports/

### Interface Pattern
All ports use Python `Protocol` for structural subtyping:

```python
class LLMPort(Protocol):
    def generate(self, prompt: str, ...) -> str: ...
```

### Ports List

| Port | Methods |
|------|---------|
| `LLMPort` | `generate(prompt) → str` |
| `EmbeddingsPort` | `embed_one(text)`, `embed_many(texts)` |
| `VectorStorePort` | `upsert_points()`, `search_scored()` |
| `KeywordStorePort` | `search(query, ...)` |
| `RerankerPort` | `rerank(query, chunks, top_n)` |
| `CachePort` | `get_json()`, `set_json()` |

---

## 📁 src/application/services/

### `chunking.py`
**Purpose:** Token-aware text splitting.

```python
from src.application.services.chunking import chunk_text_token_aware

chunks = chunk_text_token_aware(
    text,
    ChunkSpec(max_tokens=512, overlap_tokens=50)
)
```

### `fusion.py`
**Purpose:** Reciprocal Rank Fusion for merging search results.

```python
from src.application.services.fusion import rrf_fusion

fused = rrf_fusion(
    vector_hits=vec_results,
    keyword_hits=kw_results,
    out_limit=40
)
```

**Algorithm:**
- RRF_score = Σ 1/(k + rank)
- Combines without score calibration
- Robust to different score distributions

### `embedding_cache.py`
**Purpose:** Redis-backed embedding cache.

```python
cached_emb = CachedEmbeddings(
    embeddings=openai_emb,
    cache=redis_cache,
    ttl_seconds=604800  # 7 days
)

# First call: API request + cache write
vector = cached_emb.embed_one("Hello")

# Second call: instant cache hit
vector = cached_emb.embed_one("Hello")
```

### `hydrate.py`
**Purpose:** Fill chunk texts from database.

```python
hydrated = hydrate_chunk_texts(
    tenant_id=tenant,
    chunks=vector_results,  # Empty text
    reader=postgres_reader
)
# Now chunks have text
```

---

## 📁 src/application/use_cases/

### `upload_document.py`

**Flow:**
1. Calculate SHA256 hash
2. Check for existing (idempotency)
3. Store file
4. Create document record
5. Enqueue indexing task

```python
result = await upload_use_case.execute(
    UploadDocumentRequest(
        tenant_id="user123",
        filename="report.pdf",
        content_type="application/pdf",
        data=file_bytes,
    )
)
# Returns: {document_id: "...", status: "queued"}
```

### `ask_question_hybrid.py`

**Flow:**
1. Embed question (cached)
2. Vector search
3. Hydrate texts
4. Keyword search
5. RRF fusion
6. Cross-Encoder reranking
7. Build prompt
8. LLM generation

```python
answer = use_case.execute(
    AskHybridRequest(
        tenant_id="user123",
        question="What is the main topic?",
        document_id=None,  # Or specific doc for ChatPDF mode
        k_vec=30,
        k_kw=30,
        rerank_top_n=8,
    )
)
# Returns: Answer(text="...", sources=["chunk1", "chunk2"])
```

---

## 📁 src/adapters/

### `llm/openai_llm.py`
OpenAI ChatCompletion adapter.

### `embeddings/openai_embeddings.py`
OpenAI Embeddings adapter with batch support.

### `vector/qdrant_store.py`
Qdrant adapter with minimal payload design.

### `rerank/cross_encoder.py`
Local Cross-Encoder using SentenceTransformers.

### `cache/redis_cache.py`
Redis JSON cache with TTL support.

### `persistence/postgres/`
PostgreSQL repositories:
- `repo_documents.py` - Document CRUD
- `repo_chunks.py` - Chunk deduplication
- `keyword_store.py` - Full-text search
- `chunk_text_reader.py` - Text hydration

---

## 📁 src/api/v1/

### `routes_health.py`
```
GET /health         → Basic health check
GET /health/ready   → Readiness probe
```

### `routes_documents.py`
```
POST /api/v1/documents/upload       → Upload document
GET  /api/v1/documents/{id}/status  → Check status
GET  /api/v1/documents              → List documents
```

### `routes_queries.py`
```
POST /api/v1/queries/ask-hybrid     → Hybrid RAG question
POST /api/v1/queries/ask            → Alias for ask-hybrid
```

---

## 📁 src/workers/

### `celery_app.py`
Celery application configuration.

### `tasks.py`
**`index_document`** task:
1. Extract text
2. Chunk
3. Deduplicate
4. Batch embed
5. Store chunks
6. Upsert vectors
7. Update status
