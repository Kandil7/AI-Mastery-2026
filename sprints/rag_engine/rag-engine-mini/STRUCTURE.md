# RAG Engine Mini - Project Structure (Stage 1)

> **Production-Ready RAG Starter Template** with Clean Architecture, Hybrid Search (Vector + Keyword), Reranking, and Multi-Tenant Support.

---

## 📁 Complete File Tree / شجرة الملفات الكاملة

```
rag-engine-mini/
├── 📄 README.md                           # Project overview, quickstart, E2E examples (AR+EN)
├── 📄 pyproject.toml                      # Python dependencies + project metadata (Poetry/uv compatible)
├── 📄 .env.example                        # Environment variables template with all configuration options
├── 📄 Makefile                            # Dev commands: run, worker, test, format, lint, migrate, seed
├── 📄 alembic.ini                         # Alembic migrations configuration file
├── 📄 .gitignore                          # Git ignore patterns for Python/Docker/IDE files
│
├── 📂 src/                                # Main application source code
│   ├── 📄 __init__.py                     # Package marker
│   ├── 📄 main.py                         # FastAPI app factory + ASGI entry point
│   │
│   ├── 📂 core/                           # Core infrastructure & configuration
│   │   ├── 📄 __init__.py                 # Package marker
│   │   ├── 📄 config.py                   # Pydantic Settings: all env-based configuration
│   │   ├── 📄 logging.py                  # Structured logging setup (structlog)
│   │   ├── 📄 observability.py            # Metrics & tracing wiring (Prometheus-ready)
│   │   └── 📄 bootstrap.py                # DI container: wires all Ports ↔ Adapters
│   │
│   ├── 📂 domain/                         # Pure domain layer (no external dependencies)
│   │   ├── 📄 __init__.py                 # Package marker
│   │   ├── 📄 entities.py                 # Domain entities: TenantId, DocumentId, Chunk, Answer, etc.
│   │   └── 📄 errors.py                   # Domain-specific exceptions
│   │
│   ├── 📂 application/                    # Application layer: Use Cases + Ports + Pure Services
│   │   ├── 📄 __init__.py                 # Package marker
│   │   │
│   │   ├── 📂 ports/                      # Interfaces (Dependency Inversion Principle)
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   ├── 📄 llm.py                  # LLMPort: generate(prompt) → str
│   │   │   ├── 📄 embeddings.py           # EmbeddingsPort: embed_one, embed_many
│   │   │   ├── 📄 vector_store.py         # VectorStorePort: upsert, search_scored
│   │   │   ├── 📄 keyword_store.py        # KeywordStorePort: FTS search with doc-filter
│   │   │   ├── 📄 document_repo.py        # DocumentRepoPort: CRUD + status management
│   │   │   ├── 📄 document_idempotency.py # Idempotency port: file hash lookup/create
│   │   │   ├── 📄 document_reader.py      # DocumentReaderPort: get stored file metadata
│   │   │   ├── 📄 chunk_repo.py           # ChunkRepoPort: chunk_store upsert + doc mapping
│   │   │   ├── 📄 chunk_text_reader.py    # ChunkTextReaderPort: hydrate text by IDs
│   │   │   ├── 📄 chat_repo.py            # ChatRepoPort: sessions + turns persistence
│   │   │   ├── 📄 cache.py                # CachePort: get/set JSON with TTL (Redis)
│   │   │   ├── 📄 file_store.py           # FileStorePort: save uploaded files
│   │   │   ├── 📄 task_queue.py           # TaskQueuePort: enqueue background indexing
│   │   │   └── 📄 reranker.py             # RerankerPort: rerank(query, chunks) → top_n
│   │   │
│   │   ├── 📂 use_cases/                  # Business orchestration (one file per use case)
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   ├── 📄 upload_document.py      # UploadDocumentUseCase: validate + store + enqueue
│   │   │   ├── 📄 index_document.py       # IndexDocumentUseCase: extract → chunk → embed → upsert
│   │   │   ├── 📄 ask_question.py         # AskQuestionUseCase: simple vector RAG
│   │   │   └── 📄 ask_question_hybrid.py  # AskQuestionHybridUseCase: vector + keyword + RRF + rerank
│   │   │
│   │   └── 📂 services/                   # Pure domain services (no I/O)
│   │       ├── 📄 __init__.py             # Package marker
│   │       ├── 📄 text_extraction.py      # TextExtractor Protocol + ExtractedText model
│   │       ├── 📄 chunking.py             # Token-aware chunking with overlap
│   │       ├── 📄 prompt_builder.py       # RAG prompt construction with guardrails
│   │       ├── 📄 fusion.py               # RRF fusion for hybrid retrieval
│   │       ├── 📄 scoring.py              # ScoredChunk dataclass
│   │       ├── 📄 embedding_cache.py      # CachedEmbeddings: wraps embeddings + Redis
│   │       └── 📄 hydrate.py              # hydrate_chunk_texts: fill text from DB
│   │
│   ├── 📂 adapters/                       # External implementations of Ports
│   │   ├── 📄 __init__.py                 # Package marker
│   │   │
│   │   ├── 📂 llm/                        # LLM provider adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   ├── 📄 openai_llm.py           # OpenAI ChatCompletion adapter
│   │   │   └── 📄 ollama_llm.py           # Ollama local LLM adapter
│   │   │
│   │   ├── 📂 embeddings/                 # Embedding provider adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   ├── 📄 openai_embeddings.py    # OpenAI text-embedding adapter
│   │   │   └── 📄 local_embeddings.py     # SentenceTransformers local adapter
│   │   │
│   │   ├── 📂 vector/                     # Vector store adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   └── 📄 qdrant_store.py         # Qdrant vector store (minimal payload)
│   │   │
│   │   ├── 📂 rerank/                     # Reranker adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   ├── 📄 cross_encoder.py        # CrossEncoder local reranker (SentenceTransformers)
│   │   │   ├── 📄 llm_reranker.py         # LLM-based reranker (fallback)
│   │   │   └── 📄 noop_reranker.py        # No-op reranker (passthrough)
│   │   │
│   │   ├── 📂 extraction/                 # Text extraction adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   └── 📄 default_extractor.py    # PDF/DOCX/TXT extraction (pypdf, python-docx)
│   │   │
│   │   ├── 📂 cache/                      # Cache adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   └── 📄 redis_cache.py          # Redis JSON cache adapter
│   │   │
│   │   ├── 📂 filestore/                  # File storage adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   ├── 📄 local_store.py          # Local filesystem storage
│   │   │   └── 📄 s3_store.py             # S3-compatible storage (stub)
│   │   │
│   │   ├── 📂 queue/                      # Task queue adapters
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   └── 📄 celery_queue.py         # Celery task queue adapter
│   │   │
│   │   └── 📂 persistence/                # Database persistence layer
│   │       ├── 📄 __init__.py             # Package marker
│   │       │
│   │       └── 📂 postgres/               # PostgreSQL adapters
│   │           ├── 📄 __init__.py         # Package marker
│   │           ├── 📄 db.py               # SQLAlchemy engine, Base, SessionLocal
│   │           ├── 📄 models.py           # User + Document ORM models
│   │           ├── 📄 models_chunk_store.py # ChunkStoreRow + DocumentChunkRow ORM models
│   │           ├── 📄 models_chat.py      # ChatSession + ChatTurn ORM models
│   │           ├── 📄 repo_users.py       # UserLookupRepo: API key → user_id
│   │           ├── 📄 repo_documents.py   # PostgresDocumentRepo: CRUD + status
│   │           ├── 📄 repo_documents_idempotency.py  # File hash idempotency repo
│   │           ├── 📄 repo_chunks.py      # PostgresChunkDedupRepo: chunk_store + mapping
│   │           ├── 📄 repo_chat.py        # PostgresChatRepo: sessions + turns
│   │           ├── 📄 keyword_store.py    # PostgresKeywordStore: FTS with tsvector
│   │           ├── 📄 chunk_text_reader.py # PostgresChunkTextReader: text hydration
│   │           │
│   │           └── 📂 migrations/         # Alembic migrations directory
│   │               ├── 📄 env.py          # Alembic environment configuration
│   │               ├── 📄 script.py.mako  # Migration script template
│   │               └── 📂 versions/       # Migration version files
│   │                   ├── 📄 001_create_users_documents.py  # Users + Documents tables
│   │                   ├── 📄 002_add_chunk_store.py         # chunk_store + document_chunks + tsv
│   │                   ├── 📄 003_add_chat_tables.py         # chat_sessions + chat_turns
│   │                   └── 📄 004_add_document_hash.py       # file_sha256 idempotency column
│   │
│   ├── 📂 api/                            # FastAPI routes (thin controllers)
│   │   ├── 📄 __init__.py                 # Package marker
│   │   │
│   │   ├── 📂 v1/                         # API version 1
│   │   │   ├── 📄 __init__.py             # Package marker
│   │   │   ├── 📄 deps.py                 # Request dependencies: auth, tenant extraction
│   │   │   ├── 📄 routes_health.py        # Health check endpoints
│   │   │   ├── 📄 routes_documents.py     # Document upload/list/status endpoints
│   │   │   └── 📄 routes_queries.py       # Ask endpoints: /ask, /ask-hybrid
│   │   │
│   │   └── 📄 schemas.py                  # Pydantic request/response DTOs
│   │
│   └── 📂 workers/                        # Celery background workers
│       ├── 📄 __init__.py                 # Package marker
│       ├── 📄 celery_app.py               # Celery app configuration
│       └── 📄 tasks.py                    # index_document task with full pipeline
│
├── 📂 tests/                              # Test suite
│   ├── 📄 __init__.py                     # Package marker
│   ├── 📄 conftest.py                     # Pytest fixtures + test database setup
│   │
│   ├── 📂 unit/                           # Unit tests (isolated, no I/O)
│   │   ├── 📄 __init__.py                 # Package marker
│   │   ├── 📄 test_chunking.py            # Chunking service tests
│   │   ├── 📄 test_prompt_builder.py      # Prompt builder tests
│   │   ├── 📄 test_fusion.py              # RRF fusion tests
│   │   └── 📄 test_entities.py            # Domain entities tests
│   │
│   └── 📂 integration/                    # Integration tests (with DB/services)
│       ├── 📄 __init__.py                 # Package marker
│       ├── 📄 test_upload_flow.py         # Upload → index flow integration test
│       └── 📄 test_ask_flow.py            # Ask hybrid flow integration test
│
├── 📂 scripts/                            # Utility scripts
│   ├── 📄 seed_user.py                    # Create demo user with API key
│   ├── 📄 benchmark.py                    # Performance benchmarking script
│   └── 📄 eval_retrieval.py               # Retrieval quality evaluation (golden Q&A)
│
├── 📂 docker/                             # Docker configuration
│   ├── 📄 Dockerfile                      # Production-ready multi-stage Dockerfile
│   ├── 📄 docker-compose.yml              # Full stack: api + worker + postgres + redis + qdrant
│   ├── 📄 docker-compose.dev.yml          # Development override (hot reload)
│   └── 📄 .dockerignore                   # Docker build ignore patterns
│
├── 📂 docs/                               # Documentation (Arabic + English)
│   ├── 📄 architecture.md                 # Detailed architecture + text diagrams
│   ├── 📄 modules.md                      # Module-by-module explanation
│   ├── 📄 workflows.md                    # Key workflows: upload, index, ask-hybrid
│   └── 📄 contributing.md                 # Coding standards, naming, git workflow
│
└── 📂 notebooks/                          # Educational Jupyter notebooks
    ├── 📄 01_intro_and_setup.ipynb        # Project intro, setup, architecture overview
    ├── 📄 02_end_to_end_rag.ipynb         # E2E RAG flow walkthrough
    └── 📄 03_hybrid_search_and_rerank.ipynb # Hybrid retrieval deep dive
```

---

## 📋 Key Features Summary / ملخص المميزات الرئيسية

| Feature | Description (EN) | الوصف (AR) |
|---------|------------------|------------|
| **Clean Architecture** | Domain/Application/Adapters separation | فصل المجال/التطبيق/المحولات |
| **SOLID Principles** | Ports & Adapters, Dependency Injection | المنافذ والمحولات، حقن التبعيات |
| **Hybrid Search** | Vector (Qdrant) + Keyword (Postgres FTS) | بحث متجه + بحث نصي |
| **RRF Fusion** | Reciprocal Rank Fusion for result merging | دمج النتائج بخوارزمية RRF |
| **Cross-Encoder Rerank** | Local sentence-transformers reranker | إعادة الترتيب بمشفر متقاطع محلي |
| **Multi-Tenant** | user_id isolation everywhere | عزل البيانات لكل مستخدم |
| **Idempotency** | SHA256 file hash prevents re-indexing | تجزئة الملفات تمنع الفهرسة المكررة |
| **Chunk Dedup** | Per-tenant chunk deduplication | إزالة تكرار القطع لكل مستأجر |
| **Minimal Vector Payload** | Text stored in Postgres, not Qdrant | النص في Postgres، ليس في Qdrant |
| **Batch Embeddings** | Cost-effective batch embedding calls | استدعاءات تضمين دفعية موفرة |
| **Document Filtering** | ChatPDF mode: search within single doc | وضع ChatPDF: البحث داخل مستند واحد |
| **Observability Ready** | Structured logs, metrics-friendly | سجلات منظمة، جاهز للقياسات |

---

## ⏭️ Stage 2 Preview

**After approval**, I will create `README.md` with:
- Project overview / نظرة عامة
- Feature highlights / المميزات الرئيسية  
- Quickstart guide / دليل البدء السريع
- E2E example (upload + ask-hybrid) / مثال متكامل
- Architecture summary / ملخص المعمارية
- Troubleshooting / استكشاف الأخطاء

---

> **هل الهيكل يناسب متطلباتك؟ / Does this structure meet your requirements?**
> 
> Reply with **"proceed"** or **"كمّل"** to start Stage 2 (README.md).
