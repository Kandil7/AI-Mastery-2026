# CHANGELOG

All notable changes to RAG Engine Mini will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-29

### Added

#### Core Architecture
- Clean Architecture with domain/application/adapters/api layers
- Ports and Adapters pattern for dependency inversion
- Multi-tenant support with API key authentication
- Dependency injection container (`bootstrap.py`)

#### Domain Layer
- Core entities: `TenantId`, `DocumentId`, `Chunk`, `Answer`
- Value objects: `ChunkSpec`, `StoredFile`, `ExtractedText`
- Domain errors: `DocumentNotFoundError`, `FileTooLargeError`, etc.

#### Application Layer
- 14 ports (interfaces) for all external dependencies
- Pure services: chunking, fusion, prompt building, caching
- Use cases: `UploadDocumentUseCase`, `AskQuestionHybridUseCase`

#### Adapters
- LLM: OpenAI, Ollama
- Embeddings: OpenAI, Local (SentenceTransformers)
- Vector Store: Qdrant
- Keyword Store: PostgreSQL Full-Text Search
- Reranking: Cross-Encoder, LLM-based, No-op
- Cache: Redis
- File Storage: Local filesystem
- Text Extraction: PDF, DOCX, TXT
- Task Queue: Celery

#### API Layer
- FastAPI application with versioned routes
- Health check endpoints
- Document upload and management
- Hybrid RAG query endpoint
- Chat session management

#### Database
- PostgreSQL schema with 5 tables
- Alembic migrations (3 versions)
- SQLAlchemy ORM models
- Full-text search with GENERATED tsvector

#### Workers
- Celery application configuration
- Document indexing task with retry logic

#### Infrastructure
- Docker Compose for local development
- Production Dockerfile
- Makefile with common commands

#### Documentation
- Comprehensive README
- Architecture documentation with diagrams
- API reference
- Configuration guide
- Deployment guide
- Developer guide
- Prompt engineering guide
- Troubleshooting guide

#### Testing
- Pytest configuration
- Unit tests for core services
- Integration tests for API

#### Educational
- Jupyter notebooks for learning
- Bilingual documentation (English + Arabic)

### Technical Decisions

- Token-aware chunking using tiktoken
- RRF fusion for hybrid search (no score calibration needed)
- Minimal payload in Qdrant (text in Postgres only)
- GENERATED tsvector for automatic FTS updates
- Chunk deduplication via SHA256 hash
- Idempotent uploads via file hash
- 7-day TTL for embedding cache
- Local Cross-Encoder for cost-effective reranking

---

## Future Roadmap

### [0.2.0] - Planned
- Streaming responses
- Document deletion cascade
- Batch upload API
- Webhook notifications
- Rate limiting

### [0.3.0] - Planned
- Multi-modal support (images)
- Evaluation framework
- A/B testing for prompts
- Custom extractors API
