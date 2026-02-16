# Project Overview & Architecture

## What is RAG Engine Mini?

RAG Engine Mini is a **production-ready Retrieval-Augmented Generation (RAG) system** designed for educational purposes and real-world deployment. It demonstrates best practices in AI engineering, software architecture, testing, and DevOps.

## Why This Project Matters

### The Problem RAG Solves

Large Language Models (LLMs) like GPT-4 have a critical limitation: **they don't know your data**.

**Example Scenario:**
```
User: "What's our company's Q3 revenue?"
Standard LLM: "I don't have access to your company's financial data."
```

**With RAG:**
```
User: "What's our company's Q3 revenue?"
RAG System: 
  1. Searches internal documents for "Q3 revenue"
  2. Finds: "Q3 Financial Report.pdf"
  3. Retrieves relevant passage: "Q3 revenue was $5.2M..."
  4. LLM generates: "According to the Q3 Financial Report, 
     your company's Q3 revenue was $5.2M, up 15% from Q2."
```

### Key Benefits:

1. **Reduces Hallucinations**: Answers are grounded in actual documents
2. **Provides Citations**: Users can verify sources
3. **Private Data**: Your data stays in your infrastructure
4. **Cost Effective**: Smaller LLMs work with good retrieval
5. **Real-time**: Add new documents, get updated answers immediately

## Project Goals

### Educational Goals:
- Teach production-grade RAG implementation
- Demonstrate clean architecture patterns
- Show comprehensive testing strategies
- Illustrate DevOps best practices

### Technical Goals:
- Hybrid search (vector + keyword)
- Multi-tenant architecture
- Scalable design
- Enterprise security
- Observable systems

### Real-World Applicability:
- Customer support chatbots
- Internal knowledge bases
- Legal document analysis
- Medical literature review
- Research paper synthesis

## System Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
│  (Web UI, Mobile App, API Consumers)                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                               │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │ Auth         │ Rate Limit   │ Validation   │ Logging      │  │
│  │ (JWT/API Key)│ (Redis)      │ (Pydantic)   │ (Structured) │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌─────────────────┐ ┌──────────┐ ┌─────────────────┐
│  PostgreSQL     │ │  Redis   │ │   Qdrant        │
│  (Metadata      │ │  (Cache  │ │  (Vector DB)    │
│   & Auth)       │ │   & Queue)│ │                 │
└─────────────────┘ └──────────┘ └─────────────────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Chunking     │ │ Embedding    │ │ Vector Search│            │
│  │ Service      │ │ Service      │ │ Service      │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Keyword      │ │ Reranking    │ │ LLM Service  │            │
│  │ Search (FTS) │ │ (Cross-Enc)  │ │ (OpenAI/etc) │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Document Ingestion:**
```
1. User uploads document
   ↓
2. File stored (S3/Local)
   ↓
3. Text extracted (PDF/DOCX/etc)
   ↓
4. Text chunked (semantic/fixed)
   ↓
5. Chunks embedded (OpenAI/HF)
   ↓
6. Vectors stored in Qdrant
   ↓
7. Metadata stored in PostgreSQL
```

**Query Flow:**
```
1. User asks question
   ↓
2. Question embedded (same model)
   ↓
3. Parallel searches:
   a. Vector search (Qdrant) - semantic similarity
   b. Keyword search (PostgreSQL FTS) - exact matches
   ↓
4. Results fused (RRF algorithm)
   ↓
5. Top-K results reranked (cross-encoder)
   ↓
6. Context + Question sent to LLM
   ↓
7. Generated answer + sources returned
```

## Technology Stack

### Backend:
- **Python 3.8+**: Core language
- **FastAPI**: Web framework (async, type-safe)
- **SQLAlchemy 2.0**: ORM for PostgreSQL
- **Pydantic**: Data validation
- **Celery**: Background task processing
- **Alembic**: Database migrations

### Data Storage:
- **PostgreSQL**: Primary database (documents, users, metadata)
- **Qdrant**: Vector database (embeddings, similarity search)
- **Redis**: Caching and task queue
- **S3/GCS/Azure**: Object storage for files

### AI/ML:
- **OpenAI API**: GPT-4, GPT-3.5, embeddings
- **HuggingFace**: Open-source models
- **Sentence-Transformers**: Local embeddings
- **Cross-Encoders**: Reranking

### Infrastructure:
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Terraform**: Infrastructure as Code
- **Helm**: Kubernetes package management
- **Prometheus/Grafana**: Monitoring

### Testing:
- **pytest**: Unit and integration testing
- **Locust**: Load testing
- **Bandit**: Security scanning
- **MyPy**: Type checking

## Project Structure

```
rag-engine-mini/
├── src/                          # Source code
│   ├── main.py                   # FastAPI application entry
│   ├── core/                     # Core utilities
│   │   ├── config.py             # Configuration management
│   │   ├── logging.py            # Structured logging
│   │   └── observability.py      # Metrics & tracing
│   ├── api/                      # API layer
│   │   ├── v1/                   # Version 1 routes
│   │   │   ├── routes_ask.py     # Q&A endpoint
│   │   │   ├── routes_documents.py
│   │   │   └── routes_auth.py
│   │   └── middleware/           # Custom middleware
│   ├── application/              # Business logic
│   │   ├── services/             # Core services
│   │   │   ├── chunking.py       # Text chunking
│   │   │   ├── fusion.py         # RRF fusion
│   │   │   └── prompt_builder.py
│   │   └── use_cases/            # Use case implementations
│   │       ├── ask_question_hybrid.py
│   │       └── upload_document.py
│   ├── adapters/                 # External integrations
│   │   ├── llm/                  # LLM providers
│   │   │   ├── openai_llm.py
│   │   │   └── huggingface_llm.py
│   │   ├── persistence/          # Database adapters
│   │   │   └── postgres/
│   │   └── vector/               # Vector stores
│   │       └── qdrant_store.py
│   └── domain/                   # Domain layer
│       ├── entities.py           # Domain entities
│       └── errors.py             # Domain errors
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── performance/              # Load tests
│   └── security/                 # Security tests
├── docs/                         # Documentation
│   ├── learning/                 # Educational materials
│   │   ├── python/               # Python patterns
│   │   ├── security/             # Security guides
│   │   └── testing/              # Testing guides
│   └── _build/                   # Sphinx output
├── sdk/                          # Client SDKs
│   ├── python/                   # Python SDK
│   └── javascript/               # JS SDK
├── scripts/                      # Utility scripts
│   ├── seed_sample_data.py       # Database seeding
│   ├── smoke_test.py             # Health checks
│   └── backup.py                 # Backup operations
├── config/                       # Configuration files
│   ├── kubernetes/               # K8s manifests
│   ├── helm/                     # Helm charts
│   └── terraform/                # Terraform modules
├── notebooks/                    # Jupyter notebooks
│   └── learning/                 # Educational notebooks
├── Dockerfile                    # Container image
├── docker-compose.yml            # Local development
├── Makefile                      # Build automation
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

## Key Design Principles

### 1. Clean Architecture / Hexagonal Architecture

```
┌─────────────────────────────────────────┐
│              API Layer                  │  <- FastAPI routes
│         (Controllers/Routes)            │
├─────────────────────────────────────────┤
│           Application Layer             │  <- Use cases, services
│      (Business Logic Orchestration)     │
├─────────────────────────────────────────┤
│            Domain Layer                 │  <- Entities, rules
│        (Core Business Logic)            │
├─────────────────────────────────────────┤
│           Adapter Layer                 │  <- External integrations
│    (DB, LLM, Vector Store, etc)         │
└─────────────────────────────────────────┘
```

**Benefits:**
- **Testability**: Easy to mock adapters
- **Flexibility**: Swap implementations (e.g., Qdrant → Pinecone)
- **Clarity**: Clear separation of concerns

### 2. Multi-Tenancy

Every piece of data is scoped to a tenant:
```python
# Every query includes tenant_id
SELECT * FROM documents WHERE tenant_id = 'tenant-123'

# Vector search with tenant filter
collection.search(
    vector=embedding,
    filter={"tenant_id": "tenant-123"}  # Isolation
)
```

**Benefits:**
- Data isolation
- Customer-specific scaling
- Single deployment, multiple customers

### 3. Async-First

All I/O operations are async:
```python
async def ask_question(question: str) -> Answer:
    # Concurrent operations
    vector_results, keyword_results = await asyncio.gather(
        vector_search(question),
        keyword_search(question)
    )
    # ... process results
```

**Benefits:**
- Better resource utilization
- Higher throughput
- Non-blocking operations

### 4. Observability

Every layer emits metrics, logs, and traces:
```python
# Structured logging
logger.info("document_uploaded", 
    extra={"document_id": doc_id, "size": size, "tenant": tenant_id})

# Metrics
metrics.counter("documents_uploaded_total", labels={"tenant": tenant_id})
metrics.histogram("upload_duration_seconds")

# Tracing
with tracer.start_as_current_span("document_processing"):
    # ... processing steps
```

**Benefits:**
- Debug issues quickly
- Performance insights
- Alert on anomalies

## Getting Started

### Quick Start (5 minutes):

```bash
# 1. Clone the repository
git clone https://github.com/your-org/rag-engine-mini.git
cd rag-engine-mini

# 2. Install dependencies
make install

# 3. Start infrastructure
docker-compose up -d

# 4. Run migrations
alembic upgrade head

# 5. Seed sample data
python scripts/seed_sample_data.py

# 6. Start the API
make run

# 7. Test it!
curl http://localhost:8000/health
```

### Learning Path Recommendation:

**Week 1: Foundation**
- Read this overview
- Set up environment
- Run the application
- Explore the codebase

**Week 2: Core Implementation**
- Study API layer
- Understand database models
- Learn document processing
- Implement chunking

**Week 3: Testing**
- Write unit tests
- Add integration tests
- Run performance tests
- Security audit

**Week 4: Production**
- Build SDK
- Create documentation
- Deploy to Kubernetes
- Monitor and optimize

## Key Concepts Explained

### Embeddings

**What:** Numerical representations of text
**Analogy:** Like converting words into GPS coordinates
**Example:**
```
"cat" → [0.1, -0.3, 0.8, ...]  (768 or 1536 dimensions)
"dog" → [0.2, -0.2, 0.9, ...]  (similar to cat)
"car" → [-0.5, 0.8, 0.1, ...]  (different from cat)
```

**Why:** Computers understand numbers, not words. Similar concepts = similar vectors.

### Vector Search

**What:** Find similar vectors using distance metrics
**Analogy:** Like finding nearest neighbors on a map
**Algorithm:** HNSW (Hierarchical Navigable Small World) - approximate nearest neighbors
**Trade-off:** Speed vs accuracy (99% accuracy, 100x faster than exact)

### Hybrid Search

**Problem:**
- Vector search: Good for semantic similarity
- Keyword search: Good for exact matches

**Solution:** Combine both with RRF (Reciprocal Rank Fusion):
```python
# Vector ranks: DocA=1, DocB=3, DocC=5
# Keyword ranks: DocB=1, DocA=4, DocC=6

# RRF Score = Σ 1/(k + rank)
# k=60 (smoothing constant)

DocA: 1/(60+1) + 1/(60+4) = 0.032
DocB: 1/(60+3) + 1/(60+1) = 0.032  <- Best!
DocC: 1/(60+5) + 1/(60+6) = 0.015
```

### Reranking

**First Pass (Fast):** Retrieve 100 candidates using bi-encoder (one embedding per document)
**Second Pass (Accurate):** Re-rank top 20 using cross-encoder (joint embedding of query + document)

**Why two passes?**
- Bi-encoder: Fast (1ms), less accurate
- Cross-encoder: Slow (50ms), very accurate
- Combined: Fast enough, very accurate

## Common Questions

**Q: Can I use this in production?**
A: Yes! It includes enterprise features like multi-tenancy, security, monitoring, and scalability.

**Q: Do I need to pay for OpenAI?**
A: Yes for GPT-4, but you can use free alternatives (HuggingFace, Ollama for local models).

**Q: How much does it cost to run?**
A: ~$50-200/month for small deployments. Main costs: LLM API, hosting, vector DB.

**Q: Can it handle millions of documents?**
A: Yes, with proper scaling. PostgreSQL and Qdrant both scale horizontally.

**Q: Is my data private?**
A: Yes! Everything runs in your infrastructure. No data sent to third parties (if using local models).

## Next Steps

1. **Continue to Module 2:** Environment Setup
2. **Clone the repo** and explore
3. **Join the community** for support
4. **Build something** with what you learn!

---

**Ready to dive deeper?** Continue to the Environment Setup notebook!
