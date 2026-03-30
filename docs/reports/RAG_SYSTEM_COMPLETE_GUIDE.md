# Complete RAG System Implementation Guide 2026

## Arabic Islamic Literature RAG System

Production-grade Retrieval-Augmented Generation system for Arabic Islamic literature corpus.

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Component Guide](#component-guide)
6. [Dataset Integration](#dataset-integration)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)

---

## System Overview

### What This System Does

This RAG (Retrieval-Augmented Generation) system enables intelligent querying of the Arabic Islamic literature corpus, including:

- **8,425+ extracted books** from Islamic scholarship
- **Multiple domains**: Tafsir, Hadith, Fiqh, Aqeedah, Arabic Language, History
- **Multi-source ingestion**: Files, APIs, databases, webhooks
- **Advanced retrieval**: Hybrid search (semantic + BM25), reranking
- **Domain specialists**: Scholar agents for each Islamic knowledge domain

### Key Features (2026 Production Standards)

| Feature | Description |
|---------|-------------|
| **Multi-source Ingestion** | Files (PDF, DOCX, TXT, MD), APIs, Databases, Webhooks |
| **Advanced Chunking** | Fixed, Recursive, Semantic, Late, Agentic, Islamic-specific |
| **Multi-model Embeddings** | Sentence Transformers, OpenAI, Cohere with caching |
| **Vector Databases** | Qdrant (production), ChromaDB (dev), In-memory (testing) |
| **Hybrid Retrieval** | Semantic + BM25 with Reciprocal Rank Fusion |
| **Cross-Encoder Reranking** | BGE-Reranker, Cohere Rerank API |
| **Query Transformation** | Rewriting, Decomposition, HyDE, Step-back |
| **Multi-provider LLM** | OpenAI, Anthropic, Ollama, HuggingFace |
| **Response Guardrails** | Hallucination detection, Grounding checks |
| **Evaluation Pipeline** | Islamic-specific metrics, continuous evaluation |

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARABIC ISLAMIC RAG PIPELINE 2026                     │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ DATA SOURCES │     │  INGESTION   │     │   PROCESSING │
│              │     │   PIPELINE   │     │   PIPELINE   │
│ • 8,425 Books│────▶│              │────▶│              │
│ • Arabic Web │     │ • Multi-     │     │ • Chunking   │
│ • Hadith DB  │     │   source     │     │ • Embedding  │
│ • APIs       │     │ • Incremental│     │ • Indexing   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────────────────────┐
                     │      VECTOR DATABASE         │
                     │   (Qdrant / ChromaDB)        │
                     │                              │
                     │  ┌────────┐  ┌────────┐     │
                     │  │ Chunks │  │Metadata│     │
                     │  │ +      │  │ +      │     │
                     │  │Vectors │  │Filters │     │
                     │  └────────┘  └────────┘     │
                     └──────────────────────────────┘
                                    │
                                    │ ◀──── Query Flow ────▶
                                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   RESPONSE   │     │  GENERATION  │     │   RETRIEVAL  │
│   & STREAM   │◀────│   ENHANCEMENT│◀────│   & RERANK   │
│              │     │              │     │              │
│ • Streaming  │     │ • Prompt     │     │ • Hybrid     │
│ • Citations  │     │ • Context    │     │ • Reranking  │
│ • Feedback   │     │ • Guardrails │     │ • Filtering  │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│      EVALUATION & MONITORING LOOP        │
│                                          │
│  • Retrieval Quality (Precision@K)      │
│  • Answer Quality (Faithfulness)        │
│  • User Feedback (Thumbs up/down)       │
│  • Cost & Latency Tracking              │
└──────────────────────────────────────────┘
```

### Component Flow

```python
# Complete RAG Pipeline Flow

class RAGPipeline:
    def __init__(self):
        # Phase 1: Ingestion
        self.ingestion = MultiSourceIngestionPipeline()
        
        # Phase 2: Processing
        self.chunker = AdvancedChunker(strategy="islamic")
        self.embeddings = EmbeddingPipeline(provider="sentence_transformers")
        
        # Phase 3: Storage
        self.vector_db = VectorStore(config)
        self.bm25 = BM25Index()
        
        # Phase 4: Query Transformation
        self.query_transformer = QueryTransformer()
        
        # Phase 5: Retrieval
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        
        # Phase 6: Generation
        self.llm = LLMClient(provider="openai")
        self.generator = RAGGenerator()
        
        # Phase 7: Evaluation
        self.evaluator = IslamicRAGEvaluator()

    async def query(self, question: str):
        # Transform query
        transformed = await self.query_transformer.transform(question)
        
        # Retrieve
        candidates = await self.retriever.search(transformed.rewritten_query, top_k=50)
        
        # Rerank
        reranked = self.reranker.rerank(candidates, question, top_k=5)
        
        # Generate
        response = await self.generator.generate(question, reranked)
        
        # Evaluate
        metrics = await self.evaluator.evaluate(question, response, reranked)
        
        return response
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for faster embedding generation)
- 16GB+ RAM recommended for large dataset

### Step 1: Clone and Setup

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026
```

### Step 2: Install Dependencies

```bash
# Core dependencies
pip install numpy pydantic python-multipart

# Embeddings & Vector Search
pip install sentence-transformers torch torchvision

# For Qdrant (production)
pip install qdrant-client

# For ChromaDB (development)
pip install chromadb

# For OpenAI
pip install openai

# For Anthropic
pip install anthropic

# For document parsing
pip install pdfplumber python-docx beautifulsoup4

# For BM25
pip install rank-bm25

# For reranking
pip install sentence-transformers  # Includes CrossEncoder

# For evaluation
pip install scikit-learn

# For API
pip install fastapi uvicorn python-multipart

# For async HTTP
pip install aiohttp

# For Arabic text processing
pip install arabic-reshaper python-bidi
```

### Step 3: Environment Variables

Create `.env` file:

```bash
# OpenAI (for LLM and embeddings)
OPENAI_API_KEY=sk-...

# Anthropic (alternative LLM)
ANTHROPIC_API_KEY=sk-ant-...

# Cohere (for reranking)
COHERE_API_KEY=...

# Qdrant (vector database)
QDRANT_API_KEY=...  # Optional for local
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## Quick Start

### Basic Usage

```python
import asyncio
from rag_system import create_islamic_rag

async def main():
    # Create RAG system
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Basic query
    result = await rag.query("ما هو التوحيد في الإسلام؟")
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    
    # Domain-specific query
    result = await rag.query_as_scholar("fiqh", "ما حكم الزكاة؟")
    print(f"Fiqh Answer: {result['answer']}")
    
    # Comparative query
    result = await rag.compare_madhhabs("ما حكم الصيام؟")
    print(f"Comparison: {result['madhhab_results']}")

asyncio.run(main())
```

### Indexing Documents

```python
from rag_system import create_islamic_rag

async def index_documents():
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Index all documents (may take hours)
    await rag.index_documents()
    
    # Or index with limit for testing
    await rag.index_documents(limit=100)
    
    # Or index specific categories
    await rag.index_documents(categories=["التفسير", "الحديث"])

asyncio.run(index_documents())
```

---

## Component Guide

### 1. Data Ingestion Pipeline

**File**: `rag_system/src/data/multi_source_ingestion.py`

```python
from rag_system.src.data.multi_source_ingestion import (
    MultiSourceIngestionPipeline,
    DataSource,
    DataSourceType,
    ConnectorType,
    create_file_source,
)

# Create pipeline
pipeline = MultiSourceIngestionPipeline(
    update_strategy="incremental",  # or "full"
    batch_size=100,
)

# Add file source
books_source = create_file_source(
    name="extracted_books",
    path="datasets/extracted_books",
    priority=10,
)
pipeline.add_source(books_source)

# Add API source
api_source = create_api_source(
    name="islamic_api",
    url="https://api.example.com/books",
    api_key="...",
)
pipeline.add_source(api_source)

# Ingest
results = await pipeline.ingest_all_sources()
```

### 2. Advanced Chunking

**File**: `rag_system/src/processing/advanced_chunker.py`

```python
from rag_system.src.processing.advanced_chunker import (
    create_chunker,
    ChunkingStrategy,
)

# Create chunker with strategy
chunker = create_chunker(
    strategy="islamic",  # fixed, recursive, semantic, late, agentic, islamic
    chunk_size=512,
    chunk_overlap=50,
)

# Chunk document
chunks = chunker.chunk({
    "id": "book_001",
    "content": text,
    "metadata": {"category": "tafsir"},
})

# Get recommended settings for category
from rag_system.src.processing.advanced_chunker import get_recommended_chunking

config = get_recommended_chunking("quran")
# Returns: {"strategy": "islamic", "chunk_size": 384, ...}
```

### 3. Embedding Pipeline

**File**: `rag_system/src/processing/embedding_pipeline.py`

```python
from rag_system.src.processing.embedding_pipeline import (
    create_embedding_pipeline,
    EmbeddingProvider,
)

# Create pipeline
pipeline = create_embedding_pipeline(
    provider="sentence_transformers",  # or "openai", "cohere"
    cache_dir="rag_system/data/embedding_cache",
)

# Generate embeddings
result = await pipeline.embed_texts([
    "ما هو التوحيد؟",
    "Explain Tawhid",
])

print(f"Embeddings shape: {result.embeddings.shape}")
print(f"Cost: ${result.cost_usd}")
print(f"Cache hits: {result.cache_hits}")
```

### 4. Vector Store

**File**: `rag_system/src/retrieval/vector_store.py`

```python
from rag_system.src.retrieval.vector_store import (
    create_vector_store,
    VectorStoreConfig,
)

# Create store
store = create_vector_store(
    store_type="qdrant",  # or "chroma", "memory"
    collection_name="arabic_islamic_literature",
    vector_size=768,
    persist_directory="rag_system/data/vector_db",
)

# Add vectors
store.add_vectors(
    ids=["chunk_001", "chunk_002"],
    vectors=[emb1, emb2],
    payloads=[
        {"book_title": "...", "content": "..."},
        {"book_title": "...", "content": "..."},
    ],
)

# Search
results = store.search(
    query_vector=query_emb,
    top_k=5,
    filters={"category": "tafsir"},
)
```

### 5. Query Transformer

**File**: `rag_system/src/retrieval/query_transformer.py`

```python
from rag_system.src.retrieval.query_transformer import (
    create_query_transformer,
)

transformer = create_query_transformer(
    enable_hyde=True,
    enable_decomposition=True,
    enable_step_back=True,
)

# Transform query
result = await transformer.transform("ما حكم الزكاة وكيفية حسابها؟")

print(f"Type: {result.query_type.value}")  # multi_hop
print(f"Rewritten: {result.rewritten_query}")
print(f"Sub-queries: {result.sub_queries}")
print(f"HyDE doc: {result.hypothetical_document[:100]}")
```

### 6. Hybrid Retriever

**File**: `rag_system/src/retrieval/hybrid_retriever.py`

```python
from rag_system.src.retrieval.hybrid_retriever import (
    HybridRetriever,
    BM25Index,
    Reranker,
)

# Create retriever
retriever = HybridRetriever(
    vector_store=store,
    embedding_model=embedding_pipeline,
    bm25=bm25_index,
    weights={"semantic": 0.6, "bm25": 0.4},
)

# Search
results = await retriever.search(
    query="ما هو التوحيد؟",
    top_k=50,
)

# Rerank
reranker = Reranker(model_name="BAAI/bge-reranker-base")
reranked = reranker.rerank(
    query="ما هو التوحيد؟",
    candidates=results,
    top_k=5,
)
```

### 7. LLM Generation

**File**: `rag_system/src/generation/generator.py`

```python
from rag_system.src.generation.generator import (
    LLMClient,
    LLMProvider,
    RAGGenerator,
)

# Create LLM client
llm = LLMClient(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    temperature=0.3,
    max_tokens=2000,
)

# Create generator
generator = RAGGenerator(
    llm_client=llm,
    include_citations=True,
    max_context_chunks=5,
)

# Generate
result = await generator.generate(
    query="ما هو التوحيد؟",
    retrieved_chunks=chunks,
)

print(f"Answer: {result.answer}")
print(f"Citations: {result.citations}")
```

---

## Dataset Integration

### Available Datasets

```
datasets/
├── extracted_books/          # 8,425 Islamic books
│   ├── 1_الفواكه_العذاب.txt
│   ├── ...
│   └── 10999_*.txt
├── arabic_web/
│   └── arabicweb24_clean.txt
├── Sanadset 368K Data on Hadith Narrators/
│   └── Sanadset 368K Data on Hadith Narrators/
├── metadata/
│   ├── books.json
│   ├── authors.json
│   ├── categories.json
│   └── guid_index.json
└── system_book_datasets/
    ├── book/
    ├── service/
    ├── store/
    ├── update/
    └── user/
```

### Metadata Schema

```json
{
  "books": [
    {
      "id": 1,
      "title": "الفواكه العذاب في الرد على من لم يحكم السنة والكتاب",
      "author_str": "أحمد بن محمد بن إبراهيم الفزاري",
      "cat_name": "العقيدة",
      "date": 1234
    }
  ],
  "categories": [
    {
      "id": 1,
      "cat_name": "التفسير",
      "parent_id": null
    }
  ],
  "authors": [
    {
      "id": 1,
      "name": "ابن كثير",
      "date": "700-774"
    }
  ]
}
```

### Indexing Strategy

```python
# Recommended indexing approach

async def index_all_datasets():
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Phase 1: Primary sources (10-20% of docs, 80% of questions)
    print("Indexing primary sources...")
    await rag.index_documents(
        categories=["التفسير", "كتب السنة", "الفقه العام"],
        limit=500,
    )
    
    # Phase 2: Secondary sources
    print("Indexing secondary sources...")
    await rag.index_documents(
        categories=["العقيدة", "النحو والصرف", "التاريخ"],
        limit=1000,
    )
    
    # Phase 3: Archival sources
    print("Indexing archival sources...")
    await rag.index_documents(limit=None)  # All remaining
    
    # Save indexes
    rag._pipeline._save_indexes()
```

---

## API Reference

### FastAPI Service

**File**: `rag_system/src/api/service.py`

```python
from fastapi import FastAPI
from rag_system.api.service import app

# Run with:
# uvicorn rag_system.src.api.service:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

```yaml
POST /api/v1/query:
  summary: Query the RAG system
  body:
    query: str
    top_k: int = 5
    domain: str = null
  response:
    answer: str
    sources: list
    latency_ms: float

POST /api/v1/query/stream:
  summary: Query with streaming response
  body:
    query: str
  response: Streaming text

POST /api/v1/compare:
  summary: Compare madhhab opinions
  body:
    question: str
  response:
    madhhab_results: dict
    consensus: dict

GET /api/v1/stats:
  summary: Get system statistics
  response:
    total_chunks: int
    total_documents: int
    categories: list

POST /api/v1/index:
  summary: Trigger indexing
  body:
    limit: int = null
    categories: list = null
```

### Example API Usage

```python
import requests

# Query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"query": "ما هو التوحيد؟", "top_k": 5},
)
result = response.json()

# Stream
response = requests.post(
    "http://localhost:8000/api/v1/query/stream",
    json={"query": "ما هو التوحيد؟"},
    stream=True,
)
for chunk in response.iter_content():
    print(chunk.decode(), end="")

# Compare
response = requests.post(
    "http://localhost:8000/api/v1/compare",
    json={"question": "ما حكم الصيام؟"},
)
result = response.json()
```

---

## Best Practices

### 1. Chunking Strategy Selection

| Document Type | Recommended Strategy | Chunk Size |
|---------------|---------------------|------------|
| Quran/Tafsir | islamic | 384 |
| Hadith | islamic | 256 |
| Fiqh | islamic | 512 |
| General | recursive | 512 |
| Poetry | islamic | 256 |
| History | semantic | 768 |

### 2. Embedding Model Selection

| Use Case | Recommended Model |
|----------|------------------|
| Arabic-only | MARBERT |
| Multilingual | mpnet-multilingual |
| High accuracy | OpenAI ada-3-large |
| Cost-effective | OpenAI ada-3-small |

### 3. Retrieval Optimization

```python
# For high precision
retriever = HybridRetriever(
    weights={"semantic": 0.7, "bm25": 0.3},
)
reranked = reranker.rerank(top_k=3)

# For high recall
retriever = HybridRetriever(
    weights={"semantic": 0.5, "bm25": 0.5},
    retrieval_top_k=100,
)
reranked = reranker.rerank(top_k=10)
```

### 4. Cost Optimization

```python
# Use caching
embedding_pipeline = create_embedding_pipeline(
    cache_dir="rag_system/data/embedding_cache",
)

# Use local models when possible
embedding_pipeline = create_embedding_pipeline(
    provider="sentence_transformers",
)

# Set budget limits
config = EmbeddingConfig(
    track_costs=True,
    budget_limit_usd=100.0,
)
```

### 5. Production Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
```

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory during indexing
```python
# Solution: Reduce batch size
pipeline = MultiSourceIngestionPipeline(batch_size=50)
```

**Issue**: Slow embedding generation
```python
# Solution: Use GPU or reduce model size
pipeline = create_embedding_pipeline(
    provider="sentence_transformers",
    model="bert-base-multilingual-cased",  # Smaller
)
```

**Issue**: Poor retrieval quality
```python
# Solution: Enable reranking
retriever = HybridRetriever(...)
reranker = Reranker(model_name="BAAI/bge-reranker-large")
```

---

## Next Steps

1. **Index your documents**: Start with primary sources
2. **Test queries**: Verify retrieval quality
3. **Tune parameters**: Adjust chunking, retrieval weights
4. **Deploy API**: Set up FastAPI service
5. **Monitor & evaluate**: Track metrics, user feedback

---

**Last Updated**: March 27, 2026
**Version**: 1.0.0
**Status**: Production Ready
