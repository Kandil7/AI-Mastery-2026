# RAG Module

**Retrieval-Augmented Generation pipeline.**

## Overview

This module implements a complete RAG pipeline with:
- Document chunking
- Embedding generation
- Vector storage
- Hybrid retrieval
- Reranking
- Response generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  Documents → Chunking → Embedding → Vector Store           │
│                              ↓                               │
│  Query → Embedding → Retrieval → Reranking → Generation    │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Chunking (`chunking.py`)
- `SemanticChunker` - Semantic boundary detection
- `FixedSizeChunker` - Fixed-size chunks
- `HierarchicalChunker` - Multi-level chunking

### Retrieval (`retrieval.py`)
- `HybridRetrieval` - Combine dense and sparse retrieval
- `BM25Retrieval` - Sparse retrieval
- `DenseRetrieval` - Vector similarity

### Reranking (`reranking.py`)
- `CrossEncoderReranker` - Cross-encoder reranking
- `LLMReranker` - LLM-based reranking

### Core (`core.py`)
- `RAGPipeline` - End-to-end pipeline
- `Document` - Document data model
- `DocumentChunk` - Chunk data model

## Usage

```python
from src.rag import RAGPipeline, SemanticChunker, HybridRetrieval
from src.embeddings import SentenceTransformerEmbeddings
from src.vector_stores import FAISSVectorStore

# Initialize components
embed_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
vector_store = FAISSVectorStore(dim=384)
chunker = SemanticChunker(chunk_size=512)

# Create pipeline
pipeline = RAGPipeline(
    embed_model=embed_model,
    vector_store=vector_store,
    chunker=chunker,
    retriever=HybridRetrieval(vector_store),
)

# Add documents
docs = [
    {"id": "1", "content": "AI is transforming industries."},
    {"id": "2", "content": "Machine learning enables automation."},
]
pipeline.add_documents(docs)

# Query
results = pipeline.query("How is AI impacting business?")
print(results[0].content)
```

## Advanced Features

### Query Enhancement
```python
from src.production import QueryEnhancementPipeline

enhancer = QueryEnhancementPipeline()
enhanced = enhancer.enhance("What is AI?")
# Returns: rewritten query + hypothetical document
```

### Semantic Caching
```python
from src.production import SemanticCache

cache = SemanticCache(embedder=embed_model.encode)
cached = cache.get("What is AI?")
if cached:
    return cached
```

### Cost Optimization
```python
from src.production import CostOptimizer

optimizer = CostOptimizer(embedder=embed_model.encode)
cached = optimizer.get_cached(query)
if cached:
    return cached

model, classification = optimizer.route(query)
response = call_llm(model, query)
optimizer.cache_and_track(query, response, model, tokens)
```

## Related Modules

- [`src/embeddings`](../embeddings/) - Embedding models
- [`src/vector_stores`](../vector_stores/) - Vector databases
- [`src/production`](../production/) - Production components
- [`src/evaluation`](../evaluation/) - RAG evaluation
