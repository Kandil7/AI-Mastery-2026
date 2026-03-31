# From Prototype to Production: Enterprise-Ready RAG Systems

A strategic guide for building production-grade RAG systems.

---

## Table of Contents

1. [Why Simple RAG Fails](#1-why-simple-rag-fails)
2. [The 5 Production Pillars](#2-the-5-production-pillars)
3. [Pillar 1: Data Pipeline](#pillar-1-data-pipeline)
4. [Pillar 2: Hybrid Retrieval](#pillar-2-hybrid-retrieval)
5. [Pillar 3: Query Enhancement](#pillar-3-query-enhancement)
6. [Pillar 4: Evaluation & Observability](#pillar-4-evaluation--observability)
7. [Pillar 5: Cost Optimization](#pillar-5-cost-optimization)
8. [Using This Module](#using-this-module)

---

## 1. Why Simple RAG Fails

| Failure Mode | Cause | Impact |
|--------------|-------|--------|
| **Contextual Blindness** | LLM ignores retrieved docs | Hallucinations |
| **Stale Knowledge** | Outdated embeddings | Wrong answers |
| **Fragmented Context** | Naive chunking | Incomplete answers |
| **Superficial Relevance** | Vector-only search | Missed exact terms |

**Bottom line**: A production system cannot be a scaled-up prototype.

---

## 2. The 5 Production Pillars

```
┌─────────────────────────────────────────────────────────────┐
│                 PRODUCTION RAG PILLARS                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│   DATA      │   HYBRID    │   QUERY     │   EVAL &    │COST │
│  PIPELINE   │  RETRIEVAL  │ ENHANCEMENT │ OBSERV.     │ OPT │
├─────────────┼─────────────┼─────────────┼─────────────┼─────┤
│ Semantic    │ BM25+Dense  │ Query       │ Faithfulness│Cache│
│ Chunking    │             │ Rewriting   │             │     │
│             │             │             │             │     │
│ Hierarchical│ Metadata    │ HyDE        │ Hallucin.   │Route│
│ Chunks      │ Filtering   │             │ Detection   │     │
│             │             │             │             │     │
│ Metadata    │ Cross-      │ Multi-Query │ Latency     │Track│
│ Extraction  │ Encoder     │             │ Percentiles │     │
└─────────────┴─────────────┴─────────────┴─────────────┴─────┘
```

---

## Pillar 1: Data Pipeline

**Location**: `src/production/data_pipeline.py`

### Chunking Strategies

| Strategy | When to Use | Code |
|----------|-------------|------|
| **Semantic** | General text | `SemanticChunker` |
| **Hierarchical** | Need precision + context | `HierarchicalChunker` |
| **Fixed** | Simple cases only | `FixedSizeChunker` |

### Example

```python
from src.production.data_pipeline import (
    ProductionDataPipeline,
    HierarchicalChunker,
    MetadataExtractor
)

# Production pipeline with hierarchical chunking
pipeline = ProductionDataPipeline(
    chunker=HierarchicalChunker(
        parent_chunk_size=2000,
        child_chunk_size=400
    )
)

# Process document
chunks = pipeline.process(
    content=document_text,
    format="markdown",
    doc_id="doc_001"
)

# Chunks include:
# - Parent chunks (for context)
# - Child chunks (for retrieval)
# - Rich metadata (doc type, entities, etc.)
```

---

## Pillar 2: Hybrid Retrieval

Uses existing `src/retrieval/` module with:

| Method | Purpose |
|--------|---------|
| **BM25** | Exact keyword matching |
| **Dense** | Semantic similarity |
| **Hybrid** | Combined (recommended) |

**Key insight**: Semantic search understands concepts; keyword search finds exact terms. Production needs both.

---

## Pillar 3: Query Enhancement

**Location**: `src/production/query_enhancement.py`

### Techniques

| Technique | What It Does |
|-----------|--------------|
| **Query Rewriting** | "tell me about sales" → "Q3 sales report" |
| **HyDE** | Generate hypothetical answer, embed that |
| **Multi-Query** | Search from multiple angles |
| **Synonym Expansion** | "revenue" → also search "sales", "income" |

### Example

```python
from src.production.query_enhancement import QueryEnhancementPipeline

pipeline = QueryEnhancementPipeline()
enhanced = pipeline.enhance("What are our ML best practices?")

# Use all query variants for retrieval
for query_variant in enhanced.get_all_queries():
    results = retriever.retrieve(query_variant)
```

---

## Pillar 4: Evaluation & Observability

**Location**: `src/production/observability.py`

### Key Metrics

| Category | Metrics |
|----------|---------|
| **Quality** | Faithfulness, relevance, hallucination rate |
| **Latency** | P50, P90, P99 |
| **Cost** | Tokens, cache hit rate |

### Example

```python
from src.production.observability import RAGObservability

obs = RAGObservability()

# Track a full request
with obs.track_request("query"):
    results = rag_pipeline(query)

# Assess quality
quality = obs.assess_quality(
    query=query,
    response=results.answer,
    context=results.contexts
)

# Export Prometheus metrics
print(obs.get_prometheus_metrics())
```

---

## Pillar 5: Cost Optimization

**Location**: `src/production/caching.py`

### Techniques

| Technique | Savings |
|-----------|---------|
| **Semantic Cache** | 60-90% on repeat queries |
| **Model Routing** | 40-70% via tiered models |
| **Token Tracking** | Visibility for optimization |

### Example

```python
from src.production.caching import CostOptimizer

optimizer = CostOptimizer(
    embedder=embed_function,
    cache_threshold=0.90
)

# Check cache first
cached = optimizer.get_cached(query)
if cached:
    return cached  # Free!

# Route to appropriate model
model, classification = optimizer.route(query)

# After LLM call, cache and track
optimizer.cache_and_track(
    query=query,
    response=response,
    model=model,
    prompt_tokens=100,
    completion_tokens=200
)
```

---

## Using This Module

### Full Production RAG

```python
from src.production import (
    ProductionDataPipeline,
    QueryEnhancementPipeline,
    CostOptimizer,
    RAGObservability
)
from src.retrieval import HybridRetriever
from src.reranking import CrossEncoderReranker

# 1. INGEST
pipeline = ProductionDataPipeline()
chunks = pipeline.process_batch(documents)

# 2. INDEX
retriever = HybridRetriever(alpha=0.5)
retriever.index([c.to_document() for c in chunks])

# 3. OPTIMIZE
optimizer = CostOptimizer(embedder=embed_fn)
obs = RAGObservability()
query_enhancer = QueryEnhancementPipeline()

# 4. QUERY (production loop)
def answer(query: str) -> str:
    with obs.track_request("query"):
        # Check cache
        cached = optimizer.get_cached(query)
        if cached:
            obs.record_cache_hit(True)
            return cached
        
        obs.record_cache_hit(False)
        
        # Enhance query
        enhanced = query_enhancer.enhance(query)
        
        # Retrieve with all variants
        all_results = []
        for q in enhanced.get_all_queries():
            all_results.extend(retriever.retrieve(q, top_k=10))
        
        # Rerank
        reranked = CrossEncoderReranker().rerank(
            query, 
            [r.document for r in all_results],
            top_k=5
        )
        
        # Generate (with appropriate model)
        model, _ = optimizer.route(query)
        response = llm_generate(query, reranked, model)
        
        # Quality check
        obs.assess_quality(query, response, reranked)
        
        # Cache and track cost
        optimizer.cache_and_track(query, response, model, 100, 200)
        
        return response
```

---

## Summary

| Pillar | Key Takeaway |
|--------|--------------|
| **Data Pipeline** | Semantic chunking preserves context |
| **Hybrid Retrieval** | Combine semantic + keyword |
| **Query Enhancement** | Bridge vocabulary gap |
| **Observability** | Measure quality, not just latency |
| **Cost Optimization** | Semantic cache + model routing |

**Result**: 60-90% cost reduction, higher quality, production reliability.
