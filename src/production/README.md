# Production Engineering Module

**Enterprise-ready components for production RAG systems.**

This module implements the **5 pillars of production RAG**:

1. **Data Pipeline** - Semantic/hierarchical chunking, metadata extraction
2. **Hybrid Retrieval** - Integration with `src/retrieval` module
3. **Query Enhancement** - Query rewriting, HyDE, multi-query generation
4. **Evaluation & Observability** - Metrics, quality monitoring, alerting
5. **Cost Optimization** - Semantic caching, model routing, cost tracking

## Quick Start

```python
from src.production import (
    ProductionDataPipeline,
    SemanticChunker,
    QueryEnhancementPipeline,
    SemanticCache,
    CostOptimizer,
    RAGObservability,
)

# Initialize data pipeline
pipeline = ProductionDataPipeline(
    chunker=SemanticChunker(chunk_size=512),
    embedder=your_embedding_model,
)

# Process documents
documents = pipeline.process_documents(raw_docs)

# Initialize cost optimizer
optimizer = CostOptimizer(
    embedder=your_embedding_model,
    cache_threshold=0.90,
)

# Use in RAG flow
cached = optimizer.get_cached(query)
if cached:
    return cached

# Route to appropriate model
model, classification = optimizer.route(query)
response = call_llm(model, query)

# Track cost
optimizer.cache_and_track(query, response, model, tokens)
```

## Components

### Data Pipeline (`data_pipeline.py`)

- `Document` - Document data model
- `DocumentChunk` - Chunk data model
- `BaseChunker` - Abstract chunking strategy
- `FixedSizeChunker` - Fixed-size chunking
- `SemanticChunker` - Semantic boundary detection
- `HierarchicalChunker` - Multi-level chunking
- `MetadataExtractor` - Automatic metadata extraction
- `ProductionDataPipeline` - End-to-end pipeline

### Query Enhancement (`query_enhancement.py`)

- `EnhancedQuery` - Enhanced query data model
- `QueryRewriter` - Query rewriting strategies
- `HyDEGenerator` - Hypothetical document generation
- `MultiQueryGenerator` - Multiple query variants
- `SynonymExpander` - Query expansion
- `QueryEnhancementPipeline` - Complete enhancement pipeline

### Cost Optimization (`caching.py`)

- `SemanticCache` - Vector-based response caching
- `ModelRouter` - Intelligent model selection
- `CostTracker` - Token usage metering
- `CostOptimizer` - Unified optimization layer

### Observability (`observability.py`)

- `MetricPoint` - Metric data model
- `LatencyTracker` - Latency monitoring
- `QualityMonitor` - Quality scoring
- `RAGMetrics` - Comprehensive metrics
- `RAGObservability` - Full observability stack

### Monitoring (`monitoring.py`)

- `MetricsCollector` - Metrics collection
- `ModelMonitor` - Model performance monitoring
- `SystemMonitor` - System resource monitoring
- `AlertManager` - Alert management
- `ModelDriftDetector` - Drift detection

## Production Deployment

See the [Production Deployment Guide](../../docs/production_deployment_guide.md) for:

- Docker deployment
- Kubernetes configuration
- Monitoring setup
- Scaling strategies

## API Reference

The production API is served via FastAPI:

```bash
make run-api
```

Endpoints:
- `GET /health` - Health check
- `POST /predict` - Model predictions
- `GET /models` - List loaded models
- `GET /metrics` - Prometheus metrics

## Testing

```bash
# Run production tests
pytest tests/src/production/ -v

# Run with coverage
make test-cov
```

## Related Modules

- [`src/retrieval`](../retrieval/) - Hybrid retrieval strategies
- [`src/reranking`](../reranking/) - Reranking models
- [`src/rag`](../rag/) - Core RAG pipeline
- [`src/evaluation`](../evaluation/) - Evaluation frameworks
