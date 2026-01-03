# System Design: RAG System at Scale (1M Documents)

## Problem Statement

Design a Retrieval-Augmented Generation (RAG) system that can:
- Store and search 1M+ documents efficiently
- Handle 1000 QPS (queries per second)
- Provide relevant results with <500ms p95 latency
- Support both dense and sparse retrieval
- Scale horizontally as document count grows

---

## High-Level Architecture

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│       API Gateway + Load Balancer    │
│      (NGINX / AWS ALB / Kong)        │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│    Query Processing Service (FastAPI)│
│  - Query understanding                │
│  - Intent classification              │
│  - Query expansion (HyDE, Multi-Query)│
└──────┬───────────────────────────────┘
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
┌─────────┐   ┌──────────┐  ┌──────────┐
│ Hybrid  │   │ Semantic │  │   BM25   │
│Retrieval│   │  Search  │  │  Search  │
│(Fusion) │   │(Vector DB│  │(Elastic) │
└────┬────┘   └────┬─────┘  └────┬─────┘
     │             │              │
     └─────────────┴──────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │   Re-ranking     │
        │  (Cross-Encoder) │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  LLM Generation  │
        │  (GPT-4 / Llama) │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │  Response + Cite │
        └──────────────────┘
```

---

## Component Deep Dive

### 1. Document Ingestion Pipeline

**Goal**: Process 1M documents efficiently

```python
# Pseudo-implementation
class DocumentIngestionPipeline:
    def __init__(self):
        self.chunker = SemanticChunker(chunk_size=512, overlap=50)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = QdrantClient()
        self.elasticsearch = Elasticsearch()
        
    async def ingest_batch(self, documents: List[Document]):
        # Step 1: Chunk documents
        chunks = [
            chunk 
            for doc in documents 
            for chunk in self.chunker.chunk(doc)
        ]
        
        # Step 2: Generate embeddings (batched)
        embeddings = await self.embedder.encode_async(
            [c.text for c in chunks],
            batch_size=256,
            show_progress_bar=True
        )
        
        # Step 3: Store in vector DB (parallel)
        await asyncio.gather(
            self.vector_db.upsert(chunks, embeddings),
            self.elasticsearch.index(chunks)  # For BM25
        )
```

**Considerations**:
- **Chunking strategy**: Semantic vs fixed-size
- **Embedding model**: Speed vs quality trade-off
- **Batch processing**: Use Celery/RabbitMQ for scalability
- **Idempotency**: Handle duplicate ingestion

---

### 2. Vector Database Selection

**Options Comparison**:

| Database | Pros | Cons | Best For |
|----------|------|------|----------|
| **Qdrant** | Fast, Rust-based, multi-tenancy | Newer ecosystem | Production RAG |
| **Pinecone** | Managed, auto-scaling | Cost, vendor lock-in | Quick MVP |
| **Weaviate** | GraphQL, hybrid search | Complex setup | Knowledge graphs |
| **Custom HNSW** | Full control, cost-effective | Maintenance burden | Learning/research |

**Recommended**: **Qdrant** + **Elasticsearch** (hybrid search)

**Qdrant Configuration**:
```yaml
collection_config:
  vectors:
    size: 384  # all-MiniLM-L6-v2 dimension
    distance: Cosine
  optimizers_config:
    memmap_threshold: 20000  # Use disk for >20K vectors
  hnsw_config:
    m: 16                    # Connections per layer
    ef_construct: 256        # Index build quality
    full_scan_threshold: 10000
  quantization_config:
    scalar:
      type: int8             # 4x compression
      quantile: 0.99
```

**Scaling Strategy**:
- **Sharding**: Distribute 1M docs across 4 shards (250K each)
- **Replicas**: 2+ replicas for high availability
- **Memory**: ~4GB RAM per 250K documents (@384 dims)

---

### 3. Retrieval Strategy

**Hybrid Retrieval (Combining Dense + Sparse)**:

```python
class HybridRetriever:
    def __init__(self):
        self.vector_db = QdrantClient()
        self.bm25 = ElasticsearchBM25()
        self.alpha = 0.7  # Weight for semantic search
        
    async def retrieve(self, query: str, top_k: int = 10):
        # Parallel retrieval
        semantic_results, bm25_results = await asyncio.gather(
            self.semantic_search(query, top_k=50),
            self.bm25.search(query, top_k=50)
        )
        
        # Reciprocal Rank Fusion
        return self._fuse_results(semantic_results, bm25_results, top_k)
    
    def _fuse_results(self, semantic, bm25, top_k):
        scores = defaultdict(float)
        for rank, doc in enumerate(semantic):
            scores[doc.id] += self.alpha / (rank + 60)
        for rank, doc in enumerate(bm25):
            scores[doc.id] += (1 - self.alpha) / (rank + 60)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**Why Hybrid?**
- **Semantic**: Captures meaning, handles synonyms
- **BM25**: Handles exact matches, rare terms
- **Fusion**: Best of both worlds (15-20% improvement)

---

### 4. Re-ranking Layer

**Purpose**: Improve precision of top-K results

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def rerank(self, query: str, candidates: List[Document], top_k: int = 5):
        # Score all query-document pairs
        pairs = [(query, doc.text) for doc in candidates]
        scores = self.model.predict(pairs)
        
        # Re-sort by cross-encoder score
        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, _ in reranked[:top_k]]
```

**Trade-off**: Adds 50-100ms latency but +10-15% accuracy

---

### 5. Caching Strategy

**Multi-Level Caching**:

```
┌─────────────────┐
│ L1: In-Memory   │  (LRU, 1000 queries, ~5ms hit)
│    (Redis)      │
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│ L2: Semantic    │  (Embedding similarity, 10ms)
│    Cache        │
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│ L3: Vector DB   │  (Full retrieval, 100-200ms)
└─────────────────┘
```

**Implementation**:
```python
class SemanticCache:
    def __init__(self, threshold=0.95):
        self.redis = redis.Redis()
        self.embedder = SentenceTransformer()
        self.threshold = threshold
        
    async def get(self, query: str):
        # Check exact match
        if cached := self.redis.get(query):
            return cached
        
        # Check semantic similarity
        query_emb = self.embedder.encode(query)
        similar = await self.find_similar(query_emb, threshold=self.threshold)
        if similar:
            return similar.response
        
        return None  # Cache miss
```

**Expected Hit Rate**: 40-50% for common queries

---

### 6. Scaling & Performance

**Latency Breakdown** (Target: <500ms p95):

| Component | Latency | Notes |
|-----------|---------|-------|
| Query processing | 10ms | Intent + expansion |
| Semantic search | 80ms | Qdrant HNSW lookup |
| BM25 search | 50ms | Elasticsearch |
| Re-ranking | 70ms | Cross-encoder (10 docs) |
| LLM generation | 250ms | GPT-4 Turbo /claude-instant |
| **Total** | **460ms** | Within target ✓ |

**Horizontal Scaling**:
```
Load Balancer
      │
      ├─► API Pod 1 ────┐
      ├─► API Pod 2 ────┤
      ├─► API Pod 3 ────┼──► Vector DB Cluster (3 shards, 2 replicas)
      ├─► API Pod 4 ────┤
      └─► API Pod 5 ────┘
```

**Capacity Planning** (1000 QPS):
- **API Pods**: 5 pods @ 200 QPS each
- **Vector DB**: 3 shards @ 333 QPS each
- **Elasticsearch**: 3 nodes @ 333 QPS each

---

### 7. Monitoring & Observability

**Key Metrics** (Prometheus + Grafana):

```python
# Metrics to track
RETRIEVAL_LATENCY = Histogram('rag_retrieval_latency_seconds', ...)
GENERATION_LATENCY = Histogram('rag_generation_latency_seconds', ...)
CACHE_HIT_RATE = Gauge('rag_cache_hit_rate', ...)
RELEVANCE_SCORE = Histogram('rag_relevance_score', ...)  # User feedback
COST_PER_QUERY = Counter('rag_cost_per_query_usd', ...)
```

**Alerts**:
1. Latency p95 > 700ms (2 minutes)
2. Error rate > 1% (1 minute)
3. Cache hit rate < 30%
4. Vector DB memory > 90%

---

### 8. Cost Optimization

**Estimated Monthly Cost** (1M docs, 1000 QPS):

| Component | Cost | Notes |
|-----------|------|-------|
| Qdrant (self-hosted) | $300 | 3x c5.2xlarge EC2 |
| Elasticsearch | $400 | 3x r5.large |
| LLM API (GPT-4 Turbo) | $5000 | 1000 QPS × 250 tokens × $0.01/1K |
| Redis Cache | $50 | ElastiCache |
| Data Transfer | $100 | Egress |
| **Total** | **~$5,850/month** | |

**Optimization Strategies**:
1. **Use open-source LLM** (Llama 3.1 70B): -80% cost
2. **Aggressive caching**: 50% hit rate = -40% LLM cost
3. **Quantization**: int8 embeddings = -75% vector DB cost
4. **Batch processing**: Group queries = +30% throughput

---

## Trade-offs & Decisions

| Decision | Option A | Option B | Choice |
|----------|----------|----------|--------|
| Embedding Model | all-MiniLM (fast) | BGE-large (accurate) | MiniLM (speed) |
| Vector DB | Managed (Pinecone) | Self-hosted (Qdrant) | Qdrant (cost) |
| LLM | GPT-4 (quality) | Llama 3 (cheap) | Hybrid (smart routing) |
| Chunking | Fixed 512 tokens | Semantic (sentences) | Semantic (quality) |

---

## Interview Discussion Points

1. **How would you handle document updates?**
   - Use versioning in Qdrant
   - Incremental updates, not full reindex
   - TTL on cache entries

2. **What if latency exceeds 500ms?**
   - Reduce re-ranking candidates (10 → 5)
   - Switch to faster LLM (GPT-3.5 / claude-instant)
   - Pre-compute query embeddings for common patterns

3. **How to ensure result quality?**
   - A/B test hybrid vs pure semantic
   - User feedback loop (thumbs up/down)
   - Offline evaluation with RAGAS metrics

4. **Security considerations?**
   - Multi-tenancy: Row-level security in Qdrant
   - PII filtering before LLM generation
   - Rate limiting per API key

---

## Conclusion

This design handles 1M documents at 1000 QPS with <500ms p95 latency by:
- **Hybrid retrieval** (semantic + BM25)
- **Aggressive multi-level caching**
- **Horizontal scaling** (sharding + replicas)
- **Smart cost optimization** (quantization, open-source LLMs)

**Production-ready checklist**:
- ✅ Handles 10x traffic spike (autoscaling)
- ✅ Fault-tolerant (replicas, health checks)
- ✅ Observable (metrics, logs, traces)
- ✅ Cost-effective (<$6K/month)
