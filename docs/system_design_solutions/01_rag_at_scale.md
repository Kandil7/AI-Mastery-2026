# System Design: RAG System at Scale (1M Documents)

## Problem Statement

Design a Retrieval-Augmented Generation (RAG) system that can:
- Store and search 1M+ documents efficiently
- Handle 1000 QPS (queries per second)
- Provide relevant results with <500ms p95 latency
- Support both dense and sparse retrieval
- Scale horizontally as document count grows

---

## دراسات حالة إضافية (2024-2025) + حلول تصميمية

- **Databricks LakehouseIQ (2024)**: مساعد بيانات مؤسسي مبني على Mosaic AI وUnity Catalog. يستخدم
  فهرسة هجينة (Vector + نص) مع وراثة أذونات الكتالوج ومُخطِّط يربط السؤال بالجداول/اللوحات
  ذات الصلة ليولّد SQL أو إجابات مؤسَّسة بالسياق. الحوكمة مدمجة لأن كل استرجاع يمر عبر الكتالوج
  والـ Lineage.

- **Snowflake Cortex Search (GA 2025)**: خدمة مُدارة لـ RAG هجيني فوق الجداول أو الـ Stages بلا
  إدارة بنية تحتية. Chunking تلقائي، Arctic Embed، فهرسة متجهية + BM25، وخيار target_lag لضبط
  حداثة البيانات. تُستهلك عبر REST/SDK مع طبقة أمان Snowflake نفسها.

- **Slack Enterprise Search (2024)**: بحث ذكي قائم على RAG فوق رسائل Slack وموصلات SaaS
  (Google Drive, O365...). استرجاع واعٍ بالأذونات (ACL) قبل الـ LLM، ولا تُخزَّن بيانات الموصلات
  خارج نطاق المستأجر. إجابات مؤسَّسة بالمراجع وروابط مصدر لتقليل الهلوسة.

---

## Enterprise RAG: Production Patterns (18 Case Studies)

**Context:** هندسة أنظمة توليد النصوص المعزز بالاسترجاع (RAG) في النطاق المؤسسي تتطلب منظومة
إنتاجية تتحمل ملايين الاستعلامات مع احترام الصلاحيات والاستجابة دون ثانية. أدناه خلاصة هندسية
قابلة للتنفيذ مبنية على 18 حالة دراسية (Notion، Intercom، Uber، Pinterest، Microsoft،
Salesforce، Airbnb، Klarna، Shopify، Intuit...).

### 1) أنماط معمارية أثبتت نجاحها

- Hybrid Retrieval هو المعيار الواقعي: متجهات + BM25 + فلاتر ميتاداتا + Reranker مخصص (Fin-cx،
  ModernBERT، Cross-Encoder) لرفع الدقة وخفض الهلوسة.
- Agentic RAG بدلاً من تدفق خطي: تفكيك السؤال، تضييق المصدر، استرجاع متوازٍ، Tool calling،
  تحقق LLM-as-Judge (Uber EAg-RAG، Shopify Agentic Loop).
- GraphRAG / Work-Graph: رسوم بيانية (Klarna/Neo4j) أو بنية الكتل (Notion) لحفظ العلاقات.
- Object/Block Chunking: تقطيع هيكلي للجداول والسجلات (Salesforce، Uber) يقلل الانجراف المعرفي.
- أذونات مدمجة في الاسترجاع: ربط الاستعلام بـ Entra ID/OAuth/RBAC قبل دخول الفهارس لمنع التسرب.

### 2) مرجع معماري مختصر (Canonical Layers)

1. **Ingestion**: تحويل PDF/Docs -> HTML/Markdown، تطبيع الجداول، ميتاداتا غنية، ملخصات جداول
   واستعلامات تاريخية (Pinterest Text-to-SQL) مع Chunking هيكلي.
2. **Indexing & Retrieval**: متجهات (Faiss/OpenSearch/Neo4j) + BM25 + فلاتر صلاحيات؛ Fusion
   (RRF/Weighted)؛ Reranker بنافذة سياق واسعة (8K+).
3. **Orchestration / Agents**: مخطط استعلام، تحديد مصدر، استرجاع متوازٍ، Tool/SQL/API، تحقق
   (Validation Agent) مع LLM-as-judge.
4. **Generation**: توجيه نموذج (Routing) بين صغير/كبير؛ Grounded generation؛ Streaming JSON جزئي.
5. **Trust & Governance**: صلاحيات في الاسترجاع، Masking مسبق، Trace لكل إجابة، تدقيق سمّية.
6. **Evaluation & Observability**: بنية تقييم سيادية (LLM+بشر)، RAGAS، مراقبة الانجراف والتكلفة.

### 3) مقاييس نجاح إنتاجية

- Latency: <800ms p95 (مع Streaming) بميزانية زمنية لكل طبقة.
- Recall@K وPrecision بعد Rerank.
- Hallucination Rate، Answer Provenance، Task Success Rate.
- Cost per Query مع مسارات نماذج بديلة وكاش دلالي متدرج.

### 4) خارطة طريق سريعة الدمج مع هذا المستند

- Align: اربط المكونات الحالية (Hybrid + Rerank + Caching) بطبقات المرجع أعلاه.
- Upgrade Retrieval: فلاتر صلاحيات + Reranker أقوى (ModernBERT/ColBERTv2) بنافذة 8K.
- Agent Layer: مخطط بسيط (decompose -> retrieve -> tool-call) قبل التوليد لدعم بيانات زمنية.
- Trust-first: فحص الصلاحيات في الاسترجاع؛ Trace ID مع كل إجابة.
- Eval Loop: تشغيل LLM-as-judge دوريًا مقابل تعليقات المستخدم.

---

## 1. Enterprise RAG Reference Architecture (قابل للتنفيذ)

### الهدف
منصة RAG مؤسسية آمنة، منخفضة الكمون، متعددة المصادر، تدعم Agentic/Graph reasoning.

### الطبقات الخمس
1) **Ingestion**: تحويل PDF/Docs -> HTML/Markdown، تطبيع الجداول، Chunking هيكلي (Blocks/Objects).
2) **Indexing & Retrieval**: متجهات + BM25 + فلاتر أذونات؛ Fusion (RRF/Weighted)؛ Reranker مخصص.
3) **Orchestration / Agents**: تفكيك الاستعلام -> تضييق المصدر -> استرجاع متوازٍ -> Tool calling -> تحقق.
4) **Generation**: توجيه نموذج (Routing) بين صغير/كبير؛ Grounded generation؛ Streaming JSON جزئي.
5) **Trust & Governance**: صلاحيات على مستوى الاسترجاع، Masking مسبق، Trace لكل إجابة، تدقيق سمّية.

### مبادئ تصميم حاسمة
- Hybrid retrieval قاعدة وليست خيارًا.
- الأمان يبدأ من المسترجع.
- كاش دلالي متدرج لتقليل LLM cost والكمون.
- تقييم سيادي مستمر (LLM+بشر) لرصد الانجراف.

---

## 2. Product Requirements Document (ملخص تنفيذي)

### Vision
تحويل RAG إلى Knowledge Operating System يخدم فرق الهندسة، الدعم، التحليلات، والقيادة.

### Functional Requirements
- FR1: وصول آمن (RBAC/ABAC، عدم تسرب البيانات).
- FR2: محرك Hybrid Retrieval مع Reranker قابل للضبط.
- FR3: Agentic reasoning + Tool/SQL/API.
- FR4: Latency <800ms p95 مع Streaming تدريجي.
- FR5: تقييم ومراقبة (Provenance، Hallucination، Cost).

### Non-Functional
Latency <1s p95 | 10M+ q/day | 99.9% توفر | SOC2/ISO | تتبع كامل لكل إجابة.

### Success Metrics
Accuracy (LLM+Human) | Recall@K | Hallucination rate | Task completion | Cost/query.

---

## 3. Internal Training Program (Enterprise RAG)

- Level 1: Foundations — لماذا يفشل RAG الساذج؟ حدود البحث المتجهي.
- Level 2: Retrieval Engineering — Hybrid، Rerankers، datasets، تحليل الأخطاء. (Intercom, Notion)
- Level 3: Agentic RAG — Planners، tool calling، state machines، retry loops. (Shopify, Intuit)
- Level 4: GraphRAG — Neo4j، multi-hop reasoning، Work-Graph. (Klarna)
- Level 5: Production & Governance — أمن، تكلفة، مراقبة، incident response.

---

## الخلاصة التنفيذية
النظام المطلوب منصة معرفة مؤسسية آمنة. الفروق الحاسمة: صلاحيات في طبقة الاسترجاع، إدخال Agent
Layer قبل التوليد، اعتماد Hybrid+Reranker دائمًا، تقييم سيادي دوري لكشف الانجراف والهلوسة.

## خطوات عملية تالـية
1) تضمين بيانات الأذونات (tenant_id, user_roles, scopes) في استعلام الاسترجاع.
2) دمج Reranker قوي (ModernBERT/ColBERTv2) بنافذة 8K وحد 10 مرشحين.
3) تنفيذ Agent skeleton: `plan -> retrieve -> tool-call -> validate -> generate`.
4) تفعيل Streaming JSON جزئي لتقليل الكمون المتراكم.
5) لوحة مراقبة: latency budget لكل طبقة، hallucination rate، cost/query، cache hit-rate.

---
## High-Level Architecture

```
+-------------+
|  User Query |
+-------------+
      |
      v
+-------------------------------+
| API Gateway / Load Balancer   |
| (NGINX / AWS ALB / Kong)      |
+---------------+---------------+
                |
                v
+-------------------------------+
| Query Processing (FastAPI)    |
| - intent / parsing / expand   |
+-------+-----------+-----------+
        |           |
        v           v
  +-----------+   +------------------+
  | Semantic  |   | Keyword / BM25   |
  | Vector DB |   | (Elastic/OpenSrch)|
  +-----+-----+   +---------+--------+
        \\             //
         v           v
     +--------------------+
     |  Fusion (RRF/Rank) |
     +---------+----------+
               |
               v
     +--------------------+
     |   Reranker         |
     | (Cross-Encoder)    |
     +---------+----------+
               |
               v
     +--------------------+
     |  LLM Generation    |
     | (GPT-4 / Llama)    |
     +---------+----------+
               |
               v
     +--------------------+
     | Response + Cite    |
     +--------------------+
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
+---------------------------+
| L1: In-Memory (Redis)     | ~5ms, LRU, ~1000 queries
+---------------------------+
            miss
             v
+---------------------------+
| L2: Semantic Cache        | ~10ms similarity
+---------------------------+
            miss
             v
+---------------------------+
| L3: Vector DB             | 100-200ms full retrieval
+---------------------------+
```
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
| **Total** | **460ms** | Within target |

**Horizontal Scaling (text view):**
- Load balancer in front of API pods.
- 5× API pods, each ~200 QPS.
- Vector DB cluster: 3 shards, 2 replicas.
- Elasticsearch: 3 nodes (~333 QPS each).
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
| LLM API (GPT-4 Turbo) | $5000 | 1000 QPS x 250 tokens x $0.01/1K |
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
   - Reduce re-ranking candidates (10 -> 5)
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
- [OK] Handles 10x traffic spike (autoscaling)
- [OK] Fault-tolerant (replicas, health checks)
- [OK] Observable (metrics, logs, traces)
- [OK] Cost-effective (<$6K/month)
