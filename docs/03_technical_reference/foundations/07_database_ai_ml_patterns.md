# AI/ML Database Patterns

## Introduction

The intersection of artificial intelligence and database systems represents one of the most significant shifts in data infrastructure in recent years. As machine learning models become increasingly embedded in production applications, the need for specialized database patterns optimized for AI workloads has grown substantially. This guide provides comprehensive coverage of database architectures and patterns designed specifically for AI and machine learning applications, covering everything from vector storage and retrieval to feature store implementations.

Traditional relational databases were designed primarily for structured data and transactional workloads. However, AI applications frequently work with unstructured data such as text, images, and audio, which requires different storage and retrieval mechanisms. The emergence of vector databases, which store and query high-dimensional embeddings, addresses this gap by enabling semantic similarity search at scale. Understanding when and how to use these specialized systems, along with hybrid approaches that combine multiple database technologies, is essential for building modern AI applications.

This guide covers vector database optimization techniques that maximize retrieval accuracy and minimize latency, embedding storage and retrieval patterns that balance storage costs with query performance, hybrid search strategies that combine semantic and keyword approaches, ML model metadata storage systems for experiment tracking and model management, and feature store patterns that enable consistent feature engineering across training and inference. Each topic includes practical implementation guidance, code examples, and trade-off analyses to help you choose the right approach for your specific use case.

The patterns and techniques presented here draw from real-world production systems and are designed to scale from prototype to production. Whether you are building a retrieval-augmented generation system, implementing a recommendation engine, or creating a feature platform for machine learning, this guide provides the foundational knowledge needed to design robust, performant AI data infrastructure.

---

## 1. Vector Database Optimization for AI Workloads

### Understanding Vector Databases

Vector databases are specialized storage systems designed to efficiently store, index, and query high-dimensional vector embeddings. Unlike traditional databases that optimize for exact matches or range queries, vector databases excel at approximate nearest neighbor (ANN) searches, finding the most similar vectors to a query vector. This capability is fundamental to many AI applications, including semantic search, recommendation systems, image retrieval, and retrieval-augmented generation.

The core challenge that vector databases address is the "curse of dimensionality." As the number of dimensions in vectors increases, the distance between any two points becomes increasingly similar, making it difficult to distinguish relevant from irrelevant results using traditional indexing approaches. Vector databases employ specialized indexing algorithms, such as hierarchical navigable small world (HNSW) graphs, inverted file (IVF) indexes, or product quantization (PQ), to maintain search efficiency even with millions or billions of vectors.

```python
# Python: Creating vector embeddings using a sentence transformer
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained model for generating text embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for a corpus of documents
documents = [
    "Machine learning is transforming software development",
    "Natural language processing enables computers to understand text",
    "Deep learning uses neural networks with many layers",
    "Computer vision allows machines to interpret images",
    "Reinforcement learning trains agents through rewards"
]

# Generate 384-dimensional embeddings
embeddings = model.encode(documents)
print(f"Embedding shape: {embeddings.shape}")  # (5, 384)
print(f"Embedding dimension: {embeddings.shape[1]}")  # 384

# Generate query embedding for similarity search
query = "How do neural networks learn?"
query_embedding = model.encode([query])[0]
print(f"Query embedding shape: {query_embedding.shape}")  # (384,)
```

### Vector Indexing Strategies

Choosing the right indexing strategy is critical for achieving the right balance between search speed, accuracy, and resource usage. Different indexing algorithms offer different trade-offs, and the optimal choice depends on your specific requirements.

**HNSW (Hierarchical Navigable Small World)** is a graph-based indexing algorithm that provides excellent search performance with high recall. It builds a multi-layer graph where each layer represents a different level of granularity, allowing for efficient navigation during search. HNSW is ideal for applications requiring high accuracy and moderate index sizes, but it requires significant memory for the graph structure.

```python
# Python: Creating an HNSW index with Pinecone
from pinecone import Pinecone

# Initialize Pinecone with API key
pc = Pinecone(api_key="your-api-key")

# Create a serverless index with HNSW
index_name = "semantic-search"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )

# Configure HNSW parameters at index creation
# (Pinecone handles HNSW internally with optimized defaults)
index = pc.Index(index_name)

# Upsert vectors with metadata
vectors = [
    {
        "id": "doc1",
        "values": embeddings[0].tolist(),
        "metadata": {
            "text": documents[0],
            "category": "ml",
            "created_at": "2024-01-15"
        }
    },
    {
        "id": "doc2",
        "values": embeddings[1].tolist(),
        "metadata": {
            "text": documents[1],
            "category": "nlp",
            "created_at": "2024-01-16"
        }
    }
]

index.upsert(vectors=vectors)
```

**IVF (Inverted File)** indexes cluster vectors into partitions and use an inverted index to quickly identify which partitions are most likely to contain nearest neighbors. This approach reduces the number of distance computations needed during search but may sacrifice some accuracy compared to HNSW. IVF indexes are more memory-efficient than HNSW and work well for very large datasets.

```sql
-- PostgreSQL with pgvector: Creating IVF index
-- First, enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create IVF index with quantization
-- nlists: number of inverted lists (more lists = faster but less accurate)
-- n probes: number of lists to search at query time
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- For larger datasets, use more lists
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);
```

**Product Quantization (PQ)** compresses high-dimensional vectors into compact codes, dramatically reducing storage requirements and enabling very fast search. PQ is particularly valuable when working with billions of vectors where memory is a constraint. However, the compression introduces some loss in accuracy.

```python
# Python: Using FAISS with Product Quantization
import faiss
import numpy as np

# Suppose we have 1 million 128-dimensional vectors
n_vectors = 1_000_000
dimension = 128
vectors = np.random.random((n_vectors, dimension)).astype('float32')

# Normalize for cosine similarity
faiss.normalize_L2(vectors)

# Create IVF index with Product Quantization
# nlist: number of clusters
# m: number of subquantizers (affects compression ratio)
# nbits: bits per subquantizer (affects compression)
nlist = 1000  # Number of clusters
m = 16        # 16 subquantizers for 128-dim = 8 dimensions each
nbits = 8     # 8 bits per subquantizer = 256 centroids per subquantizer

# Create the index
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

# Train the index (required before adding vectors)
index.train(vectors[:100000])  # Use subset for faster training
index.add(vectors)

# Search parameters
index.nprobe = 10  # Number of clusters to search

# Perform search
query = np.random.random((1, dimension)).astype('float32')
faiss.normalize_L2(query)
distances, indices = index.search(query, k=10)
```

### Performance Benchmarks

Understanding the performance characteristics of different vector database configurations helps you make informed decisions about infrastructure. The following benchmarks illustrate typical performance ranges for common configurations.

| Index Type | Dataset Size | Index Build Time | Query Latency (P99) | Recall@10 | Memory Usage |
|------------|-------------|-------------------|---------------------|-----------|---------------|
| HNSW (M=16, ef=64) | 1M vectors | 2-5 minutes | 5-15ms | 95-98% | 2-4 GB |
| HNSW (M=32, ef=128) | 1M vectors | 5-10 minutes | 10-25ms | 97-99% | 4-8 GB |
| IVF-PQ (1000 lists) | 1M vectors | 3-8 minutes | 20-50ms | 85-92% | 0.5-1 GB |
| IVF-PQ (100 lists) | 1M vectors | 2-5 minutes | 30-80ms | 80-88% | 0.3-0.5 GB |
| Flat (brute force) | 1M vectors | N/A | 200-500ms | 100% | 0.5 GB |

These benchmarks demonstrate the classic trade-off between speed and accuracy. HNSW provides the best balance for most production use cases, offering millisecond-level latency with high recall. IVF-PQ significantly reduces memory usage at the cost of some accuracy, making it suitable for very large datasets or memory-constrained environments.

```python
# Python: Benchmarking vector search performance
import time
import numpy as np
from statistics import mean, stdev

def benchmark_search(index, queries, k=10, runs=100):
    """Benchmark search performance across multiple runs"""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for query in queries:
            index.search(query, k)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    return {
        'mean_ms': mean(times),
        'p50_ms': times[len(times) // 2],
        'p95_ms': times[int(len(times) * 0.95)],
        'p99_ms': times[int(len(times) * 0.99)],
        'stdev_ms': stdev(times) if len(times) > 1 else 0
    }

# Example usage
# Generate test queries
test_queries = [np.random.random(384).astype('float32') for _ in range(100)]

# Benchmark results
results = benchmark_search(pinecone_index, test_queries, k=10, runs=50)
print(f"Mean: {results['mean_ms']:.2f}ms, P99: {results['p99_ms']:.2f}ms")
```

---

## 2. Embedding Storage and Retrieval Patterns

### Embedding Generation and Storage Pipeline

Building an efficient embedding storage and retrieval pipeline requires careful consideration of data flow, storage format, and retrieval patterns. The pipeline typically consists of several stages: data preprocessing, embedding generation, storage, and query processing. Each stage can be optimized for different trade-offs between latency, throughput, and cost.

For production systems, consider the asynchronous generation pattern where embeddings are generated in the background and stored for later retrieval. This separates the latency-sensitive query path from the computationally expensive generation process, ensuring consistent query performance.

```python
# Python: Asynchronous embedding generation pipeline
import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
import json

class EmbeddingPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.queue_name = 'embedding:pending'
        self.results_prefix = 'embedding:results'

    async def generate_and_store(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ):
        """Generate embeddings for documents and store in Redis"""
        # Extract texts from documents
        texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)

        # Store embeddings in Redis
        pipeline = self.redis.pipeline()
        for doc_id, embedding in zip(doc_ids, all_embeddings):
            key = f"{self.results_prefix}:{doc_id}"
            # Store as compressed numpy array
            pipeline.set(key, embedding.tobytes())
            pipeline.expire(key, 86400 * 7)  # 7 day TTL
        pipeline.execute()

        return {doc_id: embedding.tolist() for doc_id, embedding in zip(doc_ids, all_embeddings)}

    async def get_embedding(self, doc_id: str) -> np.ndarray:
        """Retrieve stored embedding"""
        key = f"{self.results_prefix}:{doc_id}"
        data = self.redis.get(key)
        if data:
            return np.frombuffer(data, dtype=np.float32)
        return None

# Usage example
pipeline = EmbeddingPipeline()
documents = [
    {'id': 'doc1', 'text': 'First document content'},
    {'id': 'doc2', 'text': 'Second document content'}
]
embeddings = await pipeline.generate_and_store(documents)
```

### Efficient Retrieval Patterns

Retrieving embeddings efficiently involves more than just fast vector search. The retrieval pattern must also consider metadata filtering, result ranking, and integration with downstream applications. Many production systems combine vector search with traditional database queries to provide rich, filtered results.

```python
# Python: Hybrid retrieval with metadata filtering
from pinecone import Pinecone
import psycopg2

class HybridRetriever:
    def __init__(self):
        self.pc = Pinecone(api_key="your-api-key")
        self.vector_index = self.pc.Index("documents")
        self.conn = psycopg2.connect("dbname=mydb user=postgres")

    def semantic_search_with_filter(
        self,
        query: str,
        category: str = None,
        date_from: str = None,
        top_k: int = 10
    ):
        # Generate query embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query).tolist()

        # Build metadata filter
        filter_conditions = []
        if category:
            filter_conditions.append({"category": {"$eq": category}})
        if date_from:
            filter_conditions.append({"created_at": {"$gte": date_from}})

        filter_dict = {"$and": filter_conditions} if filter_conditions else None

        # Execute vector search with filter
        results = self.vector_index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
            include_values=False
        )

        # Enrich with full document data from PostgreSQL
        doc_ids = [match['id'] for match in results['matches']]
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, content, author, created_at
                FROM documents
                WHERE id = ANY(%s)
                """,
                (doc_ids,)
            )
            doc_lookup = {row[0]: row[1:] for row in cur.fetchall()}

        # Combine results
        enriched_results = []
        for match in results['matches']:
            doc_id = match['id']
            if doc_id in doc_lookup:
                enriched_results.append({
                    'id': doc_id,
                    'score': match['score'],
                    'title': doc_lookup[doc_id][0],
                    'content': doc_lookup[doc_id][1],
                    'author': doc_lookup[doc_id][2],
                    'created_at': doc_lookup[doc_id][3]
                })

        return enriched_results
```

### Storage Optimization Strategies

Storing embeddings efficiently is crucial for managing costs and performance at scale. A 384-dimensional float32 embedding requires 1,536 bytes per vector, which becomes significant when storing millions or billions of embeddings. Several strategies can reduce storage requirements while maintaining search quality.

**Quantization** reduces the precision of floating-point numbers, dramatically decreasing storage requirements with minimal impact on search quality. Half-precision floats (FP16) use 2 bytes instead of 4, providing a 2x reduction. Integer quantization can provide even greater savings with acceptable accuracy loss.

```python
# Python: Embedding quantization
import numpy as np

def quantize_embedding(embedding: np.ndarray, bits: int = 8) -> np.ndarray:
    """Quantize embedding to reduce storage size"""
    if bits == 8:
        # Convert to unsigned 8-bit integer
        # Normalize to [0, 255] range
        min_val = embedding.min()
        max_val = embedding.max()
        normalized = (embedding - min_val) / (max_val - min_val)
        quantized = (normalized * 255).astype(np.uint8)
        return quantized, (min_val, max_val)
    elif bits == 16:
        # Half-precision float
        return embedding.astype(np.float16), None
    return embedding, None

def dequantize_embedding(quantized: np.ndarray, params=None) -> np.ndarray:
    """Restore original embedding from quantized version"""
    if quantized.dtype == np.uint8 and params:
        min_val, max_val = params
        normalized = quantized.astype(np.float32) / 255.0
        return normalized * (max_val - min_val) + min_val
    elif quantized.dtype == np.float16:
        return quantized.astype(np.float32)
    return quantized

# Example usage
embedding = np.random.random(384).astype(np.float32)
quantized, params = quantize_embedding(embedding, bits=8)

print(f"Original size: {embedding.nbytes} bytes")
print(f"Quantized size: {quantized.nbytes} bytes")
print(f"Compression ratio: {embedding.nbytes / quantized.nbytes:.1f}x")
```

**Tiered storage** keeps recent, frequently accessed embeddings in fast storage while moving older, less-accessed embeddings to cheaper storage. This approach balances performance with cost for large-scale systems.

```sql
-- PostgreSQL: Tiered storage with table partitioning
CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for different time periods
CREATE TABLE embeddings_2024_q1 PARTITION OF embeddings
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE embeddings_2024_q2 PARTITION OF embeddings
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

CREATE TABLE embeddings_2024_q3 PARTITION OF embeddings
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');

-- Default partition for older data
CREATE TABLE embeddings_archive PARTITION OF embeddings
    DEFAULT;

-- Create indexes appropriate for each tier
-- Recent data: optimize for speed
CREATE INDEX ON embeddings_2024_q1 USING ivfflat (embedding vector_cosine_ops);

-- Archive: can use slower, compressed storage
CREATE INDEX ON embeddings_archive USING ivfflat (embedding vector_cosine_ops);
```

---

## 3. Hybrid Search Strategies

### Combining Vector and Keyword Search

Pure semantic search using embeddings excels at finding conceptually related results but may miss exact keyword matches. Hybrid search combines vector similarity with traditional keyword matching (BM25 or TF-IDF) to provide both semantic understanding and precise term matching. This combination typically outperforms either approach alone for most search use cases.

The key challenge in hybrid search is effectively combining the relevance scores from different retrieval methods, which operate on different scales and capture different aspects of relevance. Score normalization and weighted combination are essential for achieving good results.

```python
# Python: Implementing hybrid search
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import redis
import json

class HybridSearchEngine:
    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.redis = redis.Redis(host='localhost', port=6379, db=0)

    def search(
        self,
        query: str,
        top_k: int = 10,
        category: str = None
    ) -> List[Dict[str, Any]]:
        # Execute searches in parallel
        vector_results = self._semantic_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)

        # Combine and rank results
        combined_scores = self._combine_scores(
            vector_results,
            keyword_results,
            top_k
        )

        return combined_scores[:top_k]

    def _semantic_search(self, query: str, top_k: int) -> Dict[str, float]:
        """Execute vector similarity search"""
        query_embedding = self.model.encode(query).tolist()

        # Search Pinecone (simplified)
        # In production, use actual Pinecone client
        results = self.redis.zadd(
            'vector_index',
            {doc_id: score for doc_id, score in vector_results}
        )

        return {doc_id: 1.0 - rank / top_k for rank, doc_id in enumerate(doc_ids)}

    def _keyword_search(self, query: str, top_k: int) -> Dict[str, float]:
        """Execute keyword search using Redisearch"""
        # Using RediSearch for keyword search
        results = self.redis.ft('idx:documents').search(
            query,
            {
                'limit': ('0', top_k)
            }
        )

        # Normalize scores to [0, 1]
        if not results.docs:
            return {}

        max_score = max(float(d.score) for d in results.docs)
        return {
            d.id: float(d.score) / max_score
            for d in results.docs
        }

    def _combine_scores(
        self,
        vector_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine scores from both methods"""
        # Get union of document IDs
        all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())

        # Calculate combined scores
        combined = []
        for doc_id in all_doc_ids:
            vec_score = vector_scores.get(doc_id, 0)
            key_score = keyword_scores.get(doc_id, 0)

            # Weighted combination
            combined_score = (
                self.vector_weight * vec_score +
                self.keyword_weight * key_score
            )
            combined.append((doc_id, combined_score))

        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)

        return [
            {'id': doc_id, 'score': score}
            for doc_id, score in combined[:top_k]
        ]
```

### RRF (Reciprocal Rank Fusion)

Reciprocal Rank Fusion is a technique for combining multiple ranked lists into a single unified ranking. It works by converting ranks to scores using the reciprocal of the rank, then summing these scores across all retrieval methods. RRF is particularly effective because it doesn't require score calibration between different retrieval methods.

```python
# Python: Reciprocal Rank Fusion implementation
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    retrieval_lists: List[List[str]],
    k: int = 60
) -> List[str]:
    """
    Combine multiple ranked lists using RRF

    Args:
        retrieval_lists: List of ranked document lists
        k: Fusion parameter (higher = more weight to lower ranks)

    Returns:
        Fused ranking
    """
    scores = {}

    for retrieval_list in retrieval_lists:
        for rank, doc_id in enumerate(retrieval_list):
            # RRF score formula
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by fused score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]
```

---

## 4. Feature Store Patterns for ML Applications

### What is a Feature Store?

A feature store is a centralized repository for storing, organizing, and serving features used in machine learning models. Features are the input variables used by ML models, and they can be derived from raw data through various transformations and aggregations. Feature stores solve the problem of feature consistency between training and inference, ensuring that the same features used during training are available during prediction.

Feature stores typically provide three main capabilities:
1. **Feature definition**: Define features and their computation logic
2. **Feature materialization**: Compute and store features for offline training
3. **Online serving**: Serve features with low latency for real-time inference

```python
# Python: Feature store interface
from typing import Dict, Any, List, Optional
from datetime import datetime

class FeatureStore:
    def __init__(self):
        self.offline_store = {}  # Simulated offline store
        self.online_store = {}   # Simulated online store

    def define_feature(
        self,
        feature_name: str,
        feature_type: str,
        description: str,
        transformation: str
    ):
        """Define a feature with its computation logic"""
        self.offline_store[feature_name] = {
            'type': feature_type,
            'description': description,
            'transformation': transformation,
            'last_updated': datetime.utcnow()
        }

    def compute_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        timestamp: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Compute features for given entities and timestamp"""
        features = {}
        for entity_id in entity_ids:
            features[entity_id] = {}
            for feature_name in feature_names:
                # Simulate feature computation
                if feature_name == 'user_age':
                    features[entity_id]['user_age'] = 30
                elif feature_name == 'user_purchase_count':
                    features[entity_id]['user_purchase_count'] = 15
                elif feature_name == 'user_avg_order_value':
                    features[entity_id]['user_avg_order_value'] = 89.99
        return features

    def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for real-time inference"""
        return self.compute_features(entity_ids, feature_names, datetime.utcnow())
```

### Offline vs Online Feature Stores

Feature stores typically separate offline and online capabilities to optimize for different requirements:

**Offline feature stores** are optimized for batch processing and historical analysis. They store features in data warehouses or data lakes and support complex transformations over large datasets. Offline features are used for model training and batch predictions.

**Online feature stores** are optimized for low-latency serving and real-time inference. They store features in low-latency databases like Redis or Cassandra and support millisecond-level response times. Online features are used for real-time model predictions.

```sql
-- PostgreSQL: Offline feature store schema
CREATE TABLE feature_definitions (
    feature_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(255) NOT NULL UNIQUE,
    feature_type VARCHAR(50) NOT NULL,
    description TEXT,
    transformation_sql TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feature_values_offline (
    feature_id INT NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (feature_id) REFERENCES feature_definitions(feature_id),
    PRIMARY KEY (feature_id, entity_id, timestamp)
);

-- Redis: Online feature store (simulated with hash keys)
-- Key format: feature:{feature_name}:{entity_id}
-- Value: JSON with feature value and timestamp
```

### Feature Engineering Best Practices

Effective feature engineering is critical for ML model performance. Feature stores help standardize and automate feature engineering processes:

1. **Consistency**: Ensure the same features are used in training and inference
2. **Versioning**: Track feature versions and changes over time
3. **Monitoring**: Monitor feature distributions and detect drift
4. **Documentation**: Document feature definitions and business meaning

```python
# Python: Feature monitoring and validation
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class FeatureMonitor:
    def __init__(self):
        self.feature_stats = {}

    def update_feature_stats(
        self,
        feature_name: str,
        values: List[float],
        timestamp: datetime
    ):
        """Update statistics for a feature"""
        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                'mean': [],
                'std': [],
                'min': [],
                'max': [],
                'count': [],
                'timestamps': []
            }

        stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }

        self.feature_stats[feature_name]['mean'].append(stats['mean'])
        self.feature_stats[feature_name]['std'].append(stats['std'])
        self.feature_stats[feature_name]['min'].append(stats['min'])
        self.feature_stats[feature_name]['max'].append(stats['max'])
        self.feature_stats[feature_name]['count'].append(stats['count'])
        self.feature_stats[feature_name]['timestamps'].append(timestamp)

    def detect_feature_drift(
        self,
        feature_name: str,
        current_values: List[float],
        reference_window: int = 7
    ) -> Dict[str, Any]:
        """Detect feature drift by comparing current distribution to historical"""
        if feature_name not in self.feature_stats:
            return {'drift_detected': False, 'reason': 'No historical data'}

        stats = self.feature_stats[feature_name]
        if len(stats['mean']) < reference_window:
            return {'drift_detected': False, 'reason': 'Insufficient history'}

        # Get reference statistics (last 7 days)
        ref_mean = np.mean(stats['mean'][-reference_window:])
        ref_std = np.mean(stats['std'][-reference_window:])

        current_mean = np.mean(current_values)
        current_std = np.std(current_values)

        # Simple z-score based drift detection
        mean_z = abs(current_mean - ref_mean) / (ref_std + 1e-8)
        std_z = abs(current_std - ref_std) / (ref_std + 1e-8)

        drift_threshold = 2.0
        drift_detected = mean_z > drift_threshold or std_z > drift_threshold

        return {
            'drift_detected': drift_detected,
            'mean_z_score': mean_z,
            'std_z_score': std_z,
            'current_mean': current_mean,
            'reference_mean': ref_mean,
            'current_std': current_std,
            'reference_std': ref_std
        }
```

---

## 5. ML Model Metadata Storage Systems

### Model Registry Patterns

A model registry is a centralized system for storing and managing machine learning models, their metadata, and associated artifacts. Model registries enable reproducibility, version control, and governance for ML models in production.

Key components of a model registry:
- **Model metadata**: Name, version, description, owner, creation date
- **Model artifacts**: Serialized model files, configuration files
- **Training metrics**: Accuracy, precision, recall, loss, etc.
- **Input/output schemas**: Expected input formats and output formats
- **Deployment information**: Where and how the model is deployed

```sql
-- PostgreSQL: Model registry schema
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    owner VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'staging',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE TABLE model_artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    artifact_path TEXT NOT NULL,
    checksum VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

CREATE TABLE model_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    metric_type VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

CREATE TABLE model_schemas (
    schema_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,
    schema_type VARCHAR(20) NOT NULL, -- 'input' or 'output'
    schema_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);
```

### Experiment Tracking Systems

Experiment tracking systems record the parameters, code, and results of ML experiments, enabling reproducibility, comparison, and collaboration. These systems track hyperparameters, training metrics, and other metadata to help teams understand what worked and why.

```python
# Python: Experiment tracking interface
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

class ExperimentTracker:
    def __init__(self):
        self.experiments = {}

    def start_experiment(
        self,
        experiment_name: str,
        tags: Dict[str, str] = None,
        description: str = ""
    ) -> str:
        """Start a new experiment and return experiment ID"""
        experiment_id = str(uuid.uuid4())
        self.experiments[experiment_id] = {
            'name': experiment_name,
            'tags': tags or {},
            'description': description,
            'start_time': datetime.utcnow(),
            'status': 'running',
            'metrics': {},
            'params': {},
            'artifacts': []
        }
        return experiment_id

    def log_param(
        self,
        experiment_id: str,
        param_name: str,
        param_value: Any
    ):
        """Log a parameter for an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['params'][param_name] = param_value

    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        metric_value: float,
        step: int = 0
    ):
        """Log a metric for an experiment"""
        if experiment_id in self.experiments:
            if metric_name not in self.experiments[experiment_id]['metrics']:
                self.experiments[experiment_id]['metrics'][metric_name] = []
            self.experiments[experiment_id]['metrics'][metric_name].append({
                'value': metric_value,
                'step': step,
                'timestamp': datetime.utcnow()
            })

    def log_artifact(
        self,
        experiment_id: str,
        artifact_name: str,
        artifact_path: str
    ):
        """Log an artifact for an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['artifacts'].append({
                'name': artifact_name,
                'path': artifact_path,
                'timestamp': datetime.utcnow()
            })

    def end_experiment(self, experiment_id: str, status: str = 'completed'):
        """End an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['end_time'] = datetime.utcnow()
            self.experiments[experiment_id]['status'] = status
```

### Model Governance and Compliance

As ML models move into production, governance and compliance become critical concerns. Model registries should support:
- **Audit trails**: Track who made changes and when
- **Access controls**: Restrict access based on roles and responsibilities
- **Compliance reporting**: Generate reports for regulatory requirements
- **Model lineage**: Track dependencies between models, data, and code

```sql
-- PostgreSQL: Audit trail for model registry
CREATE TABLE model_audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

-- Example audit log entry
INSERT INTO model_audit_logs (model_id, action, user_id, details)
VALUES (
    'model-123',
    'MODEL_DEPLOYED',
    'john.doe@example.com',
    '{"environment": "production", "deployment_method": "kubernetes", "version": "1.2.3"}'::jsonb
);
```

---

## Related Resources

- For vector databases, see [Vector Databases](../03_advanced/01_ai_ml_integration/01_vector_databases.md)
- For RAG systems, see [RAG Systems](../03_advanced/01_ai_ml_integration/02_rag_systems.md)
- For embedding storage, see [Embedding Storage](../03_advanced/01_ai_ml_integration/03_embedding_storage.md)
- For feature stores, see [Feature Stores](../03_advanced/02_specialized_databases/01_feature_stores.md)
- For model registry, see [Model Registry](../04_production/02_operations/02_model_registry.md)