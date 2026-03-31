# Qdrant Tutorial for Vector-Based RAG Systems

This tutorial provides hands-on Qdrant fundamentals specifically designed for AI/ML engineers building retrieval-augmented generation (RAG) systems, semantic search, and vector-based recommendation engines.

## Why Qdrant for AI/ML Applications?

Qdrant is a modern, Rust-based vector database optimized for production AI workloads:
- **High performance**: Optimized for low-latency vector search
- **Production-ready**: Built-in replication, sharding, and monitoring
- **Rich feature set**: HNSW, quantization, payload filtering, hybrid search
- **Cloud-native**: Kubernetes-friendly, gRPC and HTTP APIs
- **Open source**: MIT license, active community

## Setting Up Qdrant for ML Workflows

### Installation Options
```bash
# Docker (recommended for development)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest

# With persistent storage
docker run -d \
  --name qdrant-prod \
  -v /data/qdrant:/qdrant/storage \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest \
  --storage-type disk \
  --host 0.0.0.0

# Using Docker Compose (recommended for production)
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__STORAGE__TYPE=disk
      - QDRANT__SERVICE__HOST=0.0.0.0
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_data:
```

### Essential Configuration for ML Workloads
```yaml
# config.yaml for production Qdrant
storage:
  type: disk
  path: "/qdrant/storage"
  optimizers:
    max_segment_size: 1000000  # 1M points per segment
    memmap_threshold: 100000   # Use memory-mapped files for large segments

service:
  host: "0.0.0.0"
  http_port: 6333
  grpc_port: 6334
  telemetry: true

cluster:
  enabled: false  # Enable for distributed setup
  p2p:
    port: 6335
    advertised_host: "qdrant-node-1"

# Performance tuning
optimizers:
  default_segment_number: 2
  max_optimization_threads: 4
  flush_interval_sec: 5
  indexing_threshold: 10000
```

## Core Qdrant Concepts for ML Engineers

### Collections and Vectors

#### Collection Structure
```json
{
  "name": "document_embeddings",
  "vectors": {
    "size": 768,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "quantile": 0.99,
      "always_ram": true
    }
  },
  "payload_schema": {
    "text": {"type": "text"},
    "source": {"type": "keyword"},
    "timestamp": {"type": "datetime"},
    "chunk_id": {"type": "integer"}
  }
}
```

### Vector Data Model for RAG Systems
```json
{
  "id": "doc_123_chunk_456",
  "vector": [0.12, -0.34, 0.56, ...],  // 768-dim embedding
  "payload": {
    "text": "The quick brown fox jumps over the lazy dog",
    "source": "wikipedia_article",
    "document_id": "wiki_12345",
    "chunk_id": 456,
    "timestamp": "2026-02-15T10:30:00Z",
    "metadata": {
      "author": "John Doe",
      "word_count": 12,
      "section": "Introduction"
    }
  }
}
```

## Qdrant Operations for ML Workflows

### Collection Management
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Initialize client
client = QdrantClient("localhost", port=6333)

# Create collection for document embeddings
client.create_collection(
    collection_name="document_embeddings",
    vectors=VectorParams(size=768, distance=Distance.COSINE),
    hnsw_config={
        "m": 16,
        "ef_construct": 100
    },
    quantization_config={
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True
        }
    }
)

# Check collection info
collection_info = client.get_collection("document_embeddings")
print(collection_info)
```

### Data Ingestion Patterns

#### Batch Insert for Document Processing
```python
import uuid
from qdrant_client.models import PointStruct

# Prepare points for batch insertion
points = []
for i, (text, embedding, metadata) in enumerate(document_chunks):
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,  # List[float] of 768 dimensions
            payload={
                "text": text,
                "source": metadata["source"],
                "document_id": metadata["document_id"],
                "chunk_id": i,
                "timestamp": metadata["timestamp"],
                "word_count": len(text.split()),
                "section": metadata.get("section", "unknown")
            }
        )
    )

# Insert in batches (recommended for large datasets)
batch_size = 1000
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    client.upsert(
        collection_name="document_embeddings",
        points=batch
    )
```

#### Real-Time Ingestion for Streaming Data
```python
# For real-time document processing
def ingest_document实时(document_id, chunks, embeddings):
    points = []
    for i, (text, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=f"{document_id}_chunk_{i}",
                vector=embedding,
                payload={
                    "text": text,
                    "document_id": document_id,
                    "chunk_id": i,
                    "timestamp": datetime.utcnow().isoformat(),
                    "processed_at": datetime.utcnow().isoformat()
                }
            )
        )
    
    # Upsert with overwrite (idempotent)
    client.upsert(
        collection_name="document_embeddings",
        points=points,
        wait=True  # Wait for operation to complete
    )
```

## Advanced Query Patterns for RAG Systems

### Basic Vector Search
```python
# Simple nearest neighbor search
search_result = client.search(
    collection_name="document_embeddings",
    query_vector=[0.15, -0.32, 0.58, ...],  # query embedding
    limit=5,
    with_payload=True,
    with_vectors=False
)

# Print results
for hit in search_result:
    print(f"Score: {hit.score:.4f}")
    print(f"Text: {hit.payload['text'][:100]}...")
    print(f"Source: {hit.payload['source']}")
    print("---")
```

### Filtered Vector Search
```python
# Search with payload filters
search_result = client.search(
    collection_name="document_embeddings",
    query_vector=query_embedding,
    query_filter={
        "must": [
            {
                "key": "source",
                "match": {"value": "wikipedia_article"}
            },
            {
                "key": "timestamp",
                "range": {"gte": "2026-01-01T00:00:00Z"}
            },
            {
                "key": "word_count",
                "range": {"gte": 50, "lte": 500}
            }
        ]
    },
    limit=10,
    with_payload=True
)
```

### Hybrid Search (Vector + Keyword)
```python
# Combine vector similarity with keyword relevance
from qdrant_client.models import FieldCondition, MatchText

# First, get vector search results
vector_results = client.search(
    collection_name="document_embeddings",
    query_vector=query_embedding,
    limit=20,
    with_payload=True
)

# Then filter by keyword (or use Qdrant's hybrid search if available)
keyword_filter = {
    "must": [
        {
            "key": "text",
            "match": {"text": "machine learning"}
        }
    ]
}

# Or use hybrid search (Qdrant 1.3+)
hybrid_result = client.search(
    collection_name="document_embeddings",
    query_vector=query_embedding,
    query_fusion="rrf",  # Reciprocal Rank Fusion
    limit=10,
    with_payload=True,
    with_vectors=False
)
```

### Multi-Query Search for Better Recall
```python
# Generate multiple query embeddings and combine results
query_embeddings = [
    generate_embedding(query_text),
    generate_embedding(f"what is {query_text}"),
    generate_embedding(f"{query_text} definition")
]

all_results = []
for emb, weight in zip(query_embeddings, [1.0, 0.8, 0.6]):
    results = client.search(
        collection_name="document_embeddings",
        query_vector=emb,
        limit=10,
        with_payload=True
    )
    # Add weight to scores
    for result in results:
        result.score *= weight
    all_results.extend(results)

# Merge and deduplicate results
merged_results = {}
for result in all_results:
    if result.id not in merged_results:
        merged_results[result.id] = result
    else:
        merged_results[result.id].score = max(merged_results[result.id].score, result.score)

# Sort by score
final_results = sorted(merged_results.values(), key=lambda x: x.score, reverse=True)[:10]
```

## Performance Optimization for ML Workloads

### Index Configuration Tuning
```python
# Optimize HNSW parameters for your dataset size
def get_optimal_hnsw_params(vector_count):
    if vector_count < 100_000:
        return {"m": 16, "ef_construct": 100, "ef_search": 100}
    elif vector_count < 1_000_000:
        return {"m": 32, "ef_construct": 100, "ef_search": 100}
    elif vector_count < 10_000_000:
        return {"m": 64, "ef_construct": 100, "ef_search": 100}
    else:
        return {"m": 128, "ef_construct": 200, "ef_search": 200}

# Update collection configuration
client.update_collection(
    collection_name="document_embeddings",
    hnsw_config=get_optimal_hnsw_params(5_000_000)
)
```

### Quantization for Memory Efficiency
```python
# Enable scalar quantization for memory savings
client.update_collection(
    collection_name="document_embeddings",
    quantization_config={
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True
        }
    }
)

# Or product quantization for very large datasets
client.update_collection(
    collection_name="document_embeddings",
    quantization_config={
        "product": {
            "compression_ratio": "x16",
            "always_ram": False
        }
    }
)
```

### Sharding and Replication
```yaml
# For distributed setup (config.yaml)
cluster:
  enabled: true
  p2p:
    port: 6335
    advertised_host: "qdrant-node-1"
  consensus:
    thread_pool_size: 8
    max_message_queue_size: 10000

# Create sharded collection
client.create_collection(
    collection_name="large_document_embeddings",
    vectors=VectorParams(size=768, distance=Distance.COSINE),
    shard_number=3,  # 3 shards
    replication_factor=2  # 2 replicas
)
```

## Qdrant for Production RAG Systems

### RAG Pipeline Implementation
```python
class RAGSystem:
    def __init__(self, qdrant_client, llm_client):
        self.qdrant = qdrant_client
        self.llm = llm_client
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for RAG"""
        query_vector = self._embed_query(query)
        
        results = self.qdrant.search(
            collection_name="document_embeddings",
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Extract context
        contexts = []
        for hit in results:
            contexts.append({
                "text": hit.payload["text"],
                "source": hit.payload["source"],
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id")
            })
        
        return contexts
    
    def generate_response(self, query: str, contexts: List[Dict]) -> str:
        """Generate response using LLM with retrieved context"""
        # Format context for prompt
        context_text = "\n\n".join([
            f"Source: {ctx['source']}\nContent: {ctx['text']}"
            for ctx in contexts
        ])
        
        prompt = f"""Answer the question based on the following context:

{context_text}

Question: {query}
Answer:"""
        
        return self.llm.generate(prompt)
    
    def rag_query(self, query: str) -> Dict:
        """Complete RAG pipeline"""
        contexts = self.retrieve_context(query)
        response = self.generate_response(query, contexts)
        
        return {
            "query": query,
            "response": response,
            "retrieved_contexts": contexts,
            "latency_ms": self._measure_latency()
        }
```

### Monitoring and Observability
```python
# Enable metrics collection
# Qdrant exposes Prometheus metrics at /metrics

# Custom monitoring
import time
from typing import Callable

def timed_operation(operation: Callable, operation_name: str):
    start_time = time.time()
    try:
        result = operation()
        duration = time.time() - start_time
        # Log metrics
        print(f"{operation_name} completed in {duration:.2f}s")
        return result
    except Exception as e:
        duration = time.time() - start_time
        print(f"{operation_name} failed after {duration:.2f}s: {e}")
        raise

# Example usage
def search_with_monitoring(query_vector):
    return timed_operation(
        lambda: client.search(
            collection_name="document_embeddings",
            query_vector=query_vector,
            limit=5
        ),
        "vector_search"
    )
```

## Common Qdrant Pitfalls for ML Engineers

### 1. Vector Dimension Mismatch
- **Problem**: Query vector dimension doesn't match collection
- **Solution**: Validate dimensions before search
- **Example**: Collection: 768-dim, Query: 1536-dim → error

### 2. Over-quantization
- **Problem**: Aggressive quantization reducing search quality
- **Solution**: Test recall@k with different quantization levels
- **Best practice**: Start with int8 quantization, test impact

### 3. Poor Filter Design
- **Problem**: Complex filters slowing down search
- **Solution**: Use simple filters, avoid nested conditions
- **Optimization**: Create index on frequently filtered fields

### 4. Ignoring Collection Size
- **Problem**: Using default HNSW parameters for large datasets
- **Solution**: Tune M and ef_search based on dataset size
- **Rule of thumb**: Larger datasets need higher M values

## Visual Diagrams

### Qdrant Architecture for RAG Systems
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Embedding      │───▶│   Qdrant        │
│  (Natural Lang) │    │  Model (e.g.,   │    │  (Vector DB)    │
└─────────────────┘    │  text-embedding)│    └────────┬────────┘
                       └─────────────────┘             │
                                                        ▼
                                            ┌─────────────────────┐
                                            │  Retrieval Results  │
                                            │  • Top-K documents  │
                                            │  • Similarity scores│
                                            │  • Metadata         │
                                            └────────┬────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │  LLM Prompt Engineering │
                                        │  • Context formatting   │
                                        │  • Instruction tuning   │
                                        └─────────────────────────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │     Generated Response  │
                                        └─────────────────────────┘
```

### Qdrant Data Flow
```
Document Processing Pipeline:
Raw Documents → [Chunking] → [Embedding] → Qdrant (Ingestion)
       ↑                   │                      │
       │                   ▼                      ▼
[Metadata Extraction] ← [Payload Enrichment] ← [Vector Storage]

Query Processing Pipeline:
User Query → [Embedding] → Qdrant (Search) → [Results] → LLM
       ↑                   │                      │
       └── [Hybrid Search] ←── [Filtering] ←── [Re-ranking]
```

## Hands-on Exercises

### Exercise 1: Basic RAG System
1. Create a Qdrant collection for document embeddings
2. Ingest sample documents with metadata
3. Implement vector search with filtering
4. Build simple RAG pipeline with LLM integration

### Exercise 2: Performance Tuning
1. Test different HNSW parameters (M, ef_search)
2. Compare int8 vs float32 quantization
3. Measure recall@k and latency trade-offs
4. Optimize for your specific use case

### Exercise 3: Production RAG Pipeline
1. Implement multi-query search for better recall
2. Add hybrid search with keyword filtering
3. Build monitoring and logging
4. Test with realistic query patterns

## Best Practices Summary

1. **Start with default parameters**: M=16, ef_construct=100, ef_search=100
2. **Use quantization**: int8 for 50-70% memory savings with minimal quality loss
3. **Validate vector dimensions**: Ensure query and collection dimensions match
4. **Monitor recall@k**: Track search quality as you tune parameters
5. **Use payload filtering**: Combine vector search with metadata filters
6. **Implement batching**: For large-scale ingestion
7. **Plan for scaling**: Use sharding for >10M vectors
8. **Test with real queries**: Use representative query sets for tuning

This tutorial provides the foundation for building production-grade vector search systems using Qdrant for RAG, semantic search, and AI-powered applications.