# Vector Databases

Vector databases are specialized database systems designed for storing and searching high-dimensional vector embeddings, essential for AI/ML applications like semantic search, similarity matching, and retrieval-augmented generation (RAG).

## Overview

Vector databases optimize for approximate nearest neighbor (ANN) search in high-dimensional spaces, where traditional databases struggle with performance. For senior AI/ML engineers, understanding vector databases is critical for building modern AI applications.

## Core Concepts

### Vector Embeddings
- **Definition**: Numerical representations of data (text, images, audio)
- **Dimensions**: Typically 1536-4096 dimensions for modern models
- **Similarity metrics**: Cosine, Euclidean, Manhattan distance
- **Use cases**: Semantic search, recommendation systems, RAG

### Approximate Nearest Neighbor (ANN) Search
- **Challenge**: Exact search is O(n) - too slow for large datasets
- **Solution**: Approximate algorithms trade small accuracy loss for massive speed gains
- **Key techniques**: Indexing, quantization, graph-based search

## Vector Database Architectures

### IVF (Inverted File)
- **Method**: Partition vectors into clusters (inverted index)
- **Search**: Find nearest clusters, then search within clusters
- **Advantages**: Good balance of speed and accuracy
- **Disadvantages**: Requires parameter tuning (number of clusters)

### HNSW (Hierarchical Navigable Small World)
- **Method**: Graph-based indexing with hierarchical layers
- **Search**: Greedy navigation through graph
- **Advantages**: Excellent recall at reasonable speed
- **Disadvantages**: Memory-intensive, slower build time

### PQ (Product Quantization)
- **Method**: Compress vectors by dividing into subvectors and quantizing
- **Search**: Approximate search with compression
- **Advantages**: Memory efficiency, good for large-scale systems
- **Disadvantages**: Accuracy trade-off, complex implementation

## Implementation Examples

### pgvector (PostgreSQL Extension)
```sql
-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    embedding vector(1536)  -- OpenAI embeddings are 1536 dimensions
);

-- Create indexes for fast similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 100);

-- Query for similar documents
SELECT id, title,
       1 - (embedding <=> $1) AS similarity
FROM documents
ORDER BY embedding <=> $1
LIMIT 5;
```

### Milvus (Open-Source)
```python
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]
schema = CollectionSchema(fields, "Document collection")

# Create collection
collection = Collection("documents", schema)

# Create index
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)

# Search
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search([query_embedding], "embedding", search_params, limit=5)
```

### Pinecone (Managed Service)
```python
from pinecone import Pinecone

# Initialize client
pc = Pinecone(api_key="your-api-key")
index = pc.Index("documents")

# Upsert vectors with metadata
vectors = [
    {
        "id": "doc1",
        "values": [0.1, 0.3, 0.5, ...],  # 1536-dimensional embedding
        "metadata": {"title": "Introduction to Databases", "category": "technology"}
    },
    {
        "id": "doc2",
        "values": [0.2, 0.4, 0.6, ...],
        "metadata": {"title": "Machine Learning Basics", "category": "ai"}
    }
]
index.upsert(vectors=vectors)

# Query for similar documents
results = index.query(
    vector=[0.1, 0.3, 0.5, ...],
    top_k=3,
    include_metadata=True,
    filter={"category": {"$eq": "technology"}}
)
```

## AI/ML Integration Patterns

### RAG (Retrieval-Augmented Generation)
- **Architecture**: LLM + Vector DB + Prompt engineering
- **Workflow**: 
  1. User query → embedding
  2. Retrieve relevant documents from vector DB
  3. Inject retrieved context into LLM prompt
  4. Generate response with context

### Recommendation Systems
- **Content-based**: Similar items based on embeddings
- **Collaborative filtering**: User-item interaction embeddings
- **Hybrid approaches**: Combine multiple embedding sources

### Anomaly Detection
- **Embedding similarity**: Detect outliers in embedding space
- **Time-series patterns**: Embed time-series data for anomaly detection
- **Multi-modal**: Combine text, image, and structured data embeddings

## Performance Optimization

### Index Tuning Parameters
- **IVF**: `nlist` (number of clusters), `nprobe` (clusters to search)
- **HNSW**: `M` (neighbors per node), `ef_construction` (build quality), `ef_search` (search quality)
- **PQ**: `m` (subvectors), `nbits` (bits per subvector)

### Hybrid Search Strategies
- **Filter + Vector**: Apply metadata filters before vector search
- **Multi-stage**: Coarse search → fine-grained refinement
- **Ensemble**: Combine results from multiple indexes

```sql
-- Hybrid search example
SELECT id, title, similarity
FROM (
    SELECT id, title,
           1 - (embedding <=> $1) AS similarity,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $1) as rn
    FROM documents
    WHERE category = 'technology' AND published_date > '2024-01-01'
) ranked
WHERE rn <= 10;
```

## Scalability Considerations

### Horizontal Scaling
- **Sharding**: Partition vectors by metadata or hash
- **Replication**: Read replicas for query scaling
- **Distributed architectures**: Clustered vector databases

### Memory Management
- **Quantization**: Reduce memory footprint (PQ, SQ)
- **Compression**: Lossy compression for embeddings
- **Caching**: Frequently accessed vectors in memory

## Best Practices

1. **Start simple**: Begin with IVF, optimize later
2. **Tune parameters**: Test different index configurations
3. **Monitor recall**: Balance speed vs accuracy
4. **Consider hybrid approaches**: Combine vector search with traditional filtering
5. **Plan for growth**: Choose scalable architecture early
6. **Test with real data**: Synthetic data may not reflect production patterns

## Related Resources

- [Indexing Fundamentals] - Basic indexing concepts
- [Query Optimization] - How vector search affects query performance
- [AI/ML System Design] - Vector databases in ML system architecture
- [RAG Systems] - Comprehensive RAG implementation guide