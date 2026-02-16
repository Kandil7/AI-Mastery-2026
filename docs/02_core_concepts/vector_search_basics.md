# Vector Search Basics for AI/ML Engineers

This document covers the fundamentals of vector search—essential knowledge for AI/ML engineers working with embedding-based systems, RAG architectures, and similarity search applications.

## What is Vector Search?

Vector search (also called similarity search or nearest neighbor search) finds items that are most similar to a query vector in a high-dimensional space. Unlike traditional database queries that match exact values, vector search operates on mathematical similarity.

### Core Components
- **Vectors**: Numerical representations of data (embeddings)
- **Distance metrics**: Mathematical functions to measure similarity
- **Index structures**: Data structures optimized for fast approximate search
- **Search algorithms**: Methods to find nearest neighbors efficiently

## Embeddings and Vector Representations

### What are Embeddings?
Embeddings are dense vector representations that capture semantic meaning:
- **Text embeddings**: Sentence-BERT, OpenAI embeddings, etc.
- **Image embeddings**: ResNet features, CLIP embeddings
- **Audio embeddings**: Whisper, Wav2Vec features
- **Graph embeddings**: Node2Vec, GraphSAGE

### Vector Properties
- **Dimensionality**: Typically 128-1536 dimensions for modern models
- **Normalization**: Often L2-normalized for cosine similarity
- **Sparsity**: Dense vectors (most values non-zero) vs sparse vectors
- **Quantization**: Reduced precision for storage efficiency

Example embedding (simplified):
```
[0.12, -0.34, 0.56, -0.23, 0.89, ..., 0.01]  # 768 dimensions
```

## Similarity Metrics

### Distance Functions
- **Euclidean distance**: √Σ(xi - yi)²
- **Manhattan distance**: Σ|xi - yi|
- **Cosine similarity**: (x·y) / (||x|| ||y||)
- **Dot product**: x·y (for normalized vectors)

### Choosing the Right Metric
- **Cosine similarity**: Most common for text embeddings (measures angle)
- **Euclidean distance**: Better for geometric data, clustering
- **Dot product**: Fastest computation, equivalent to cosine for normalized vectors
- **Jaccard similarity**: For sparse binary vectors

### Normalization Impact
- **L2 normalization**: Converts vectors to unit length, making dot product = cosine similarity
- **Why normalize?**: Prevents magnitude bias, improves search quality
- **When not to normalize**: When magnitude carries meaningful information

## Index Structures for Vector Search

### Exact Search
- **Linear scan**: O(n) complexity, brute force
- **Only feasible** for small datasets (< 100K vectors)
- **Guarantees** exact nearest neighbors

### Approximate Nearest Neighbor (ANN) Indexes
Designed for large-scale search with acceptable accuracy trade-offs:

#### 1. Hierarchical Navigable Small World (HNSW)
- **Graph-based**: Builds multi-layer graph structure
- **Performance**: Excellent recall vs speed trade-off
- **Memory usage**: Higher than other methods
- **Parameters**: M (neighbors per node), ef_construction, ef_search

#### 2. Inverted File Index (IVF)
- **Clustering-based**: K-means clustering + inverted index
- **Performance**: Good balance of speed and memory
- **Parameters**: nlist (number of clusters), nprobe (clusters to search)

#### 3. Product Quantization (PQ)
- **Compression-based**: Divides vectors into subvectors, quantizes each
- **Performance**: Very memory efficient, good for large datasets
- **Variants**: OPQ (optimized PQ), SQ (scalar quantization)

#### 4. Annoy (Approximate Nearest Neighbors Oh Yeah)
- **Tree-based**: Random projection trees
- **Performance**: Good for moderate-sized datasets
- **Memory efficiency**: Very good

## Vector Search Workflow

### 1. Data Ingestion
```python
# Example: Adding vectors to Qdrant
points = [
    {
        "id": "doc_1",
        "vector": [0.12, -0.34, 0.56, ...],  # 768-dim embedding
        "payload": {
            "text": "The quick brown fox jumps over the lazy dog",
            "source": "document_collection",
            "timestamp": "2026-02-15T10:30:00Z"
        }
    },
    # ... more points
]

collection.upload_points(points)
```

### 2. Index Construction
- **Offline indexing**: Build index after data ingestion
- **Online indexing**: Incremental updates (more complex)
- **Parameter tuning**: Balance between recall, speed, and memory

### 3. Query Processing
```python
# Example: Searching in Qdrant
search_result = collection.query(
    query_vector=[0.15, -0.32, 0.58, ...],  # query embedding
    limit=5,
    with_payload=True,
    with_vectors=False
)
```

### 4. Result Ranking
- **Score normalization**: Convert distances to relevance scores
- **Re-ranking**: Use cross-encoders or other models for final ranking
- **Hybrid search**: Combine vector search with keyword search

## Performance Considerations for ML Workloads

### Throughput vs Latency Trade-offs
- **Batch search**: High throughput, acceptable latency (offline processing)
- **Real-time search**: Low latency, moderate throughput (RAG systems)
- **Interactive search**: Balanced requirements (exploratory analysis)

### Memory Optimization
- **Quantization**: FP16, INT8, or binary quantization
- **Compressed indexes**: PQ, SQ reduce memory footprint
- **Offloading**: Move cold data to slower storage

### Scaling Strategies
- **Sharding**: Horizontal partitioning by data subsets
- **Replication**: Read replicas for high availability
- **Hybrid approaches**: Combine different index types

## Common Vector Search Patterns in AI/ML

### Retrieval-Augmented Generation (RAG)
```
User Query → Embedding → Vector Search → Retrieved Documents → LLM Prompt → Response
```

### Recommendation Systems
- **Content-based**: Similar items based on embeddings
- **Collaborative filtering**: User-item interaction embeddings
- **Hybrid approaches**: Combine multiple signals

### Anomaly Detection
- **Outlier detection**: Find vectors far from cluster centers
- **Novelty detection**: Identify vectors dissimilar to training data
- **Drift detection**: Monitor embedding distribution changes

### Clustering and Segmentation
- **Unsupervised learning**: Group similar vectors
- **Topic modeling**: Cluster text embeddings for themes
- **Customer segmentation**: Group users by behavior embeddings

## Visual Diagrams

### Vector Search Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query Vector  │───▶│  Search Engine  │───▶│  Index Structure │
│ (embedding)     │    │ (query processor)│    │ (HNSW, IVF, etc.)│
└─────────────────┘    └────────┬────────┘    └────────┬────────┘
                                 │                         │
                                 ▼                         ▼
                      ┌─────────────────┐       ┌─────────────────┐
                      │  Candidate Set  │◀──────│   Vector Store  │
                      │ (approximate NN)│       │ (raw vectors + metadata)│
                      └────────┬────────┘       └─────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │  Re-ranking     │
                    │ (optional: cross-encoder, reranker)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Results        │
                    │ (top-k matches) │
                    └─────────────────┘
```

### HNSW Index Structure
```
Layer 3 (Sparse): o──o──o──o──o──o──o──o
                   │  │  │  │  │  │  │  │
Layer 2:          o──o──o──o──o──o──o──o
                  │╲ │╲ │╲ │╲ │╲ │╲ │╲ │╲
Layer 1:         o──o──o──o──o──o──o──o
                 │╲│╲│╲│╲│╲│╲│╲│╲│╲│╲│╲│╲│╲│╲│╲│╲
Layer 0 (Dense): o──o──o──o──o──o──o──o──o──o──o──o──o──o──o──o
```
*Each node connects to neighbors in the same layer and higher layers*

### Vector Search Quality Metrics
```
Recall@k = (Number of relevant items in top-k results) / (Total relevant items)
Precision@k = (Number of relevant items in top-k results) / k
MRR (Mean Reciprocal Rank) = Average of 1/rank for first relevant item
```

## Best Practices for AI/ML Engineers

### Index Design
1. **Start with HNSW**: Best default for most use cases
2. **Tune parameters**: M=16, ef_construction=100, ef_search=100 are good defaults
3. **Consider dataset size**: IVF for very large datasets (> 10M vectors)
4. **Test with real queries**: Use representative query sets for tuning

### Data Preparation
1. **Normalize vectors**: L2 normalization for cosine similarity
2. **Handle missing data**: Decide strategy for incomplete embeddings
3. **Version embeddings**: Track which embedding model was used
4. **Metadata design**: Include searchable fields for hybrid search

### Production Considerations
1. **Monitor recall**: Track search quality over time
2. **Handle cold starts**: Strategies for new data/indexes
3. **Graceful degradation**: Fallback mechanisms when search fails
4. **Cost optimization**: Balance between quality and computational cost

## Common Pitfalls

1. **Ignoring normalization**: Using raw embeddings without normalization
2. **Wrong distance metric**: Using Euclidean for normalized text embeddings
3. **Over-tuning indexes**: Spending too much time on marginal improvements
4. **Not monitoring drift**: Embedding model changes affecting search quality
5. **Underestimating memory**: Large vector indexes consuming excessive RAM
6. **Ignoring metadata**: Only searching on vectors, not combining with filters

## Recommended Tools for ML Workflows

- **Qdrant**: Modern, Rust-based, excellent for production RAG
- **Milvus**: Comprehensive, supports multiple index types
- **Pinecone**: Managed service, easy integration
- **Weaviate**: GraphQL interface, built-in modules
- **Chroma**: Simple, Python-native, good for prototyping

This foundation enables AI/ML engineers to implement effective vector search systems that power modern AI applications like RAG, recommendation engines, and semantic search.