---

# Case Study 23: Vector Database Implementation at Notion AI - Weaviate for Semantic Search

## Executive Summary

**Problem**: Enable semantic search across millions of user documents with high recall and low latency for AI-powered knowledge retrieval.

**Solution**: Implemented Weaviate vector database with hybrid search (vector + keyword + metadata filtering) for embedding-based page retrieval, integrated with Notion's document storage system.

**Impact**: Achieved 95%+ recall@5, sub-100ms response times, and enabled AI-powered search that understands context and meaning rather than just keywords.

**System design snapshot**:
- SLOs: p99 <100ms; recall@5 >95%; 99.99% availability
- Scale: 10M+ documents, 1M+ users, 50K+ queries/sec at peak
- Cost efficiency: Optimized HNSW parameters for memory vs performance trade-off
- Data quality: Automated embedding updates and drift detection
- Reliability: Multi-region deployment with automatic failover

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PostgreSQL     │    │   Weaviate      │    │   Redis Cache    │
│  • Document metadata│  • Vector embeddings│  • Query results  │
│  • User permissions│  • Hybrid search   │  • Session state   │
└────────┬──────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                         │                           │
         └───────────┬─────────────┴───────────┬───────────────┘
                     │                         │
             ┌───────▼─────────┐     ┌─────────▼─────────┐
             │   Embedding Model│     │   API Gateway    │
             │  • all-MiniLM-L6 │     │  • Rate limiting │
             └─────────────────┘     └─────────────────────┘
```

## Implementation Details

### Weaviate Configuration
- **HNSW algorithm**: `m=16`, `ef_construction=100` for optimal recall/latency trade-off
- **Quantization**: PQ (Product Quantization) for memory efficiency
- **Hybrid search**: Vector similarity + keyword search + metadata filtering
- **Sharding**: Automatic sharding for horizontal scalability

### Embedding Strategy
- **Model**: `all-MiniLM-L6-v2` for cost-effective embeddings
- **Chunking**: Hierarchical chunking (document → section → paragraph)
- **Update strategy**: Incremental updates for new content, full re-embedding for major changes
- **Drift detection**: Monitor embedding distribution changes

### Integration with Notion
- **Real-time indexing**: Webhooks trigger embedding updates on document changes
- **Permission handling**: PostgreSQL permissions enforced before vector search
- **Fallback mechanism**: Keyword search when vector search fails
- **Query expansion**: HyDE (Hypothetical Document Embeddings) for better recall

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Recall@5 | 95%+ | >90% |
| P99 Latency | <100ms | <120ms |
| Throughput | 50K+ QPS | 40K QPS |
| Memory Usage | 3.8GB per 1M vectors | <4GB |
| Update Latency | <2s per document | <5s |

## Key Lessons Learned

1. **Hybrid search** is essential for production RAG systems - pure vector search has limitations
2. **Parameter tuning** (HNSW `m`, `ef_construction`) significantly impacts recall/latency trade-offs
3. **Real-time indexing** is critical for user experience in collaborative applications
4. **Permission integration** must happen before vector search to maintain security
5. **Fallback mechanisms** ensure reliability when vector search fails

## Optimization Techniques

- **Quantization**: Reduced memory usage by 60% with minimal recall impact
- **Caching**: Redis cache for frequent queries reduced database load by 70%
- **Batch processing**: Async embedding generation for background updates
- **Query optimization**: Pre-filtering by metadata before vector search

> 💡 **Pro Tip**: For enterprise RAG systems, always implement comprehensive monitoring of recall@k, latency percentiles, and embedding drift. These metrics are more important than raw throughput numbers.