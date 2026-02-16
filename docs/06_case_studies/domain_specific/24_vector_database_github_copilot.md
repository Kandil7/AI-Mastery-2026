---

# Case Study 24: Hybrid Vector Database at GitHub Copilot - pgvector + Elasticsearch Integration

## Executive Summary

**Problem**: Enable code search and completion across billions of code snippets with high precision and low latency for AI-powered developer assistance.

**Solution**: Implemented hybrid architecture using pgvector for code embeddings and Elasticsearch for metadata filtering and full-text search, integrated with GitHub's existing PostgreSQL infrastructure.

**Impact**: Achieved 50K+ queries/sec with P99 <120ms, enabling real-time AI code suggestions that understand context and patterns.

**System design snapshot**:
- SLOs: p99 <120ms; precision@10 >85%; 99.99% availability
- Scale: 1B+ code snippets, 100M+ developers, 50K+ QPS at peak
- Cost efficiency: Leverage existing PostgreSQL infrastructure to minimize new infrastructure costs
- Data quality: Automated code analysis and embedding validation
- Reliability: Multi-region deployment with automatic failover

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PostgreSQL     │    │   pgvector      │    │ Elasticsearch   │
│  • Code metadata │  • Vector embeddings│  • Metadata filtering│
│  • Repository data│  • HNSW indexing  │  • Full-text search │
└────────┬──────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                         │                           │
         └───────────┬─────────────┴───────────┬───────────────┘
                     │                         │
             ┌───────▼─────────┐     ┌─────────▼─────────┐
             │   Embedding Model│     │   API Gateway    │
             │  • CodeBERT-base │     │  • Rate limiting │
             └─────────────────┘     └─────────────────────┘
```

## Implementation Details

### PostgreSQL + pgvector
- **HNSW indexing**: `m=16`, `ef_construction=100` for optimal performance
- **Vector dimensions**: 768 (CodeBERT-base embeddings)
- **Hybrid queries**: `WHERE metadata = ? AND <->` pattern for combined filtering
- **Partitioning**: Time-based partitioning for code snippet storage

### Elasticsearch Integration
- **Metadata indexing**: Repository, language, license, stars, etc.
- **Full-text search**: For keyword-based code discovery
- **Query routing**: Intelligent routing based on query type (semantic vs keyword)
- **Caching**: Redis cache for frequent queries and popular repositories

### Embedding Strategy
- **Model**: CodeBERT-base for code-specific embeddings
- **Chunking**: Function-level and file-level chunking strategies
- **Update strategy**: Incremental updates for new commits, batch re-embedding for major changes
- **Quality control**: Automated validation of embedding quality and coverage

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Precision@10 | 85%+ | >80% |
| P99 Latency | <120ms | <150ms |
| Throughput | 50K+ QPS | 40K QPS |
| Memory Usage | 5.1GB per 1M vectors | <6GB |
| Update Latency | <5s per repository | <10s |

## Key Lessons Learned

1. **Hybrid approach** leverages strengths of both relational and search databases
2. **Existing infrastructure reuse** significantly reduces migration complexity and cost
3. **Intelligent query routing** improves performance by directing queries to optimal systems
4. **Code-specific embeddings** outperform generic models for programming tasks
5. **Incremental updates** are essential for maintaining freshness without overwhelming systems

## Optimization Techniques

- **Index optimization**: Custom HNSW parameters for code embedding characteristics
- **Caching strategy**: Multi-layer caching (Redis + application cache)
- **Batch processing**: Async embedding generation for background updates
- **Query optimization**: Pre-filtering by metadata before vector search
- **Sharding**: Horizontal scaling of PostgreSQL instances for high throughput

## Challenges and Solutions

- **Challenge**: High dimensionality of code embeddings (768 vs 1536 for text)
  - **Solution**: Optimized HNSW parameters for lower-dimensional vectors

- **Challenge**: Mixed query types (semantic vs keyword)
  - **Solution**: Intelligent query parser and router

- **Challenge**: Real-time updates for rapidly changing codebase
  - **Solution**: Change data capture (CDC) with Kafka for event-driven updates

> 💡 **Pro Tip**: For code-focused RAG systems, prioritize precision over recall. Developers need highly relevant results, not just many results. The cost of irrelevant suggestions is much higher than missing some relevant ones.