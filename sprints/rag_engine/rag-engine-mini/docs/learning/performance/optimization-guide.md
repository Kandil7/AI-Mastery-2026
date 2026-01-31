# Performance & Optimization Guide
# ==================================

## Overview

Guide for optimizing RAG Engine performance.

## Metrics to Monitor

- **P50 Latency**: < 500ms for queries
- **P95 Latency**: < 2s for queries
- **P99 Latency**: < 5s for queries
- **Cache Hit Rate**: > 50%
- **Error Rate**: < 1%
- **Token Usage**: Monitor costs

## Optimization Strategies

### 1. Caching

- Embedding cache with TTL
- Query response cache
- Document metadata cache

### 2. Database

- Connection pooling
- Query optimization (indexes)
- Read replicas

### 3. Search

- Vector search with HNSW
- Hybrid search (FTS + vector)
- Reranking top results

### 4. LLM

- Use appropriate model (gpt-4o-mini for speed)
- Stream long responses
- Batch when possible

## Load Testing

```bash
# Install Locust
pip install locust

# Run load test
locust -f loadtest.py --host http://localhost:8000 --users 100 --spawn-rate 10 --run-time 60
```

## Performance Tuning

### Database

```sql
-- Add indexes
CREATE INDEX idx_documents_tenant_status ON documents(tenant_id, status);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);

-- Tune PostgreSQL
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

### Redis

```conf
# Increase max memory
maxmemory 2gb
maxmemory-policy allkeys-lru
```

---
**Document Version:** 1.0
**Last Updated:** 2026-01-31
**Author:** AI-Mastery-2026
