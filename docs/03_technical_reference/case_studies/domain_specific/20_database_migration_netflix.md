---

# Case Study 20: Database Migration at Netflix - Hybrid MySQL+Cassandra Architecture

## Executive Summary

**Problem**: Scale video streaming infrastructure to 250M+ users with 250K+ writes/sec for viewing history while maintaining 99.99% availability.

**Solution**: Implemented hybrid architecture with MySQL for OLTP workloads and Cassandra for high-volume write-heavy data, complemented by EVCache for caching.

**Impact**: Achieved 99.99% availability, sub-100ms response times, petabyte-scale storage capacity, and 80% reduction in MySQL load through caching.

**System design snapshot**:
- SLOs: p99 <100ms; 99.99% availability; 250K+ writes/sec for viewing history
- Scale: 250M+ users, 100PB+ data storage, 50+ Cassandra clusters with 500+ nodes
- Cost efficiency: >80% reduction in MySQL load through EVCache (Memcached-based)
- Data quality: Event sourcing via Kafka for auditability and consistency
- Reliability: Master-master synchronous replication for MySQL across 3 AZs

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MySQL (OLTP)  │    │  Cassandra (NoSQL)│    │   EVCache (Redis)│
│  • User accounts │    │  • Viewing history│    │  • >95% hit rate  │
│  • Billing       │    │  • Activity logs |    │  • Sub-ms latency │
│  • Entitlements  │    │  • 250K+ writes/sec│  │                   │
└────────┬──────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                         │                           │
         └───────────┬─────────────┴───────────┬───────────────┘
                     │                         │
             ┌───────▼─────────┐     ┌─────────▼─────────┐
             │   Kafka (Event   │     │   S3 (Cold Storage)│
             │   Streaming)    │     │  • 100PB+ historical│
             └─────────────────┘     └─────────────────────┘
```

## Implementation Details

### MySQL Configuration
- **Master-master synchronous replication** across 3 availability zones
- **Logical replication** for cross-service data sharing
- **Custom WAL processing** for atomicity across microservices
- **Partial indexes** and **BRIN indexes** for time-range queries

### Cassandra Clusters
- **50+ clusters**, **500+ nodes** total
- **Replication Factor**: RF=3 for durability
- **Consistency Level**: QUORUM for balanced performance and consistency
- **Time-series optimization**: Partition key design for efficient time-range queries

### EVCache (Memcached-based)
- **100TB+ memory capacity**
- **>95% cache hit rate**
- **Custom eviction policies** optimized for viewing history patterns
- **Sub-millisecond latency** for cached operations

## Performance Metrics

| Metric | Value | Improvement |
|--------|-------|-------------|
| Availability | 99.99% | +0.01% vs previous architecture |
| Viewing History Write Latency | <15ms P99 | 40% reduction from legacy system |
| Cache Hit Rate | >95% | Reduced MySQL load by 80% |
| Data Volume | 100PB+ | Scales linearly with subscriber growth |
| Throughput | 250K+ writes/sec | Handles peak traffic during major events |

## Key Lessons Learned

1. **Separate concerns by access pattern**, not just data type
2. **Read-write separation** is critical for high-throughput systems
3. **Event sourcing** enables eventual consistency without losing auditability
4. **Multi-layer caching** (application + database + CDN) provides compounding benefits
5. **Horizontal scaling** of NoSQL complements vertical scaling of RDBMS

## Migration Strategy

- **Strangler Fig pattern**: Gradual replacement of legacy functionality
- **Dual-write validation**: Ensured data consistency during transition
- **Canary releases**: 5% → 25% → 50% → 100% traffic routing
- **Comprehensive monitoring**: Real-time metrics for latency, error rates, and consistency

> 💡 **Pro Tip**: Always document your database decisions in a `DATABASE_DECISION_LOG.md` with rationale, trade-offs, and validation metrics.