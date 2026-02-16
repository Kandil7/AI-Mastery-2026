---

# Case Study 21: Real-Time Ride Matching at Uber - Schemaless NoSQL Architecture

## Executive Summary

**Problem**: Process 100K+ ride requests per second with strict latency requirements (<100ms) while maintaining 99.99% reliability during peak hours.

**Solution**: Built custom Schemaless NoSQL layer on MySQL with Redis for geospatial indexing and ScyllaDB for high-throughput event logging, using shard-per-core architecture.

**Impact**: Achieved <100ms matching latency, 99.99% reliability during peak hours, and scalable infrastructure handling 100K+ TPS.

**System design snapshot**:
- SLOs: p99 <100ms; 99.99% reliability; 100K+ TPS during peak hours
- Scale: 450M+ users, 100K+ ride requests/sec at peak
- Cost efficiency: Optimal resource utilization through shard-per-core architecture
- Data quality: Real-time validation and consistency checks
- Reliability: Multi-region failover with <30s recovery time

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Schemaless NoSQL │    │    Redis Cluster │    │   ScyllaDB      │
│  (MySQL-based)    │    │  • Session store │    │  • High-throughput│
│  • Driver state   │    │  • Geospatial,   │    │  • Event logging  │
│  • Trip metadata  │    │  • Rate limiting │    │                   │
└────────┬──────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                         │                           │
         └───────────┬─────────────┴───────────┬───────────────┘
                     │                         │
             ┌───────▼─────────┐     ┌─────────▼─────────┐
             │   Kafka Streams │     │   Prometheus/Grafana│
             │  • Real-time ETL │     │  • Monitoring & alerting│
             └─────────────────┘     └─────────────────────┘
```

## Implementation Details

### Schemaless NoSQL (MySQL-based)
- **Custom MySQL layer** with sharding per driver ID
- **Shard-per-core architecture**: Each CPU core handles one shard for optimal resource utilization
- **Atomic operations**: Find-and-modify pattern for driver state updates
- **Change streams**: For real-time sync across services

### Redis Cluster
- **Geo-indexing**: For real-time location queries and proximity calculations
- **Sorted sets**: For surge pricing and driver ranking
- **Rate limiting**: Token bucket algorithm for API protection
- **Session management**: TTL-based expiration for driver availability

### ScyllaDB
- **Time-series data**: For trip analytics and fraud detection
- **High-throughput writes**: Optimized for event logging at scale
- **Consistency tuning**: QUORUM consistency for critical data, ONE for analytics

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Match Latency | <100ms P99 | <150ms |
| Request Throughput | 100K+ TPS | 80K TPS |
| System Reliability | 99.99% | 99.95% |
| Failover Time | <30 seconds | <60 seconds |
| Cache Hit Rate | >92% | >90% |

## Key Lessons Learned

1. **Shard-per-core architecture** maximizes hardware utilization and reduces contention
2. **Geo-spatial indexing** in Redis enables real-time location queries at scale
3. **Custom database layers** can outperform generic solutions for specific workloads
4. **Real-time monitoring** is non-negotiable for mission-critical systems
5. **Multi-layer caching** (Redis + application cache) provides compounding benefits

## Migration Strategy

- **Blue-green deployment**: Zero downtime migration from legacy MySQL to Schemaless NoSQL
- **Feature flags**: Gradual rollout with traffic routing control
- **Shadow mode**: Run new system in parallel without serving traffic initially
- **Chaos engineering**: Test failure scenarios pre-migration

> 💡 **Pro Tip**: For real-time systems, prioritize latency over throughput when designing database architectures. The 99th percentile latency is often more important than average performance.