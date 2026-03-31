---

# Case Study 26: Real-Time Fraud Detection System - Graph + Time-Series Database Architecture

## Executive Summary

**Problem**: Detect fraudulent transactions in real-time across 100M+ users with sub-100ms latency while maintaining 99.99% availability.

**Solution**: Implemented hybrid architecture using Neo4j for relationship analysis, TimescaleDB for time-series transaction data, and Redis for real-time scoring and caching.

**Impact**: Achieved 95% fraud detection accuracy, 85% reduction in false positives, sub-50ms P99 latency, and $25M annual savings from prevented fraud.

**System design snapshot**:
- SLOs: p99 <100ms; 99.99% availability; 95%+ fraud detection accuracy
- Scale: 100M+ users, 50K+ transactions/sec at peak, 1B+ historical transactions
- Cost efficiency: 40% reduction in infrastructure costs vs legacy system
- Data quality: Automated validation of fraud patterns and model drift
- Reliability: Multi-region deployment with automatic failover

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Neo4j Graph   │    │ TimescaleDB     │    │    Redis        │
│  • Fraud rings  │    │  • Transaction logs│    │  • Real-time scoring│
│  • Relationship │    │  • Time-series  │    │  • Session state │
│  • Path analysis│    │  • Aggregations │    │  • Rate limiting │
└────────┬──────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                         │                           │
         └───────────┬─────────────┴───────────┬───────────────┘
                     │                         │
             ┌───────▼─────────┐     ┌─────────▼─────────┐
             │   Kafka Streams │     │   ML Scoring Engine│
             │  • Event sourcing│     │  • Real-time models│
             └─────────────────┘     └─────────────────────┘
```

## Implementation Details

### Neo4j Graph Database
- **Schema design**: 
  - Nodes: `User`, `Device`, `IP`, `Account`, `Transaction`
  - Relationships: `USED_DEVICE`, `SAME_IP`, `TRANSFERRED_TO`, `RELATED_TO`
- **Query patterns**: 
  - Multi-hop path detection: `MATCH (u:User)-[:USED_DEVICE*1..3]->(d:Device)<-[:USED_DEVICE*1..3]-(u2:User)`
  - Community detection: `CALL gds.louvain.stream()` for fraud ring identification
  - Temporal queries: `WHERE t.timestamp > $window_start AND t.timestamp < $window_end`
- **Performance optimization**: 
  - Indexes on `User.id`, `Device.fingerprint`, `IP.address`
  - Materialized views for common fraud patterns
  - Query caching for frequent pattern detection

### TimescaleDB (PostgreSQL extension)
- **Hypertable design**: `transactions` table partitioned by 1-hour intervals
- **Continuous aggregates**: Pre-computed rolling windows for velocity checks
- **Compression**: Automatic compression of older data (7+ days)
- **Retention policies**: Automatic data pruning based on regulatory requirements
- **Indexing**: BRIN indexes for time-range queries, GIN for JSON metadata

### Redis Real-time Processing
- **Sorted sets**: For velocity scoring (transactions per minute per user/device/IP)
- **Hashes**: For session state and risk scores
- **Streams**: For real-time event processing
- **Lua scripting**: Atomic fraud scoring logic
- **TTL-based expiration**: For temporary risk state storage

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Fraud Detection Accuracy | 95%+ | >90% |
| False Positive Rate | 15% | <20% |
| P99 Latency | <50ms | <100ms |
| Throughput | 50K+ TPS | 40K TPS |
| System Availability | 99.99% | 99.95% |
| Infrastructure Cost | 40% reduction | >30% reduction |

## Key Lessons Learned

1. **Graph databases excel at relationship-based fraud detection** - traditional SQL struggles with multi-hop path analysis
2. **Time-series databases optimize for temporal fraud patterns** - velocity checks, time-window aggregations
3. **Real-time scoring requires in-memory processing** - Redis provides the sub-millisecond latency needed
4. **Hybrid architecture balances strengths** - each database handles what it does best
5. **Continuous monitoring of fraud patterns** is essential for adapting to new attack vectors

## Technical Challenges and Solutions

- **Challenge**: High cardinality in graph relationships causing performance issues
  - **Solution**: Relationship filtering and sampling for initial analysis, full analysis only for high-risk cases

- **Challenge**: Real-time scoring at scale with consistent results
  - **Solution**: Deterministic scoring algorithms with idempotent operations

- **Challenge**: Data consistency across heterogeneous systems
  - **Solution**: Event sourcing with Kafka for guaranteed delivery and ordering

- **Challenge**: Regulatory compliance with data retention
  - **Solution**: Automated retention policies with audit trails

## Integration with ML Systems

### Real-time Scoring Pipeline
1. **Transaction ingestion**: Kafka → Redis (real-time scoring)
2. **Graph analysis**: Neo4j path queries for relationship detection
3. **Time-series analysis**: TimescaleDB aggregations for velocity checks
4. **ML model scoring**: Ensemble of rule-based and ML models
5. **Decision engine**: Risk score aggregation and action determination

### Model Training Integration
- **Feature engineering**: Extract graph features (centrality, community membership)
- **Temporal features**: Rolling window statistics from TimescaleDB
- **Real-time features**: Redis-sourced velocity metrics
- **Feedback loop**: Fraud outcomes used to retrain models daily

> 💡 **Pro Tip**: For fraud detection systems, prioritize recall over precision in the initial stages. It's better to flag more transactions for review than to miss fraudulent activity. The cost of false positives is much lower than the cost of missed fraud.