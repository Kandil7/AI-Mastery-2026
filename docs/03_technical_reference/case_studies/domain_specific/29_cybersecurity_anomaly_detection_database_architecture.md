---

# Case Study 29: Cybersecurity Anomaly Detection System - Time-Series + Graph Database Architecture

## Executive Summary

**Problem**: Detect sophisticated cyber attacks in real-time across 10M+ endpoints with sub-100ms latency while maintaining 99.99% availability and handling 100K+ events/sec.

**Solution**: Implemented hybrid architecture using TimescaleDB for time-series security events, Neo4j for relationship analysis of attack patterns, and Redis for real-time scoring and alerting.

**Impact**: Achieved 94% attack detection accuracy, 75% reduction in false positives, sub-60ms P99 latency, and prevented $50M+ in potential breach costs annually.

**System design snapshot**:
- SLOs: p99 <100ms; 99.99% availability; 94%+ attack detection accuracy
- Scale: 10M+ endpoints, 100K+ events/sec at peak, 1B+ historical events
- Cost efficiency: 50% reduction in infrastructure costs vs legacy SIEM
- Data quality: Automated validation of detection rules and model drift
- Reliability: Multi-region deployment with automatic failover

## Technical Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ TimescaleDB     │    │   Neo4j Graph   │    │    Redis        │
│  • Security events│  • Attack patterns│    │  • Real-time scoring│
│  • Time-series data│  • Relationship graphs│  • Alert state    │
└────────┬──────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                         │                           │
         └───────────┬─────────────┴───────────┬───────────────┘
                     │                         │
             ┌───────▼─────────┐     ┌─────────▼─────────┐
             │   Kafka Streams │     │   ML Detection Engine│
             │  • Event ingestion│     │  • Real-time models│
             └─────────────────┘     └─────────────────────┘
```

## Implementation Details

### TimescaleDB Configuration
- **Hypertable design**: `security_events` table partitioned by 5-minute intervals
- **Time-series optimization**: 
  - Continuous aggregates for rolling windows (1m, 5m, 15m, 1h)
  - Compression for older data (7+ days)
  - Retention policies based on compliance requirements
- **Indexing strategy**: 
  - BRIN indexes for time-range queries
  - GIN indexes for JSONB metadata (source, destination, protocol, user)
  - Partial indexes for high-priority events

### Neo4j Graph Database
- **Node types**: `Endpoint`, `User`, `IP`, `Process`, `File`, `NetworkFlow`
- **Relationship types**: `CONNECTED_TO`, `EXECUTED`, `ACCESSED`, `TRANSFERRED`, `RELATED_TO`
- **Query patterns**:
  - Lateral movement detection: `MATCH (e:Endpoint)-[:CONNECTED_TO*1..3]->(e2:Endpoint) WHERE e.risk_score > 0.8 AND e2.risk_score > 0.8`
  - Command-and-control detection: `MATCH (p:Process)-[:CONNECTED_TO]->(ip:IP) WHERE ip.is_c2 = true AND p.suspicious = true`
  - Data exfiltration detection: `MATCH (f:File)-[:TRANSFERRED]->(ip:IP) WHERE f.size > 1000000 AND ip.country != 'US'`
- **Performance**: Indexes on `Endpoint.id`, `IP.address`, `User.username`; materialized views for common attack patterns

### Redis Real-time Processing
- **Sorted sets**: For real-time risk scoring (events per minute per endpoint/user/IP)
- **Hashes**: For endpoint state and risk scores
- **Streams**: For real-time event processing and alerting
- **Lua scripting**: Atomic detection logic for complex correlation rules
- **TTL-based expiration**: For temporary threat state storage

## Performance Optimization

### Real-time Detection Pipeline
1. **Event ingestion**: Kafka → TimescaleDB (storage) + Redis (real-time scoring)
2. **Rule-based detection**: Redis Lua scripts for simple correlation rules
3. **Graph-based detection**: Neo4j path queries for complex attack patterns
4. **ML-based detection**: Ensemble models for anomaly detection
5. **Alert aggregation**: Risk score combination and thresholding
6. **Response automation**: Integration with SOAR platforms

### Query Optimization
- **Temporal filtering**: Leverage TimescaleDB time-partitioning for recent events
- **Metadata pre-filtering**: Redis hashes for quick risk assessment
- **Graph pruning**: Limit path depth and use relationship filters
- **Caching strategy**: Redis cache for frequent threat patterns

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Attack Detection Accuracy | 94%+ | >90% |
| False Positive Rate | 25% | <30% |
| P99 Latency | <60ms | <100ms |
| Throughput | 100K+ EPS | 80K EPS |
| Infrastructure Cost | 50% reduction | >40% reduction |
| Alert Response Time | <2 seconds | <5 seconds |

## Key Lessons Learned

1. **Time-series databases are essential for security analytics** - temporal patterns reveal attack sequences
2. **Graph databases excel at relationship-based threat detection** - traditional SQL struggles with multi-hop attack paths
3. **Real-time scoring requires in-memory processing** - Redis provides the sub-millisecond latency needed
4. **Hybrid architecture balances strengths** - each database handles what it does best
5. **Continuous learning is critical** - threat landscape evolves rapidly, models need constant updating

## Technical Challenges and Solutions

- **Challenge**: High cardinality in security events causing performance issues
  - **Solution**: Event sampling for initial analysis, full analysis only for high-risk events

- **Challenge**: Real-time correlation at scale with consistent results
  - **Solution**: Deterministic correlation algorithms with idempotent operations

- **Challenge**: Data consistency across heterogeneous systems
  - **Solution**: Event sourcing with Kafka for guaranteed delivery and ordering

- **Challenge**: Regulatory compliance with data retention
  - **Solution**: Automated retention policies with audit trails

## Integration with Security Operations

### Real-time Alerting Pipeline
1. **Event processing**: Kafka streams to detection engines
2. **Risk scoring**: Redis-based real-time scoring
3. **Correlation analysis**: Neo4j graph queries for complex patterns
4. **ML validation**: Ensemble models for final verification
5. **Alert generation**: Priority-based alert creation
6. **SOAR integration**: Automated response workflows

### Threat Intelligence Integration
- **External feeds**: Ingest STIX/TAXII feeds into Neo4j
- **IOC matching**: Real-time IOC lookup against known threats
- **Context enrichment**: Add threat context to alerts
- **Feedback loop**: Alert outcomes used to retrain models

> 💡 **Pro Tip**: For cybersecurity systems, prioritize detection coverage over precision in the initial stages. It's better to have more alerts for investigation than to miss sophisticated attacks. The cost of missed breaches far exceeds the cost of investigating false positives.