# Database Monitoring Fundamentals: Key Metrics for AI/ML Systems

## Overview

Database monitoring is essential for maintaining reliability, performance, and scalability in AI/ML production systems. Unlike traditional applications, ML workloads have unique patterns: bursty training jobs, high-throughput inference requests, and complex data dependencies that require specialized monitoring approaches.

## Core Monitoring Dimensions

### 1. Availability Metrics
- **Uptime**: Percentage of time database is operational
- **Failover time**: Time to recover from primary failure
- **Connection success rate**: Percentage of successful connection attempts
- **Query timeout rate**: Percentage of queries exceeding timeout thresholds

### 2. Performance Metrics
- **Latency percentiles**: p50, p95, p99 query response times
- **Throughput**: Queries per second (QPS), transactions per second (TPS)
- **Resource utilization**: CPU, memory, I/O, network bandwidth
- **Queue depth**: Number of pending operations in system queues

### 3. Data Integrity Metrics
- **Replication lag**: Time difference between primary and replicas
- **Consistency violations**: Number of inconsistent reads
- **Data corruption events**: Detected data integrity issues
- **Backup verification**: Success rate of backup restoration tests

## Database-Type Specific Metrics

### Relational Databases (PostgreSQL, MySQL)

#### Connection Pool Metrics
- `active_connections`: Currently active connections
- `idle_in_transaction`: Connections idle but holding locks
- `max_connections_used`: Peak connection usage
- `connection_wait_time`: Time spent waiting for connections

#### Query Performance
- `slow_queries_per_minute`: Queries exceeding threshold
- `query_cache_hit_ratio`: Cache efficiency
- `index_hit_ratio`: Index usage efficiency
- `temp_files_created`: Temporary files indicating memory pressure

#### Example PostgreSQL Monitoring Dashboard
```sql
-- Critical metrics queries
SELECT 
    current_setting('max_connections') as max_connections,
    COUNT(*) as active_connections,
    SUM(CASE WHEN state = 'idle in transaction' THEN 1 ELSE 0 END) as idle_in_tx
FROM pg_stat_activity;

SELECT 
    schemaname,
    relname,
    idx_scan,
    idx_tup_fetch,
    seq_scan,
    seq_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC LIMIT 10;
```

### NoSQL Databases (MongoDB, Cassandra, Redis)

#### MongoDB Specific
- `opcounters`: Insert, query, update, delete operations
- `globalLock`: Lock contention metrics
- `mem`: Memory usage breakdown
- `network`: Network I/O statistics

#### Cassandra Specific
- `read_latency`, `write_latency`: Per-operation latency
- `pending_compactions`: Background compaction queue
- `hinted_handoff`: Pending hinted handoffs
- `gc_pause_time`: Garbage collection pause times

#### Redis Specific
- `used_memory`: Memory usage
- `connected_clients`: Active client connections
- `instantaneous_ops_per_sec`: Operations per second
- `keyspace_hits/misses`: Cache hit ratio

### Time-Series Databases (InfluxDB, TimescaleDB)

#### Time-Series Specific
- `points_written_per_second`: Ingestion throughput
- `query_latency_by_time_range`: Latency vs time range
- `series_cardinality`: Number of unique time series
- `compaction_duration`: Background compaction performance

## AI/ML Workload Specific Metrics

### Training Workload Monitoring
- **Data loading rate**: Records/sec loaded into training pipeline
- **GPU memory pressure**: GPU memory utilization during training
- **Batch processing latency**: Time per training batch
- **Checkpoint frequency**: Model checkpoint creation rate

### Inference Workload Monitoring
- **P99 inference latency**: Critical for real-time systems
- **Request queuing time**: Time spent waiting in inference queue
- **Model version skew**: Differences between deployed model versions
- **Feature freshness**: Time since last feature update

### Feature Store Monitoring
- **Feature staleness**: Age of feature values
- **Feature coverage**: Percentage of entities with complete features
- **Real-time vs batch feature sync**: Lag between processing pipelines
- **Feature drift detection**: Statistical changes in feature distributions

## Monitoring Architecture Patterns

### Tiered Monitoring Approach
```
Tier 1: Infrastructure (CPU, memory, disk I/O)
Tier 2: Database engine (connections, queries, locks)
Tier 3: Application logic (business metrics, ML-specific)
Tier 4: Business impact (user experience, revenue impact)
```

### Real-time vs Batch Monitoring
| Type | Frequency | Use Case | Tools |
|------|-----------|----------|-------|
| Real-time | <1s | Alerting, anomaly detection | Prometheus, Datadog |
| Near-real-time | 1-60s | Performance optimization | Grafana, Kibana |
| Batch | Minutes-hours | Trend analysis, capacity planning | BigQuery, Redshift |

## Implementation Examples

### Prometheus Exporter Configuration
```yaml
# postgres_exporter.yml
datasource:
  host: postgres-prod
  port: 5432
  user: exporter
  password: secret
  sslmode: require

queries:
  - name: "pg_connections"
    query: |
      SELECT 
        count(*) as connections,
        sum(CASE WHEN state = 'active' THEN 1 ELSE 0 END) as active,
        sum(CASE WHEN state = 'idle in transaction' THEN 1 ELSE 0 END) as idle_in_tx
      FROM pg_stat_activity
    metrics:
      - connections: gauge
      - active: gauge
      - idle_in_tx: gauge

  - name: "pg_query_latency"
    query: |
      SELECT 
        percentile_disc(0.5) WITHIN GROUP (ORDER BY total_time) as p50_ms,
        percentile_disc(0.95) WITHIN GROUP (ORDER BY total_time) as p95_ms,
        percentile_disc(0.99) WITHIN GROUP (ORDER BY total_time) as p99_ms
      FROM pg_stat_statements
    metrics:
      - p50_ms: gauge
      - p95_ms: gauge
      - p99_ms: gauge
```

### AI/ML Specific Dashboard
```python
class AIDatabaseMonitor:
    def __init__(self):
        self.metrics = {
            'training_data_lag': Gauge('training_data_lag_seconds'),
            'inference_p99_latency': Gauge('inference_p99_latency_ms'),
            'feature_staleness': Gauge('feature_staleness_minutes'),
            'model_version_skew': Gauge('model_version_skew_count')
        }
    
    def collect_metrics(self):
        # Collect database metrics
        db_metrics = self._collect_db_metrics()
        
        # Collect AI/ML specific metrics
        ml_metrics = self._collect_ml_metrics()
        
        # Combine and export
        combined = {**db_metrics, **ml_metrics}
        return combined
    
    def _collect_ml_metrics(self):
        # Feature store staleness
        latest_feature_time = self.feature_store.get_latest_timestamp()
        current_time = time.time()
        staleness = (current_time - latest_feature_time) / 60  # minutes
        
        # Model version consistency
        deployed_versions = self.model_registry.get_deployed_versions()
        version_skew = len(set(deployed_versions)) - 1 if deployed_versions else 0
        
        return {
            'feature_staleness': staleness,
            'model_version_skew': version_skew
        }
```

## Real-World Production Examples

### Google's ML Infrastructure Monitoring
- **Multi-layer monitoring**: Infrastructure → database → ML pipeline
- **Anomaly detection**: ML models detect abnormal database behavior
- **Auto-remediation**: Automated responses to common failure patterns
- **Cost-aware monitoring**: Balance monitoring overhead with value

### Netflix's Database Monitoring
- **Per-service dashboards**: Individual monitoring for each microservice
- **Chaos engineering integration**: Monitor during failure injection
- **Capacity forecasting**: Predict database growth based on ML workload patterns
- **Cross-system correlation**: Link database metrics to application performance

## Debugging Techniques

### Common Failure Patterns
1. **Connection pool exhaustion**: High connection wait times
2. **Index bloat**: Poor query performance despite good hardware
3. **Write amplification**: Excessive I/O for small data changes
4. **Memory pressure**: Swapping causing severe performance degradation

### Diagnostic Workflow
1. **Identify symptom**: High latency, errors, timeouts
2. **Correlate metrics**: Check related metrics across systems
3. **Drill down**: Examine specific queries, connections, or time periods
4. **Reproduce**: Create test case to isolate issue
5. **Validate fix**: Confirm resolution with before/after metrics

## Best Practices for Senior Engineers

1. **Monitor what matters**: Focus on business-impact metrics, not just technical ones
2. **Implement SLOs**: Define service level objectives for database performance
3. **Use golden signals**: Latency, traffic, errors, saturation
4. **Design for observability**: Instrument code for rich context
5. **Automate root cause analysis**: Build tools that suggest likely causes

## Related Resources
- [System Design: ML Infrastructure Monitoring](../03_system_design/ml_infrastructure_monitoring.md)
- [Debugging Patterns: Database Performance Bottlenecks](../05_interview_prep/performance_bottleneck_identification.md)
- [Case Study: Real-time Feature Store Monitoring](../06_case_studies/realtime_feature_monitoring.md)

---
*Last updated: February 2026 | Target audience: Senior AI/ML Engineers*