# Monitoring and Observability

Comprehensive monitoring and observability are essential for maintaining healthy database systems, especially for AI/ML applications where performance degradation can impact model training and inference quality.

## Overview

Monitoring provides visibility into system health, while observability enables understanding of system behavior through logs, metrics, and traces. For senior AI/ML engineers, these capabilities are critical for proactive issue detection, performance optimization, and capacity planning.

## Monitoring Categories

### Infrastructure Metrics
- **CPU utilization**: Database process CPU usage
- **Memory usage**: Shared buffers, work memory, OS memory
- **Disk I/O**: Read/write operations, latency, throughput
- **Network**: Connection counts, throughput, errors

### Database-Specific Metrics
- **Query performance**: Latency, throughput, error rates
- **Connection pool**: Active/idle connections, wait times
- **Lock contention**: Lock waits, deadlocks
- **Replication lag**: Primary-replica delay
- **Checkpoint frequency**: WAL flush activity

### Application-Level Metrics
- **End-to-end latency**: From application to database response
- **Error rates**: Query failures, connection errors
- **Throughput**: Queries per second, transactions per second
- **Cache hit ratios**: Redis/Memcached effectiveness

## Implementation Patterns

### PostgreSQL Monitoring
```sql
-- Key performance views
SELECT 
    datname,
    numbackends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    tup_returned,
    tup_fetched,
    tup_inserted,
    tup_updated,
    tup_deleted
FROM pg_stat_database;

-- Lock monitoring
SELECT 
    pid,
    usename,
    query,
    wait_event_type,
    wait_event,
    state
FROM pg_stat_activity
WHERE wait_event_type IS NOT NULL;

-- Checkpoint monitoring
SELECT 
    checkpoints_timed,
    checkpoints_req,
    checkpoint_write_time,
    checkpoint_sync_time
FROM pg_stat_bgwriter;
```

### Prometheus/Grafana Setup
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']  # postgres_exporter

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']  # redis_exporter

# postgres_exporter configuration
data_source_name: 'postgresql://user:password@localhost:5432/dbname?sslmode=disable'
```

### OpenTelemetry Integration
```python
# Python application with OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

# Set up tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = SimpleSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument database calls
Psycopg2Instrumentor().instrument()

# Trace database operations
with tracer.start_as_current_span("database_query") as span:
    cursor.execute("SELECT * FROM users WHERE id = %s", [user_id])
    span.set_attribute("db.operation", "SELECT")
    span.set_attribute("db.table", "users")
```

## AI/ML Specific Monitoring

### Model Training Pipeline Monitoring
- **Data pipeline health**: Ingestion rate, processing latency
- **Feature generation**: Success rate, computation time
- **Model training**: GPU/CPU utilization, training loss, convergence
- **Validation metrics**: Accuracy, precision, recall over time

### Real-time Inference Monitoring
- **Latency percentiles**: P50, P90, P99 response times
- **Error rates**: 5xx errors, timeout rates
- **Throughput**: Requests per second, batch sizes
- **Resource utilization**: CPU, memory, GPU for inference servers

### Feature Store Monitoring
- **Feature freshness**: Time since last update
- **Query performance**: Latency for feature retrieval
- **Data quality**: Missing values, outliers, distribution shifts
- **Cache effectiveness**: Hit/miss ratios

## Alerting Strategies

### Tiered Alerting System
- **Critical (P0)**: System down, data loss, security breach
- **High (P1)**: Major performance degradation, high error rates
- **Medium (P2)**: Warning conditions, capacity approaching limits
- **Low (P3)**: Informational, routine maintenance

### Example Alert Rules
```yaml
# Prometheus alert rules
groups:
  - name: database-alerts
    rules:
      - alert: HighDatabaseLatency
        expr: histogram_quantile(0.99, sum(rate(pg_stat_activity_query_duration_seconds_bucket[5m])) by (le))
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High database query latency"
          description: "99th percentile query latency > 1s for 5 minutes"

      - alert: ReplicationLagHigh
        expr: pg_replication_lag_seconds > 300
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Replication lag too high"
          description: "Replication lag > 5 minutes"

      - alert: ConnectionPoolExhausted
        expr: pg_stat_activity_count > 0.9 * pg_settings_max_connections
        for: 2m
        labels:
          severity: high
        annotations:
          summary: "Connection pool nearly exhausted"
          description: "Active connections at {{ $value }}% of maximum"
```

## Observability Best Practices

### Structured Logging
- **Consistent format**: JSON logs with standard fields
- **Correlation IDs**: Trace requests across services
- **Contextual information**: Include relevant business context
- **Log levels**: Appropriate use of DEBUG, INFO, WARN, ERROR

### Distributed Tracing
- **Trace propagation**: Pass trace IDs between services
- **Span hierarchy**: Parent-child relationships for operations
- **Sampling strategies**: Adaptive sampling for high-volume systems
- **Service maps**: Visualize service dependencies

### Metric Design Principles
1. **MECE principle**: Mutually Exclusive, Collectively Exhaustive
2. **Cardinality awareness**: Avoid high-cardinality dimensions
3. **Business relevance**: Metrics should drive decisions
4. **Alertability**: Every metric should have associated alerts
5. **Historical context**: Track trends, not just current values

## Monitoring Dashboard Examples

### Database Health Dashboard
- **Overview**: Uptime, connections, queries/sec
- **Performance**: Latency percentiles, throughput
- **Resources**: CPU, memory, disk I/O
- **Replication**: Lag, status, sync status
- **Errors**: Error rates, failed queries

### AI/ML Operations Dashboard
- **Data pipeline**: Ingestion rate, processing time
- **Feature store**: Freshness, query performance
- **Model serving**: Latency, error rates, throughput
- **Training**: GPU utilization, training progress
- **Cost**: Resource consumption, cost per inference

## Related Resources

- [High Availability] - HA monitoring requirements
- [Disaster Recovery] - DR monitoring and validation
- [Database Performance Tuning] - Performance monitoring
- [AI/ML System Design] - Observability in ML systems