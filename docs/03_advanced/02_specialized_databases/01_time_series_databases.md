# Time-Series Databases

Time-series databases (TSDBs) are specialized database systems optimized for storing and querying time-stamped data. They are essential for AI/ML applications involving monitoring, IoT, financial data, and real-time analytics.

## Overview

Time-series data has unique characteristics that make traditional databases inefficient:
- **High write throughput**: Millions of data points per second
- **Time-based queries**: Range queries, aggregations over time windows
- **Data retention policies**: Automatic data expiration
- **Downsampling**: Aggregation for long-term storage
- **High cardinality**: Many unique time series (metrics + dimensions)

For senior AI/ML engineers, understanding TSDBs is critical for building real-time AI systems and monitoring ML model performance.

## Core Concepts

### Time-Series Data Model
- **Metric**: The measurement being collected (e.g., "cpu_usage")
- **Tags/Dimensions**: Key-value pairs that identify the series (e.g., "host=server1", "region=us-west")
- **Fields**: The actual values (e.g., "value=75.3", "unit=%")
- **Timestamp**: When the measurement was taken

### Cardinality Challenges
- **High cardinality**: Too many unique combinations of tags
- **Impact**: Memory usage, query performance degradation
- **Solutions**: Tag filtering, cardinality limits, sampling

## Popular Time-Series Databases

### InfluxDB
- **Architecture**: Custom storage engine optimized for time-series
- **Query language**: Flux (modern) or InfluxQL (legacy)
- **Features**: Built-in downsampling, continuous queries
- **Use cases**: IoT, infrastructure monitoring

### TimescaleDB (PostgreSQL extension)
- **Architecture**: PostgreSQL with hypertables
- **Query language**: SQL with time-series extensions
- **Features**: Full SQL support, relational capabilities
- **Use cases**: Mixed workloads, complex analytics

### Prometheus
- **Architecture**: Pull-based, single-node design
- **Query language**: PromQL
- **Features**: Alerting, service discovery
- **Use cases**: Kubernetes monitoring, microservices

### ClickHouse
- **Architecture**: Columnar storage, vectorized execution
- **Query language**: SQL dialect
- **Features**: Extremely high throughput, real-time analytics
- **Use cases**: Large-scale analytics, user behavior tracking

## Database Design Patterns

### Schema Design
```sql
-- TimescaleDB hypertable example
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name TEXT NOT NULL,
    host TEXT NOT NULL,
    region TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit TEXT,
    metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('metrics', 'time');

-- Create indexes for common queries
CREATE INDEX idx_metrics_time ON metrics (time DESC);
CREATE INDEX idx_metrics_host_time ON metrics (host, time DESC);
CREATE INDEX idx_metrics_metric_host_time ON metrics (metric_name, host, time DESC);

-- Partition by time and host for large datasets
SELECT set_chunk_time_interval('metrics', INTERVAL '1 day');
SELECT set_number_of_partitions('metrics', 4);
```

### InfluxDB Line Protocol
```
# Measurement, tags, fields, timestamp
cpu,host=server1,region=us-west usage_user=23.5,usage_system=12.8 1640995200000000000

# Multiple points in one line
memory,host=server1,region=us-west used=1024,free=2048 1640995200000000000
memory,host=server2,region=us-west used=2048,free=1024 1640995200000000000
```

### Prometheus Metrics Format
```
# HELP http_requests_total The total number of HTTP requests.
# TYPE http_requests_total counter
http_requests_total{method="post",endpoint="/api/users"} 1027
http_requests_total{method="get",endpoint="/api/users"} 2984
http_requests_total{method="post",endpoint="/api/posts"} 432
```

## Query Patterns and Optimization

### Common Query Types
- **Range queries**: Get data for a time window
- **Aggregations**: Sum, average, min, max over time windows
- **Downsampling**: Aggregate to lower resolution
- **Joins**: Combine multiple time series
- **Predictive queries**: Forecasting and anomaly detection

### Optimized Queries
```sql
-- TimescaleDB: Efficient time-range query
SELECT 
    time_bucket('5 minutes', time) as bucket,
    host,
    AVG(value) as avg_value,
    MAX(value) as max_value
FROM metrics
WHERE 
    time >= NOW() - INTERVAL '24 hours'
    AND metric_name = 'cpu_usage'
    AND host LIKE 'server%'
GROUP BY bucket, host
ORDER BY bucket DESC;

-- InfluxDB Flux: Advanced aggregation
from(bucket: "metrics")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "cpu_usage")
  |> filter(fn: (r) => r.host =~ /^server[0-9]+$/)
  |> aggregateWindow(every: 5m, fn: mean)
  |> yield(name: "mean")

-- Prometheus PromQL: Rate calculation
rate(http_requests_total{job="api-server"}[5m])
```

### Indexing Strategies
- **Time-based indexing**: Primary index on timestamp
- **Tag indexing**: Secondary indexes on high-cardinality tags
- **Composite indexes**: Time + frequently filtered tags
- **BRIN indexes**: For time-partitioned data (TimescaleDB)

## AI/ML Integration Patterns

### Real-time Model Monitoring
- **Latency tracking**: Request/response times
- **Prediction drift**: Statistical changes in predictions
- **Feature distribution**: Changes in input features
- **Error rates**: Classification/regression errors

### Anomaly Detection
- **Statistical methods**: Z-score, moving averages
- **Machine learning**: Isolation forests, autoencoders
- **Time-series models**: ARIMA, Prophet, LSTM
- **Hybrid approaches**: Combine statistical and ML methods

### Predictive Analytics
- **Forecasting**: Future values based on historical patterns
- **Trend analysis**: Long-term patterns and seasonality
- **Correlation analysis**: Relationships between different metrics
- **Causal inference**: Understanding cause-effect relationships

## Performance Optimization

### Write Optimization
- **Batching**: Group writes to reduce overhead
- **Compression**: Time-series specific compression (delta encoding)
- **Memory management**: Efficient buffer management
- **Write-ahead logging**: Ensure durability without sacrificing speed

### Read Optimization
- **Columnar storage**: Store similar data together
- **Vectorized execution**: Process multiple rows at once
- **Caching**: Frequently accessed time ranges
- **Materialized views**: Pre-computed aggregations

### Scalability Patterns
- **Horizontal scaling**: Clustered TSDBs (InfluxDB Enterprise, TimescaleDB Multi-node)
- **Sharding**: By time range or tag combinations
- **Tiered storage**: Hot (SSD), warm (HDD), cold (object storage)
- **Edge processing**: Pre-aggregate at edge before sending to central DB

## Best Practices

1. **Design for cardinality**: Limit tag combinations early
2. **Use appropriate retention**: Balance storage cost vs query needs
3. **Monitor query performance**: Identify slow queries and optimize
4. **Implement proper sampling**: For very high-frequency data
5. **Test with production-like data**: Synthetic data may not reveal issues
6. **Consider hybrid approaches**: Combine TSDB with relational for complex analytics

## Related Resources

- [Database Performance] - General performance optimization
- [Index Optimization] - Advanced indexing techniques
- [AI/ML System Design] - TSDBs in ML system architecture
- [Monitoring and Observability] - Operational aspects of TSDBs