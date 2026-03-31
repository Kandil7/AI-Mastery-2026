# TimescaleDB Tutorial for Time-Series ML Workloads

This tutorial provides hands-on TimescaleDB fundamentals specifically designed for AI/ML engineers working with time-series data for model monitoring, training metrics, and real-time analytics.

## Why TimescaleDB for ML Time-Series?

TimescaleDB combines the power of PostgreSQL with specialized time-series optimizations:
- **PostgreSQL compatibility**: Full SQL support, ACID compliance, rich ecosystem
- **Automatic partitioning**: Hypertables handle time-based partitioning automatically
- **Time-based indexing**: Optimized for time-range queries
- **Compression**: Up to 90% compression for time-series data
- **Continuous aggregates**: Pre-computed aggregations for dashboards
- **Downsampling**: Built-in functions for data reduction

## Setting Up TimescaleDB for ML Workloads

### Installation Options
```bash
# Docker (recommended for development)
docker run -d \
  --name timescaledb-ml \
  -e POSTGRES_USER=ml_user \
  -e POSTGRES_PASSWORD=ml_password \
  -e POSTGRES_DB=ml_platform \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15

# With optimized configuration for ML workloads
docker run -d \
  --name timescaledb-prod \
  -v /data/timescaledb:/var/lib/postgresql/data \
  -e POSTGRES_USER=ml_user \
  -e POSTGRES_PASSWORD=ml_password \
  -e POSTGRES_DB=ml_platform \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15 \
  -c shared_buffers=2GB \
  -c work_mem=64MB \
  -c maintenance_work_mem=512MB \
  -c effective_cache_size=6GB
```

### Essential Extensions and Setup
```sql
-- Connect to PostgreSQL and install TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Verify installation
SELECT * FROM timescaledb_informationhypertables;

-- Enable compression (requires TimescaleDB 2.0+)
SELECT add_compression_policy('model_metrics', INTERVAL '7 days');
SELECT add_compression_policy('training_logs', INTERVAL '1 day');
```

## Core TimescaleDB Concepts for ML Engineers

### Hypertables vs Regular Tables

#### Regular Table (Before TimescaleDB)
```sql
-- Standard PostgreSQL table
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    accuracy FLOAT,
    loss FLOAT,
    learning_rate FLOAT,
    epoch INTEGER,
    batch INTEGER
);

-- Manual partitioning would be required for large datasets
-- Indexes need to be created on each partition
```

#### Hypertable (TimescaleDB)
```sql
-- Create hypertable - automatic partitioning by time
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    accuracy FLOAT,
    loss FLOAT,
    learning_rate FLOAT,
    epoch INTEGER,
    batch INTEGER
);

-- Convert to hypertable (partition by time, 1-day chunks)
SELECT create_hypertable('model_metrics', 'timestamp', 
                        chunk_time_interval => INTERVAL '1 day');

-- Automatic partitioning creates chunks like:
-- model_metrics_1, model_metrics_2, etc.
```

### Data Model for ML Time-Series

#### Training Metrics Schema
```sql
-- Core hypertable for training metrics
CREATE TABLE training_metrics (
    time TIMESTAMPTZ NOT NULL,
    model_id UUID NOT NULL,
    run_id UUID NOT NULL,
    epoch INTEGER NOT NULL,
    batch INTEGER NOT NULL,
    loss FLOAT,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    learning_rate FLOAT,
    gradient_norm FLOAT,
    step_time_ms FLOAT
);

-- Create hypertable with 1-hour chunks for high-frequency data
SELECT create_hypertable('training_metrics', 'time', 
                        chunk_time_interval => INTERVAL '1 hour');

-- Add indexes for common query patterns
CREATE INDEX idx_training_metrics_model_time ON training_metrics (model_id, time DESC);
CREATE INDEX idx_training_metrics_run_time ON training_metrics (run_id, time DESC);
CREATE INDEX idx_training_metrics_epoch ON training_metrics (model_id, epoch);
```

#### Model Performance Monitoring
```sql
-- Hypertable for production model monitoring
CREATE TABLE model_performance (
    time TIMESTAMPTZ NOT NULL,
    model_id UUID NOT NULL,
    endpoint VARCHAR NOT NULL,
    request_count BIGINT,
    error_count BIGINT,
    latency_p50 FLOAT,
    latency_p90 FLOAT,
    latency_p99 FLOAT,
    throughput_rps FLOAT,
    cpu_usage FLOAT,
    memory_usage FLOAT,
    gpu_utilization FLOAT
);

-- Create hypertable with 5-minute chunks for high-frequency monitoring
SELECT create_hypertable('model_performance', 'time', 
                        chunk_time_interval => INTERVAL '5 minutes');

-- Add BRIN index for time-based queries
CREATE INDEX idx_model_performance_time_brin ON model_performance USING BRIN (time)
WITH (pages_per_range = 32);
```

## TimescaleDB Query Patterns for ML Workflows

### Basic Time-Series Queries
```sql
-- Get metrics for a specific model over last 24 hours
SELECT 
    time,
    epoch,
    loss,
    accuracy,
    learning_rate
FROM training_metrics
WHERE model_id = 'uuid-123'
  AND time > NOW() - INTERVAL '24 hours'
ORDER BY time DESC;

-- Get latest metrics for each model
SELECT DISTINCT ON (model_id) *
FROM training_metrics
ORDER BY model_id, time DESC;

-- Time-bucketed aggregation (15-minute intervals)
SELECT 
    time_bucket('15 minutes', time) as bucket,
    model_id,
    AVG(loss) as avg_loss,
    AVG(accuracy) as avg_accuracy,
    COUNT(*) as samples
FROM training_metrics
WHERE time > NOW() - INTERVAL '7 days'
GROUP BY bucket, model_id
ORDER BY bucket DESC, model_id;
```

### Continuous Aggregates for Dashboards
```sql
-- Create continuous aggregate for hourly summaries
CREATE MATERIALIZED VIEW training_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) as bucket,
    model_id,
    run_id,
    AVG(loss) as avg_loss,
    MIN(loss) as min_loss,
    MAX(loss) as max_loss,
    AVG(accuracy) as avg_accuracy,
    STDDEV(accuracy) as accuracy_stddev,
    COUNT(*) as sample_count
FROM training_metrics
GROUP BY bucket, model_id, run_id
WITH DATA;

-- Refresh policy (automatically refreshes new data)
SELECT add_continuous_aggregate_policy('training_metrics_hourly',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '10 minutes');

-- Query the continuous aggregate
SELECT 
    bucket,
    model_id,
    avg_loss,
    avg_accuracy
FROM training_metrics_hourly
WHERE bucket > NOW() - INTERVAL '24 hours'
ORDER BY bucket DESC;
```

### Advanced Time-Series Analysis

#### Moving Averages and Trends
```sql
-- Calculate moving averages with window functions
SELECT 
    time,
    model_id,
    loss,
    AVG(loss) OVER (
        PARTITION BY model_id 
        ORDER BY time 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as loss_ma_10,
    AVG(accuracy) OVER (
        PARTITION BY model_id 
        ORDER BY time 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as accuracy_ma_10
FROM training_metrics
WHERE model_id = 'uuid-123'
  AND time > NOW() - INTERVAL '1 hour'
ORDER BY time;
```

#### Anomaly Detection Patterns
```sql
-- Detect outliers using statistical methods
WITH stats AS (
    SELECT 
        model_id,
        AVG(loss) as mean_loss,
        STDDEV(loss) as std_loss
    FROM training_metrics
    WHERE time > NOW() - INTERVAL '1 hour'
    GROUP BY model_id
),
anomalies AS (
    SELECT 
        tm.*,
        (tm.loss - s.mean_loss) / s.std_loss as z_score
    FROM training_metrics tm
    JOIN stats s ON tm.model_id = s.model_id
    WHERE tm.time > NOW() - INTERVAL '1 hour'
)
SELECT *
FROM anomalies
WHERE ABS(z_score) > 3.0  -- 3 standard deviations
ORDER BY ABS(z_score) DESC;
```

## Performance Optimization for ML Time-Series

### Compression Strategies
```sql
-- Enable compression on hypertables
ALTER TABLE training_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'model_id, run_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- Manual compression (for testing)
SELECT compress_chunk('training_metrics_1');

-- Check compression status
SELECT 
    chunk_name,
    compression_status,
    before_compression_total_bytes,
    after_compression_total_bytes
FROM timescaledb_information.chunks
WHERE hypertable_name = 'training_metrics';
```

### Indexing Best Practices
```sql
-- BRIN indexes for time-series data (excellent for large datasets)
CREATE INDEX idx_training_metrics_time_brin ON training_metrics USING BRIN (time)
WITH (pages_per_range = 32);

-- Composite indexes for common query patterns
CREATE INDEX idx_training_metrics_model_run_time ON training_metrics (model_id, run_id, time DESC);
CREATE INDEX idx_training_metrics_epoch ON training_metrics (model_id, epoch);

-- Partial indexes for active data
CREATE INDEX idx_training_metrics_active ON training_metrics (model_id, time DESC)
WHERE time > NOW() - INTERVAL '24 hours';
```

### Chunk Management
```sql
-- View chunk information
SELECT 
    chunk_name,
    range_start,
    range_end,
    total_size,
    compressed_total_size
FROM timescaledb_information.chunks
WHERE hypertable_name = 'training_metrics';

-- Adjust chunk size based on data volume
-- For high-frequency data (per second): 1-hour chunks
-- For medium-frequency data: 1-day chunks  
-- For low-frequency data: 1-week chunks

-- Change chunk interval (requires recreation)
SELECT set_chunk_time_interval('training_metrics', INTERVAL '30 minutes');
```

## TimescaleDB for ML-Specific Workloads

### Experiment Tracking System
```sql
-- Hypertable for experiment tracking
CREATE TABLE experiments (
    time TIMESTAMPTZ NOT NULL,
    experiment_id UUID NOT NULL,
    model_id UUID NOT NULL,
    parameters JSONB,
    metrics JSONB,
    status TEXT,
    worker_id TEXT,
    gpu_id TEXT
);

SELECT create_hypertable('experiments', 'time', 
                        chunk_time_interval => INTERVAL '1 day');

-- Index for fast experiment lookup
CREATE INDEX idx_experiments_experiment_time ON experiments (experiment_id, time DESC);
CREATE INDEX idx_experiments_model_time ON experiments (model_id, time DESC);

-- Query recent experiments with metrics
SELECT 
    experiment_id,
    model_id,
    time,
    (metrics->>'accuracy')::FLOAT as accuracy,
    (metrics->>'loss')::FLOAT as loss,
    status
FROM experiments
WHERE model_id = 'uuid-123'
  AND time > NOW() - INTERVAL '7 days'
ORDER BY time DESC;
```

### Real-Time Monitoring Dashboard
```sql
-- Create continuous aggregate for dashboard
CREATE MATERIALIZED VIEW model_monitoring_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', time) as bucket,
    model_id,
    endpoint,
    AVG(latency_p90) as avg_latency_p90,
    AVG(throughput_rps) as avg_throughput,
    SUM(request_count) as total_requests,
    SUM(error_count) as total_errors,
    AVG(cpu_usage) as avg_cpu,
    AVG(gpu_utilization) as avg_gpu
FROM model_performance
GROUP BY bucket, model_id, endpoint
WITH DATA;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('model_monitoring_summary',
    start_offset => INTERVAL '5 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '2 minutes');
```

## Common TimescaleDB Pitfalls for ML Engineers

### 1. Chunk Size Misconfiguration
- **Problem**: Too small chunks → overhead, too large → poor query performance
- **Solution**: Start with 1-hour for high-frequency, 1-day for medium
- **Rule of thumb**: 100K-1M rows per chunk optimal

### 2. Missing Time Indexes
- **Problem**: Queries without time filters scan all chunks
- **Solution**: Always include time filters in WHERE clauses
- **Best practice**: Use `time > NOW() - INTERVAL 'X'` patterns

### 3. Over-compression
- **Problem**: Compressing hot data that's frequently queried
- **Solution**: Compress only older data, keep recent data uncompressed
- **Example**: Compress data older than 7 days

### 4. Continuous Aggregate Overhead
- **Problem**: Too many continuous aggregates affecting write performance
- **Solution**: Start with essential aggregates, monitor impact
- **Best practice**: Use `refresh_continuous_aggregate()` for manual control

## Visual Diagrams

### TimescaleDB Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│  PostgreSQL     │───▶│  TimescaleDB    │
│ (ML Training,   │    │  (Standard)     │    │  (Hypertables)  │
│  API, Dashboard)│    └─────────────────┘    └────────┬────────┘
└─────────────────┘                                     │
                                                          ▼
                                            ┌─────────────────────┐
                                            │  Chunk Management   │
                                            │  • Automatic partitioning│
                                            │  • Time-based chunks │
                                            │  • Compression       │
                                            └─────────────────────┘
                                                   ▲
                                                   │
                                           ┌─────────────────┐
                                           │  Continuous     │
                                           │  Aggregates      │
                                           │  • Pre-computed  │
                                           │  • Auto-refresh  │
                                           └─────────────────┘
```

### ML Time-Series Data Flow
```
Training Jobs → [Metrics Collector] → TimescaleDB (Raw Metrics)
       ↑                   │                      │
       │                   ▼                      ▼
[Data Pipeline] ← [Continuous Aggregates] ← [Real-time Queries]
       │                   │                      │
       ▼                   ▼                      ▼
[ML Monitoring] ← [Dashboard Queries] ← [Anomaly Detection]
```

## Hands-on Exercises

### Exercise 1: Training Metrics Hypertable
1. Create hypertable for training metrics with appropriate chunking
2. Insert simulated training data
3. Create continuous aggregates for hourly summaries
4. Query performance with different time ranges

### Exercise 2: Model Monitoring System
1. Design schema for production model monitoring
2. Implement BRIN indexes for time-based queries
3. Create continuous aggregates for dashboard data
4. Test compression and storage savings

### Exercise 3: Anomaly Detection Pipeline
1. Build statistical anomaly detection using window functions
2. Implement z-score calculation for outlier detection
3. Create alerting queries for performance degradation
4. Test with simulated anomalous data

## Best Practices Summary

1. **Start with appropriate chunk sizes**: 1-hour for high-frequency, 1-day for medium
2. **Use continuous aggregates** for dashboard queries instead of raw data
3. **Enable compression** for older data to save storage costs
4. **Always include time filters** in WHERE clauses for optimal performance
5. **Monitor chunk statistics** to tune partitioning strategy
6. **Use BRIN indexes** for large time-series datasets
7. **Plan for data retention**: Set up automated dropping of old chunks
8. **Test with realistic data volumes** before production deployment

This tutorial provides the foundation for effectively using TimescaleDB in machine learning time-series workloads, from training monitoring to production model observability.