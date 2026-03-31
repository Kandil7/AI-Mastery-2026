# PostgreSQL Basics Tutorial for AI/ML Engineers

This tutorial provides hands-on PostgreSQL fundamentals specifically tailored for AI/ML engineers who need to work with relational databases for model metadata, feature stores, and experiment tracking.

## Why PostgreSQL for AI/ML?

PostgreSQL is particularly well-suited for AI/ML workloads because:
- **Extensibility**: Supports JSONB, arrays, and custom data types
- **Advanced indexing**: GIN, GiST, BRIN, and functional indexes
- **ACID compliance**: Strong consistency for critical ML metadata
- **Rich ecosystem**: TimescaleDB extension for time-series, pgvector for vector search
- **Mature and reliable**: Battle-tested for production systems

## Setting Up PostgreSQL for ML Workflows

### Installation Options
```bash
# Docker (recommended for development)
docker run -d \
  --name postgres-ml \
  -e POSTGRES_USER=ml_user \
  -e POSTGRES_PASSWORD=ml_password \
  -e POSTGRES_DB=ml_platform \
  -p 5432:5432 \
  postgres:15

# Or use TimescaleDB for time-series capabilities
docker run -d \
  --name timescaledb-ml \
  -e POSTGRES_USER=ml_user \
  -e POSTGRES_PASSWORD=ml_password \
  -e POSTGRES_DB=ml_platform \
  -p 5432:5432 \
  timescale/timescaledb:latest-pg15
```

### Essential Extensions for ML
```sql
-- Enable essential extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- Text similarity
CREATE EXTENSION IF NOT EXISTS "hstore";    -- Key-value store
CREATE EXTENSION IF NOT EXISTS "jsonb_plperl"; -- JSON processing
-- For vector search (install pgvector separately)
-- CREATE EXTENSION IF NOT EXISTS "vector";
```

## Core PostgreSQL Concepts for ML Engineers

### Data Types for ML Workflows

#### JSONB for Flexible Metadata
```sql
-- Store model metadata with flexible schema
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    status TEXT CHECK (status IN ('draft', 'staging', 'production'))
);

-- Insert example with rich metadata
INSERT INTO models (name, description, metadata, status) VALUES
('resnet-50-v2', 'Improved ResNet-50 with better regularization',
 '{"training_config": {"epochs": 100, "batch_size": 32, "optimizer": "AdamW"},
   "metrics": {"accuracy": 0.942, "precision": 0.938},
   "artifacts": [{"name": "weights.h5", "size_bytes": 24576000}],
   "tags": ["computer_vision", "image_classification"]}',
 'production');
```

#### Array Types for Feature Vectors
```sql
-- Store small feature vectors directly in PostgreSQL
CREATE TABLE features (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    feature_vector FLOAT4[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert feature vector
INSERT INTO features (model_id, feature_vector) VALUES
('uuid-123', ARRAY[0.12, -0.34, 0.56, 0.78, ...]);
```

#### hstore for Key-Value Pairs
```sql
-- Store dynamic properties efficiently
CREATE TABLE experiments (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    parameters HSTORE,
    results HSTORE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert with hstore
INSERT INTO experiments (model_id, parameters, results) VALUES
('uuid-123',
 '"learning_rate"=>"0.001", "batch_size"=>"32", "epochs"=>"100"',
 '"accuracy"=>"0.942", "loss"=>"0.058", "training_time"=>"18432"');
```

## Advanced Query Patterns for ML Workflows

### JSONB Querying
```sql
-- Extract values from JSONB
SELECT 
    id,
    name,
    (metadata->'metrics'->>'accuracy')::FLOAT as accuracy,
    (metadata->'training_config'->>'epochs')::INTEGER as epochs
FROM models
WHERE (metadata->'metrics'->>'accuracy')::FLOAT > 0.9;

-- Search within JSON arrays
SELECT id, name
FROM models
WHERE metadata ?| ARRAY['computer_vision', 'nlp'];

-- Full-text search on JSON fields
SELECT id, name, ts_rank_cd(
    to_tsvector('english', metadata->>'description'),
    plainto_tsquery('english', 'image classification')
) as rank
FROM models
WHERE to_tsvector('english', metadata->>'description') @@ plainto_tsquery('english', 'image classification')
ORDER BY rank DESC;
```

### Window Functions for Time-Series Analysis
```sql
-- Track model performance over time
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES models(id),
    timestamp TIMESTAMPTZ NOT NULL,
    accuracy FLOAT,
    loss FLOAT,
    learning_rate FLOAT
);

-- Calculate moving averages for training metrics
SELECT 
    timestamp,
    accuracy,
    AVG(accuracy) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as accuracy_ma_10,
    loss,
    AVG(loss) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) as loss_ma_10
FROM model_metrics
WHERE model_id = 'uuid-123'
ORDER BY timestamp;
```

### Materialized Views for ML Dashboards
```sql
-- Create materialized view for model performance dashboard
CREATE MATERIALIZED VIEW model_performance_summary AS
SELECT 
    m.name as model_name,
    m.status,
    COUNT(*) as total_runs,
    AVG(mm.accuracy) as avg_accuracy,
    MAX(mm.timestamp) as last_run,
    MIN(mm.timestamp) as first_run,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mm.accuracy) as median_accuracy
FROM models m
JOIN model_metrics mm ON m.id = mm.model_id
GROUP BY m.id, m.name, m.status;

-- Refresh strategy (scheduled or manual)
REFRESH MATERIALIZED VIEW model_performance_summary;
```

## Indexing Strategies for ML Performance

### GIN Indexes for JSONB and Arrays
```sql
-- Index JSONB fields for fast querying
CREATE INDEX idx_models_metadata_gin ON models USING GIN (metadata);
CREATE INDEX idx_models_tags ON models USING GIN ((metadata->'tags'));

-- Index array columns
CREATE INDEX idx_features_vector_gin ON features USING GIN (feature_vector);

-- Query using indexed JSONB
SELECT id, name 
FROM models 
WHERE metadata @> '{"tags": ["computer_vision"]}'::jsonb;
```

### BRIN Indexes for Time-Series Data
```sql
-- Perfect for time-series metrics data
CREATE INDEX idx_model_metrics_timestamp_brin 
ON model_metrics USING BRIN (timestamp)
WITH (pages_per_range = 32);

-- BRIN works best when data is physically ordered by the indexed column
-- Ensure your INSERTs are ordered by timestamp for optimal BRIN performance
```

### Functional Indexes for Common ML Patterns
```sql
-- Index on normalized text for case-insensitive searches
CREATE INDEX idx_models_lower_name ON models (LOWER(name));

-- Index on extracted JSON values
CREATE INDEX idx_models_accuracy ON models (((metadata->'metrics'->>'accuracy')::FLOAT));
CREATE INDEX idx_models_epochs ON models (((metadata->'training_config'->>'epochs')::INTEGER));

-- Composite index for common query patterns
CREATE INDEX idx_models_status_accuracy 
ON models (status, (metadata->'metrics'->>'accuracy')::FLOAT DESC);
```

## PostgreSQL for Feature Stores

### Designing a Feature Store Schema
```sql
-- Dimension tables (slowly changing)
CREATE TABLE entities (
    entity_id UUID PRIMARY KEY,
    entity_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE features (
    feature_id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    data_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fact table (time-series features)
CREATE TABLE feature_values (
    entity_id UUID NOT NULL REFERENCES entities(entity_id),
    feature_id UUID NOT NULL REFERENCES features(feature_id),
    timestamp TIMESTAMPTZ NOT NULL,
    value NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (entity_id, feature_id, timestamp)
);

-- Add composite index for time-based queries
CREATE INDEX idx_feature_values_time ON feature_values (timestamp, entity_id, feature_id);
```

### Efficient Feature Retrieval
```sql
-- Get latest feature values for an entity
SELECT 
    f.name,
    fv.value,
    fv.timestamp
FROM features f
JOIN feature_values fv ON f.feature_id = fv.feature_id
WHERE fv.entity_id = 'user-123'
  AND fv.timestamp = (
      SELECT MAX(timestamp) 
      FROM feature_values 
      WHERE entity_id = 'user-123' AND feature_id = f.feature_id
  );

-- Get feature values over time range
SELECT 
    f.name,
    fv.timestamp,
    fv.value
FROM features f
JOIN feature_values fv ON f.feature_id = fv.feature_id
WHERE fv.entity_id = 'user-123'
  AND fv.feature_id IN (SELECT feature_id FROM features WHERE name IN ('age', 'income', 'engagement_score'))
  AND fv.timestamp >= NOW() - INTERVAL '7 days'
ORDER BY fv.timestamp, f.name;
```

## Performance Optimization for ML Workloads

### Connection Pooling
```sql
-- Use connection pooling (PgBouncer) for high-concurrency ML workloads
-- Configuration example for PgBouncer
[databases]
ml_platform = host=localhost port=5432 dbname=ml_platform

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
```

### Vacuum and Autovacuum Tuning
```sql
-- For tables with frequent updates (like metrics tables)
ALTER TABLE model_metrics SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.05,
    autovacuum_vacuum_threshold = 1000,
    autovacuum_analyze_threshold = 500
);
```

### Parallel Query Tuning
```sql
-- Enable parallel queries for analytical workloads
SET max_parallel_workers_per_gather = 4;
SET parallel_setup_cost = 1000;
SET parallel_tuple_cost = 0.01;

-- Test parallel query performance
EXPLAIN (ANALYZE, VERBOSE) 
SELECT COUNT(*), AVG(accuracy)
FROM model_metrics
WHERE timestamp > NOW() - INTERVAL '30 days';
```

## Common PostgreSQL Pitfalls for ML Engineers

### 1. JSONB vs hstore vs Regular Columns
- **Use JSONB** for complex nested structures that vary frequently
- **Use hstore** for simple key-value pairs with known keys
- **Use regular columns** for frequently queried, stable attributes

### 2. Array Size Limits
- PostgreSQL arrays have practical limits (~1GB)
- For large vectors (> 10K dimensions), consider external storage
- Use `FLOAT4[]` for memory efficiency over `FLOAT8[]`

### 3. Transaction Isolation Levels
- **READ COMMITTED** (default): Good for most ML metadata operations
- **REPEATABLE READ**: Needed for consistent experiment tracking
- **SERIALIZABLE**: Overkill for most ML use cases

### 4. Lock Contention in High-Throughput Systems
- Avoid long-running transactions during training
- Use `SELECT FOR UPDATE SKIP LOCKED` for distributed job coordination
- Consider partitioning for high-write tables

## Visual Diagrams

### PostgreSQL Architecture for ML Systems
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│  Connection     │───▶│  PostgreSQL     │
│ (ML Training,   │    │  Pool (PgBouncer)│    │  (Main DB)      │
│  API, Dashboard)│    └─────────────────┘    └────────┬────────┘
└─────────────────┘                                     │
                                                          ▼
                                            ┌─────────────────────┐
                                            │  Extensions         │
                                            │  • pgvector (vectors)│
                                            │  • TimescaleDB (TS) │
                                            │  • PostGIS (geo)    │
                                            └─────────────────────┘
```

### Feature Store Schema
```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│   Entities  │    │   Features  │    │  Feature Values │
└─────────────┘    └─────────────┘    └─────────────────┘
│ entity_id(PK)│    │ feature_id(PK)│    │ entity_id(FK)  │
│ entity_type  │    │ name         │    │ feature_id(FK) │
│ created_at  │    │ data_type    │    │ timestamp       │
└─────────────┘    └─────────────┘    │ value           │
       ▲                   ▲          │ created_at      │
       │                   │          └─────────────────┘
       └───────────┬───────┴───────────┘
                   │
           ┌─────────────────┐
           │  Model Registry │
           └─────────────────┘
           │ model_id (PK)   │
           │ name            │
           │ metadata (JSONB)│
           └─────────────────┘
```

## Hands-on Exercises

### Exercise 1: Build a Model Registry
1. Create tables for models, versions, and deployments
2. Insert sample model data with JSONB metadata
3. Write queries to find top-performing models
4. Create indexes for common query patterns

### Exercise 2: Time-Series Metrics Analysis
1. Create a table for training metrics with timestamp
2. Insert simulated training data
3. Write queries for moving averages and trend analysis
4. Create a materialized view for dashboard reporting

### Exercise 3: Feature Store Implementation
1. Design schema for entities, features, and feature values
2. Implement efficient queries for latest feature values
3. Add appropriate indexes for time-based queries
4. Test performance with realistic data volumes

## Best Practices Summary

1. **Start with JSONB** for flexible ML metadata
2. **Use appropriate indexing** (GIN for JSONB, BRIN for time-series)
3. **Design for your query patterns**, not just data structure
4. **Monitor vacuum activity** for high-write tables
5. **Use connection pooling** for production ML applications
6. **Version your schemas** like you version your code
7. **Test with realistic data volumes** before production deployment

This tutorial provides the foundation for using PostgreSQL effectively in AI/ML systems, from model registry to feature stores and time-series analysis.