# Index Optimization

Advanced indexing techniques for optimizing database performance in AI/ML applications. This document covers sophisticated indexing strategies beyond basic B-Tree indexes.

## Overview

Index optimization goes beyond creating simple indexes to include advanced techniques that maximize query performance while minimizing storage overhead and maintenance costs. For senior AI/ML engineers, mastering these techniques is essential for building high-performance data systems.

## Advanced Index Types

### Covering Indexes (Index-Only Scans)
Covering indexes include all columns needed for a query, eliminating the need to access the main table.

```sql
-- Example: Covering index for common query
CREATE INDEX idx_orders_user_status_total
ON orders(user_id, status) INCLUDE (total_amount, order_date);

-- Query that uses covering index (no table access)
SELECT user_id, total_amount, order_date
FROM orders 
WHERE user_id = 123 AND status = 'completed';
```

### Partial Indexes
Index only relevant subsets of data to reduce index size and improve maintenance.

```sql
-- Index only active records
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- Index for recent data only
CREATE INDEX idx_recent_orders ON orders(order_date) WHERE order_date > CURRENT_DATE - INTERVAL '30 days';

-- Index for high-value customers
CREATE INDEX idx_premium_customers ON orders(customer_id) 
WHERE total_amount > 1000;
```

### Function-Based Indexes
Index computed values to optimize queries with functions in WHERE clauses.

```sql
-- Index for case-insensitive searches
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- Index for date truncation
CREATE INDEX idx_orders_month ON orders(DATE_TRUNC('month', order_date));

-- Index for JSONB operations
CREATE INDEX idx_orders_metadata_category ON orders((metadata->>'category'));

-- Index for array containment
CREATE INDEX idx_products_tags ON products USING GIN (tags);
```

### Expression Indexes
Index expressions rather than raw column values.

```sql
-- Index for calculated fields
CREATE INDEX idx_order_value_tier ON orders(
    CASE 
        WHEN total_amount < 100 THEN 'low'
        WHEN total_amount < 1000 THEN 'medium'
        ELSE 'high'
    END
);

-- Index for normalized values
CREATE INDEX idx_normalized_age ON users(age / 10);
```

## Multi-Column Index Design

### Column Order Principles
1. **Selectivity first**: Most selective columns first
2. **Equality before range**: Equality conditions before range conditions
3. **Sorting requirements**: Columns used in ORDER BY last
4. **JOIN conditions**: Foreign keys for JOINs

### Example - Poor vs Good Index Design
```sql
-- Bad: Low-selectivity column first
CREATE INDEX idx_orders_status_user ON orders(status, user_id);

-- Good: High-selectivity column first  
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
```

### Composite Index Strategies
- **Prefix indexes**: Index only first N characters of long strings
- **Trigram indexes**: For fuzzy matching and similarity search
- **BRIN indexes**: For naturally ordered data (time-series)

```sql
-- Prefix index for long text
CREATE INDEX idx_users_name_prefix ON users(name varchar_pattern_ops) WITH (length=10);

-- Trigram index for similarity search
CREATE EXTENSION pg_trgm;
CREATE INDEX idx_users_name_trgm ON users USING GIN (name gin_trgm_ops);

-- BRIN index for time-series
CREATE INDEX idx_measurements_time_brin ON measurements USING BRIN (recorded_at);
```

## AI/ML Specific Indexing Patterns

### Vector Indexing
Specialized indexes for high-dimensional vector data used in AI/ML applications.

#### IVF (Inverted File)
- Partition vectors into clusters
- Search within nearest clusters
- Good balance of speed and accuracy

#### HNSW (Hierarchical Navigable Small World)
- Graph-based indexing
- Excellent recall at reasonable speed
- Memory-intensive but very fast

#### PQ (Product Quantization)
- Compress vectors for memory efficiency
- Trade-off between accuracy and storage
- Good for large-scale vector search

```sql
-- pgvector examples
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
```

### Time-Series Indexing
Optimized for time-based queries common in AI/ML monitoring and IoT applications.

```sql
-- TimescaleDB automatic indexing
CREATE TABLE sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    device_id UUID NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

-- Automatic partitioning and indexing
SELECT create_hypertable('sensor_readings', 'time');

-- Custom time-based indexing
CREATE INDEX idx_sensor_readings_device_time ON sensor_readings(device_id, recorded_at DESC);
CREATE INDEX idx_sensor_readings_time ON sensor_readings(recorded_at) WHERE recorded_at > NOW() - INTERVAL '7 days';
```

### Feature Store Indexing
Optimize for feature retrieval patterns in ML workflows.

```sql
-- Feature value indexing
CREATE INDEX idx_feature_values_entity_time ON feature_values(entity_id, timestamp DESC);
CREATE INDEX idx_feature_values_feature_time ON feature_values(feature_id, timestamp DESC);

-- Composite index for common queries
CREATE INDEX idx_feature_values_entity_feature_time 
ON feature_values(entity_id, feature_id, timestamp DESC);
```

## Index Maintenance and Monitoring

### Index Usage Analysis
Identify unused or inefficient indexes:

```sql
-- Unused indexes
SELECT indexrelname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY idx_tup_read DESC;

-- Index bloat analysis
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    CASE 
        WHEN idx_tup_read = 0 THEN 0
        ELSE (idx_tup_fetch::numeric / idx_tup_read::numeric) * 100
    END as bloat_ratio
FROM pg_stat_user_indexes
WHERE idx_tup_read > 0
ORDER BY bloat_ratio DESC;
```

### Index Rebuild Strategies
- **REINDEX**: Full rebuild (blocks writes)
- **REINDEX CONCURRENTLY**: Online rebuild (PostgreSQL 9.5+)
- **DROP and CREATE**: Manual control over timing

```sql
-- Online index rebuild
REINDEX INDEX CONCURRENTLY idx_orders_user_id;

-- Scheduled maintenance
CREATE OR REPLACE FUNCTION maintain_indexes()
RETURNS VOID AS $$
BEGIN
    -- Rebuild fragmented indexes
    PERFORM pg_advisory_xact_lock(1);
    EXECUTE 'REINDEX INDEX CONCURRENTLY idx_orders_user_id';
    EXECUTE 'REINDEX INDEX CONCURRENTLY idx_users_email';
END;
$$ LANGUAGE plpgsql;
```

## Performance Testing Framework

### Benchmarking Methodology
1. **Baseline measurement**: Current performance
2. **Index creation**: Apply new indexes
3. **Regression testing**: Ensure no regressions
4. **Load testing**: Simulate production load
5. **Monitoring**: Track real-world performance

### Example Benchmark Script
```sql
-- Create test harness
CREATE OR REPLACE FUNCTION benchmark_query(query_text TEXT)
RETURNS TABLE(execution_time_ms NUMERIC, rows_returned BIGINT) AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    result RECORD;
BEGIN
    start_time := clock_timestamp();
    
    EXECUTE query_text INTO result;
    
    end_time := clock_timestamp();
    
    RETURN QUERY SELECT 
        EXTRACT(EPOCH FROM (end_time - start_time)) * 1000 as execution_time_ms,
        COALESCE(result.rows, 0) as rows_returned;
END;
$$ LANGUAGE plpgsql;

-- Test common queries
SELECT * FROM benchmark_query('SELECT * FROM orders WHERE user_id = 123 AND status = ''completed''');
```

## Best Practices

1. **Monitor index usage**: Remove unused indexes regularly
2. **Test before deploying**: Validate performance impact
3. **Consider write overhead**: Indexes slow down writes
4. **Balance selectivity vs size**: Optimize cost/benefit ratio
5. **Document index purpose**: Why each index exists
6. **Review quarterly**: Index effectiveness changes with data growth

## Related Resources

- [Indexing Fundamentals] - Basic indexing concepts
- [Query Optimization] - How indexes affect query plans
- [Vector Databases] - Specialized indexing for AI/ML
- [Performance Tuning] - Comprehensive performance optimization