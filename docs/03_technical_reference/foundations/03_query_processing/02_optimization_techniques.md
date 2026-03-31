# Query Optimization Techniques

This document covers practical techniques for optimizing database queries, essential for senior AI/ML engineers building high-performance data pipelines and real-time inference systems.

## Overview

Query optimization is the process of improving query performance through various techniques including indexing, query rewriting, schema design, and configuration tuning. For AI/ML applications, optimized queries are critical for training data preparation, real-time inference, and model monitoring.

## Index Optimization

### Covering Indexes
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
```

### Function-Based Indexes
Index computed values to optimize queries with functions in WHERE clauses.

```sql
-- Index for case-insensitive searches
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- Index for date truncation
CREATE INDEX idx_orders_month ON orders(DATE_TRUNC('month', order_date));
```

## Query Rewriting Techniques

### Avoid SELECT *
Always specify only needed columns to reduce I/O and network transfer.

```sql
-- Bad: Select all columns
SELECT * FROM large_table WHERE condition;

-- Good: Select only needed columns
SELECT id, name, created_at FROM large_table WHERE condition;
```

### Use EXISTS Instead of COUNT for Existence Checks
```sql
-- Bad: Count all matching rows
IF (SELECT COUNT(*) FROM users WHERE email = 'test@example.com') > 0 THEN ...

-- Good: Stop after finding first match
IF EXISTS (SELECT 1 FROM users WHERE email = 'test@example.com') THEN ...
```

### Optimize JOINs
- **Join order**: Let the optimizer choose, but ensure proper indexes
- **Avoid Cartesian products**: Always specify join conditions
- **Use appropriate join types**: INNER vs LEFT vs RIGHT

```sql
-- Good: Properly indexed joins
EXPLAIN SELECT u.name, o.total_amount
FROM users u
INNER JOIN orders o ON u.user_id = o.user_id
WHERE u.created_at > '2024-01-01';

-- Bad: Missing join condition (Cartesian product)
SELECT * FROM table1 t1, table2 t2; -- No ON clause!
```

## Materialized Views and Caching

### Materialized Views
Pre-compute expensive aggregations and joins for fast querying.

```sql
-- Create materialized view for daily analytics
CREATE MATERIALIZED VIEW daily_user_metrics AS
SELECT 
    DATE(created_at) as day,
    COUNT(*) as new_users,
    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active_users,
    AVG(age) as avg_age
FROM users
GROUP BY DATE(created_at);

-- Refresh strategy
CREATE OR REPLACE FUNCTION refresh_daily_metrics()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_user_metrics;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh (e.g., via cron or database scheduler)
```

### Query Caching
Leverage database query caches and application-level caching.

```sql
-- Database-level: Enable query cache (PostgreSQL)
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '16GB';

-- Application-level: Redis cache for frequent queries
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

def get_user_with_cache(user_id):
    cache_key = f"user:{user_id}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Query database
    user = db.query("SELECT * FROM users WHERE user_id = %s", [user_id])
    
    # Cache result
    r.setex(cache_key, 300, json.dumps(user))  # 5 minute TTL
    return user
```

## Configuration Tuning

### Memory Settings
Optimize memory allocation for query performance.

```sql
-- PostgreSQL memory settings
ALTER SYSTEM SET shared_buffers = '4GB';           -- Shared memory buffer pool
ALTER SYSTEM SET work_mem = '64MB';                -- Memory per sort/hash operation
ALTER SYSTEM SET maintenance_work_mem = '1GB';     -- Memory for maintenance operations
ALTER SYSTEM SET effective_cache_size = '16GB';    -- OS cache estimate
```

### Parallel Query Settings
Enable parallel query execution for large datasets.

```sql
-- PostgreSQL parallel query settings
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET parallel_workers = 8;
ALTER SYSTEM SET parallel_setup_cost = 1000;
ALTER SYSTEM SET parallel_tuple_cost = 0.1;
```

## AI/ML Specific Optimization Patterns

### Training Data Preparation
Optimize queries that generate training datasets.

```sql
-- Bad: Multiple sequential queries
SELECT * FROM users WHERE created_at < '2024-01-01';
SELECT * FROM orders WHERE user_id IN (...);
SELECT * FROM features WHERE user_id IN (...);

-- Good: Single optimized query with joins
EXPLAIN SELECT 
    u.user_id,
    u.age,
    u.gender,
    COUNT(o.order_id) as purchase_count,
    SUM(o.total_amount) as total_spent,
    ARRAY_AGG(DISTINCT p.category) as purchased_categories
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE u.created_at < '2024-01-01'
GROUP BY u.user_id, u.age, u.gender;
```

### Real-time Inference Optimization
Ensure low-latency queries for production inference.

```sql
-- Use covering indexes for fast lookups
CREATE INDEX idx_model_versions_latest 
ON model_versions(model_id, created_at DESC) 
INCLUDE (artifact_path, metrics, hyperparameters);

-- Query for latest model version
SELECT artifact_path, metrics, hyperparameters
FROM model_versions
WHERE model_id = 'recommendation-model-1'
ORDER BY created_at DESC
LIMIT 1;
```

### Feature Store Optimization
Optimize time-series feature queries.

```sql
-- Partitioned table for time-series features
CREATE TABLE feature_values (
    feature_id UUID NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (feature_id, entity_id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create partitions for time ranges
CREATE TABLE feature_values_2024_q1 PARTITION OF feature_values
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

-- Index for time-range queries
CREATE INDEX idx_feature_values_time ON feature_values(timestamp) 
WHERE timestamp > CURRENT_DATE - INTERVAL '30 days';
```

## Performance Monitoring and Diagnostics

### Query Performance Metrics
Monitor key metrics to identify optimization opportunities:

```sql
-- Slow query log (PostgreSQL)
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s

-- Query statistics
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    rows,
    shared_blks_hit,
    shared_blks_read
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

### Index Usage Statistics
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
    bloat_ratio
FROM (
    SELECT 
        schemaname,
        tablename,
        indexname,
        indexrelid,
        CASE 
            WHEN idx_tup_read = 0 THEN 0
            ELSE (idx_tup_fetch::numeric / idx_tup_read::numeric) * 100
        END as bloat_ratio
    FROM pg_stat_user_indexes
) subquery
WHERE bloat_ratio > 50;
```

## Optimization Checklist

### Before Optimization
1. **Measure baseline performance** using EXPLAIN ANALYZE
2. **Identify bottlenecks** (seq scans, sorts, hash tables)
3. **Understand query patterns** and business requirements

### During Optimization
1. **Add appropriate indexes** (covering, partial, function-based)
2. **Rewrite queries** for better execution plans
3. **Tune configuration** for workload characteristics
4. **Consider materialized views** for expensive aggregations

### After Optimization
1. **Verify performance improvement**
2. **Test under load** to ensure scalability
3. **Monitor long-term performance**
4. **Document changes** for future reference

## Related Resources

- [Execution Plans] - Understanding query execution strategies
- [Indexing Strategies] - Creating effective indexes
- [Database Performance Tuning] - Comprehensive performance optimization
- [AI/ML Query Patterns] - Specialized patterns for machine learning applications