# Partitioning Techniques

Partitioning is a database-level technique for dividing large tables into smaller, more manageable pieces. Unlike sharding (which distributes across servers), partitioning occurs within a single database instance.

## Overview

Partitioning improves query performance, manageability, and availability by organizing data into logical segments. For AI/ML applications dealing with large datasets, effective partitioning is crucial for maintaining performance as data grows.

## Partitioning Types

### Range Partitioning
- **Method**: Data divided by ranges of values in a column
- **Example**: Time-based partitioning (by date/month/year)
- **Advantages**: Efficient range queries, natural data organization
- **Disadvantages**: Can create hotspots if ranges are uneven

### List Partitioning
- **Method**: Data divided by explicit list of values
- **Example**: Partition by region, category, or status
- **Advantages**: Precise control over data placement
- **Disadvantages**: Manual management, limited scalability

### Hash Partitioning
- **Method**: Hash function on partition key determines partition
- **Example**: `partition_id = hash(user_id) % num_partitions`
- **Advantages**: Even data distribution, simple implementation
- **Disadvantages**: Inefficient for range queries, difficult to add partitions

### Composite Partitioning
- **Method**: Combination of multiple partitioning strategies
- **Example**: Range + Hash (time range + hash of user_id)
- **Advantages**: Best of both worlds, flexible organization
- **Disadvantages**: Complex setup and management

## Implementation Examples

### PostgreSQL Partitioning
```sql
-- Range partitioning by date
CREATE TABLE measurements (
    id BIGSERIAL PRIMARY KEY,
    device_id INT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL
) PARTITION BY RANGE (recorded_at);

-- Create partitions
CREATE TABLE measurements_2024_q1 PARTITION OF measurements
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE measurements_2024_q2 PARTITION OF measurements
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Automatic partition creation (PostgreSQL 12+)
CREATE OR REPLACE FUNCTION create_partition()
RETURNS TRIGGER AS $$
DECLARE
    partition_name TEXT;
BEGIN
    partition_name := 'measurements_' || TO_CHAR(NEW.recorded_at, 'YYYY_QQ');
    
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = partition_name) THEN
        EXECUTE format('CREATE TABLE %I PARTITION OF measurements FOR VALUES FROM (%L) TO (%L)',
            partition_name, 
            DATE_TRUNC('quarter', NEW.recorded_at),
            DATE_TRUNC('quarter', NEW.recorded_at) + INTERVAL '3 months');
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER create_partition_trigger
BEFORE INSERT ON measurements
FOR EACH ROW EXECUTE FUNCTION create_partition();
```

### TimescaleDB Hypertables
```sql
-- TimescaleDB automatically handles time-based partitioning
CREATE TABLE sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    device_id UUID NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION
);

-- Convert to hypertable (automatic partitioning)
SELECT create_hypertable('sensor_readings', 'time');

-- Add compression for older data
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id'
);

-- Add compression policy
SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');
```

### MongoDB Sharding (Partitioning at cluster level)
```javascript
// Enable sharding for database
sh.enableSharding("ai_ml_db");

// Shard collection by user_id
sh.shardCollection("ai_ml_db.feature_values", { "user_id": 1 });

// Add shards to cluster
sh.addShard("shard1:27017");
sh.addShard("shard2:27017");
sh.addShard("shard3:27017");
```

## AI/ML Specific Partitioning Patterns

### Time-Series Data
- **Time-based partitioning**: By day, week, month, quarter
- **Hot/cold separation**: Recent data (hot) vs historical (cold)
- **Compression**: Compress older partitions for storage efficiency
- **Retention policies**: Automatically drop old partitions

### Feature Store Partitioning
- **By entity type**: Users, products, sessions
- **By feature group**: Numerical, categorical, embedding features
- **By time window**: Real-time vs batch features
- **By model version**: Separate partitions for different model versions

### Model Registry Partitioning
- **By model type**: Classification, regression, NLP
- **By team/department**: Isolation between teams
- **By environment**: Development, staging, production
- **By usage pattern**: High-frequency vs low-frequency models

## Performance Benefits

### Query Performance
- **Partition pruning**: Database eliminates irrelevant partitions
- **Parallel processing**: Multiple partitions processed concurrently
- **Index efficiency**: Smaller indexes per partition
- **Cache efficiency**: Hot partitions fit in memory

### Maintenance Operations
- **Faster VACUUM/ANALYZE**: Per-partition operations
- **Online maintenance**: Work on one partition while others serve
- **Data lifecycle management**: Drop old partitions instead of DELETE
- **Backup/restore**: Per-partition operations

### Example - Partition Pruning Analysis
```sql
-- Without partitioning
EXPLAIN SELECT * FROM large_table WHERE created_at > '2024-01-01';

-- With partitioning
EXPLAIN SELECT * FROM measurements WHERE recorded_at > '2024-01-01';
-- Output shows: "Partition Pruning: enabled"
-- Only relevant partitions scanned
```

## Best Practices

### Design Considerations
1. **Choose partition key carefully**: Should match query patterns
2. **Size partitions appropriately**: 10-100GB typical range
3. **Plan for growth**: Number of partitions should scale with data
4. **Consider maintenance**: Automated partition creation/deletion
5. **Monitor partition balance**: Prevent hotspots

### Implementation Guidelines
- **Start with range partitioning**: Most common and useful
- **Use composite partitioning** for complex workloads
- **Implement automated partition management**: Avoid manual intervention
- **Test query patterns**: Ensure partitioning helps your use cases
- **Monitor performance**: Track before/after partitioning benefits

### Common Mistakes
- **Too many partitions**: Overhead from metadata management
- **Too few partitions**: No performance benefit
- **Poor partition key**: Doesn't match query patterns
- **Ignoring maintenance**: Partitions become unmanageable
- **Forgetting statistics**: Outdated stats hurt query planning

## Advanced Techniques

### Partitioned Indexes
- **Local indexes**: Index per partition (faster maintenance)
- **Global indexes**: Single index across all partitions (better for cross-partition queries)
- **Partial indexes**: Index only relevant partitions

### Materialized Views with Partitioning
```sql
-- Partitioned materialized view for daily aggregates
CREATE MATERIALIZED VIEW daily_metrics PARTITION BY RANGE (day) AS
SELECT 
    DATE(recorded_at) as day,
    device_id,
    COUNT(*) as readings_count,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value
FROM sensor_readings
GROUP BY DATE(recorded_at), device_id;

-- Create partitions for materialized view
CREATE TABLE daily_metrics_2024_q1 PARTITION OF daily_metrics
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
```

## Related Resources

- [Sharding Strategies] - Horizontal partitioning across servers
- [Indexing Strategies] - How partitioning affects indexing
- [Query Optimization] - Partition-aware query optimization
- [Time-Series Databases] - Specialized partitioning for time-series data