# Sharding Strategies

Sharding is the process of horizontally partitioning data across multiple database instances to achieve scalability, fault tolerance, and improved performance. For senior AI/ML engineers, understanding sharding strategies is essential for building systems that can handle massive datasets and high throughput.

## Overview

Sharding distributes data based on a shard key, allowing databases to scale beyond single-node limitations. The choice of sharding strategy significantly impacts performance, scalability, and operational complexity.

## Core Sharding Concepts

### Shard Key Selection
- **High cardinality**: Many unique values for good distribution
- **Uniform distribution**: Avoid hotspots and uneven load
- **Query patterns**: Align with common access patterns
- **Growth considerations**: Handle future data growth

### Shard Types
- **Range-based**: Data partitioned by value ranges
- **Hash-based**: Data partitioned by hash of shard key
- **Directory-based**: Lookup table maps keys to shards
- **Geographic**: Partition by geographic location

## Sharding Strategies

### Range-Based Sharding
- **Implementation**: Partition by ordered values (IDs, timestamps)
- **Pros**: Efficient range queries, natural ordering
- **Cons**: Hotspots at boundaries, uneven distribution
- **Use cases**: Time-series data, sequential IDs

```sql
-- Example: User ID range sharding
-- Shard 0: user_id < 1000000
-- Shard 1: 1000000 <= user_id < 2000000  
-- Shard 2: 2000000 <= user_id < 3000000

-- Query optimization
SELECT * FROM users WHERE user_id BETWEEN 500000 AND 1500000;
-- This query spans two shards (shard 0 and shard 1)
```

### Hash-Based Sharding
- **Implementation**: Hash function determines shard assignment
- **Pros**: Uniform distribution, no hotspots
- **Cons**: Inefficient range queries, requires all shards for full scans
- **Use cases**: User sessions, random access patterns

```sql
-- Hash function example
function get_shard_id(key, num_shards) {
    return hash(key) % num_shards;
}

-- Example: User email hashing
-- shard_id = hash('user@example.com') % 4
-- Results in uniform distribution across 4 shards

-- Query pattern implications
SELECT * FROM users WHERE email = 'user@example.com';
-- Efficient: goes to single shard
SELECT * FROM users WHERE created_at > '2024-01-01';
-- Inefficient: must query all shards
```

### Directory-Based Sharding
- **Implementation**: Central lookup table maps keys to shards
- **Pros**: Flexible re-sharding, supports complex routing
- **Cons**: Single point of failure, additional latency
- **Use cases**: Multi-tenant applications, complex routing rules

```sql
-- Directory table schema
CREATE TABLE shard_directory (
    entity_type VARCHAR(50),
    entity_id VARCHAR(255),
    shard_id INT NOT NULL,
    PRIMARY KEY (entity_type, entity_id)
);

-- Example entries
INSERT INTO shard_directory VALUES 
('user', 'user_123', 2),
('user', 'user_456', 0),
('tenant', 'company_a', 1),
('tenant', 'company_b', 3);

-- Query routing
SELECT shard_id FROM shard_directory 
WHERE entity_type = 'user' AND entity_id = 'user_123';
```

### Geographic Sharding
- **Implementation**: Partition by geographic location
- **Pros**: Low-latency access, compliance with data residency
- **Cons**: Complex management, uneven data distribution
- **Use cases**: Global applications, regulatory compliance

```sql
-- Geographic shard mapping
CREATE TABLE geo_shards (
    region_code CHAR(2) PRIMARY KEY,
    shard_endpoint TEXT NOT NULL,
    capacity INT DEFAULT 1000000
);

INSERT INTO geo_shards VALUES 
('US', 'us-db.example.com', 5000000),
('EU', 'eu-db.example.com', 3000000),
('AP', 'ap-db.example.com', 2000000);

-- User assignment based on location
UPDATE users SET region_code = 'US' WHERE country = 'United States';
UPDATE users SET region_code = 'EU' WHERE country IN ('Germany', 'France', 'UK');
```

## Advanced Sharding Patterns

### Composite Sharding
- **Implementation**: Combine multiple sharding strategies
- **Example**: Hash + Range (hash for distribution, range for time)
- **Pros**: Best of both worlds, flexible query patterns
- **Cons**: Increased complexity, more sophisticated routing

```sql
-- Composite: Tenant ID (hash) + Timestamp (range)
-- shard_id = hash(tenant_id) % 4
-- Within shard: partition by month

-- Query optimization
SELECT * FROM events 
WHERE tenant_id = 'company_a' 
  AND event_time BETWEEN '2024-01-01' AND '2024-01-31';
-- Routes to single shard, then uses local range partitioning
```

### Dynamic Sharding
- **Implementation**: Automatic shard creation and rebalancing
- **Pros**: Handles growth automatically, no manual intervention
- **Cons**: Complex implementation, potential for rebalancing overhead
- **Use cases**: Cloud-native applications, unpredictable growth

### Hotspot Mitigation
- **Salting**: Add random prefix to shard key to distribute hot keys
- **Pre-splitting**: Create shards in advance for expected growth
- **Tiered sharding**: Separate hot data from cold data

```sql
-- Salting example for high-traffic users
-- Original: user_id = '12345'
-- Salted: shard_key = 'salt_' + hash(user_id + random_salt)

-- Pre-splitting example
-- Create 100 shards initially for expected 1M users
-- Each shard handles ~10K users initially
```

## Database-Specific Implementations

### PostgreSQL with Citus
```sql
-- Citus extension for PostgreSQL
CREATE EXTENSION citus;

-- Create distributed table
SELECT create_distributed_table('users', 'user_id');

-- Automatic sharding by user_id
-- Citus handles routing and query planning

-- Custom sharding configuration
SELECT master_create_distributed_table('events', 'tenant_id', 'hash');
SELECT master_create_worker_node('worker1', '192.168.1.10', 5432);
SELECT master_add_node('worker1', 5432);
```

### MongoDB Sharding
```javascript
// Enable sharding for database
sh.enableSharding("analytics_db");

// Shard collection by hashed _id
sh.shardCollection("analytics_db.events", { "_id": "hashed" });

// Shard collection by range (time-based)
sh.shardCollection("analytics_db.logs", { "timestamp": 1 });

// Configure zones for geographic distribution
sh.addShardTag("shard1.example.com:27017", "us-east");
sh.addShardTag("shard2.example.com:27017", "eu-west");

sh.updateZoneKeyRange(
    "analytics_db.events",
    { "region": "us-east" },
    { "region": "us-east" },
    "us-east"
);
```

### Cassandra Partitioning
```cql
-- Primary key design for optimal partitioning
CREATE TABLE sensor_readings (
    device_id UUID,
    reading_time TIMESTAMP,
    temperature DOUBLE,
    PRIMARY KEY ((device_id, date(reading_time)), reading_time)
);

-- Composite partition key: device_id + date
-- Clustering column: reading_time (descending)
-- Benefits: Even distribution, efficient time-range queries
```

## AI/ML Specific Considerations

### Training Data Sharding
- **Feature sharding**: Distribute features across nodes
- **Sample sharding**: Distribute training samples
- **Model parameter sharding**: Split model parameters across workers
- **Gradient sharding**: Distribute gradient computation

### Real-time Inference Sharding
- **Model version sharding**: Different model versions on different shards
- **User segment sharding**: Different user segments on different shards
- **Request type sharding**: Different inference types on different shards
- **Geographic sharding**: Local models for regional users

### Data Pipeline Sharding
- **ETL sharding**: Parallel processing of data batches
- **Stream processing**: Partition streams for parallel processing
- **CDC sharding**: Distribute change capture across nodes
- **Analytics sharding**: Partition analytical workloads

## Performance Optimization

### Query Routing Optimization
- **Smart routing**: Cache shard mappings locally
- **Connection pooling**: Maintain connections to frequently used shards
- **Batch routing**: Route multiple queries together
- **Local execution**: Execute queries on nodes holding relevant data

### Load Balancing
- **Dynamic rebalancing**: Move data between shards based on load
- **Auto-scaling**: Add/remove shards based on demand
- **Health monitoring**: Redirect traffic from unhealthy shards
- **Capacity planning**: Predict growth and provision ahead of time

### Indexing Strategies
- **Local indexes**: Indexes on individual shards
- **Global indexes**: Distributed index structures
- **Covering indexes**: Include frequently accessed columns
- **Partial indexes**: For hot data subsets

## Best Practices

1. **Choose shard key carefully**: It's difficult to change later
2. **Test with realistic data**: Synthetic data may not reveal hotspot issues
3. **Monitor shard balance**: Track data distribution and query patterns
4. **Plan for rebalancing**: Have strategy for when shards become unbalanced
5. **Implement proper backup**: Distributed backups across shards
6. **Consider hybrid approaches**: Combine sharding with caching layers

## Related Resources

- [Distributed Databases] - Foundation for sharding implementations
- [CAP Theorem] - Understanding trade-offs in distributed systems
- [Database Performance] - General performance optimization techniques
- [AI/ML System Design] - Sharding in ML system architecture