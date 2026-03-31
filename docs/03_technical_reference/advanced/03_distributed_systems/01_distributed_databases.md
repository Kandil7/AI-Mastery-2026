# Distributed Databases

Distributed databases are database systems that store and process data across multiple nodes, providing scalability, fault tolerance, and high availability. They are essential for building modern AI/ML systems that require handling massive datasets and high throughput.

## Overview
Distributed databases are designed to handle large amounts of data and high traffic. They are built to scale horizontally, meaning they can add more nodes to increase capacity. This makes them suitable for handling large datasets and high throughput.

Distributed databases address the limitations of single-node systems by:
- **Horizontal scaling**: Add nodes to increase capacity
- **Fault tolerance**: Survive node failures
- **Geographic distribution**: Low-latency access from multiple regions
- **High availability**: Continuous operation during maintenance

For senior AI/ML engineers, understanding distributed database architectures is critical for building production-scale AI systems.

## Core Concepts

### CAP Theorem
- **Consistency**: All nodes see the same data at the same time
- **Availability**: Every request receives a response (success/failure)
- **Partition tolerance**: System continues operating despite network partitions
- **Trade-offs**: Choose 2 out of 3 in practice

### Consistency Models
- **Strong consistency**: Linearizable reads/writes
- **Eventual consistency**: Updates propagate eventually
- **Causal consistency**: Preserve causal relationships
- **Session consistency**: Per-session guarantees

### Replication Strategies
- **Synchronous replication**: Wait for all replicas before acknowledging
- **Asynchronous replication**: Acknowledge immediately, replicate later
- **Quorum-based**: Require majority of replicas for writes/reads
- **Multi-leader**: Multiple nodes can accept writes

## Popular Distributed Database Systems

### Cassandra
- **Architecture**: Masterless, peer-to-peer
- **Data model**: Wide-column store
- **Consistency**: Tunable consistency levels
- **Use cases**: Time-series data, IoT, high-write workloads

### CockroachDB
- **Architecture**: SQL database with strong consistency
- **Data model**: Relational with distributed transactions
- **Consistency**: Strong consistency with Raft consensus
- **Use cases**: Financial systems, e-commerce, applications requiring ACID

### MongoDB Atlas
- **Architecture**: Document database with sharding
- **Data model**: JSON-like documents
- **Consistency**: Tunable consistency levels
- **Use cases**: Content management, user profiles, real-time analytics

### TiDB
- **Architecture**: MySQL-compatible distributed SQL
- **Data model**: Relational with horizontal scaling
- **Consistency**: Strong consistency with Raft
- **Use cases**: OLTP workloads, hybrid transactional/analytical processing

## Database Design Patterns

### Sharding Strategies
```sql
-- Range-based sharding (by ID)
CREATE TABLE users_sharded (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    created_at TIMESTAMPTZ
);

-- Shard 0: id < 1000000
-- Shard 1: 1000000 <= id < 2000000
-- Shard 2: 2000000 <= id < 3000000

-- Hash-based sharding (more balanced)
-- shard_id = hash(id) % num_shards

-- Directory-based sharding (flexible)
CREATE TABLE shard_directory (
    entity_type VARCHAR(50),
    entity_id VARCHAR(255),
    shard_id INT NOT NULL,
    PRIMARY KEY (entity_type, entity_id)
);
```

### Multi-Region Deployment
```sql
-- Cross-region replication patterns
-- Active-active: Both regions accept writes
-- Active-passive: One region primary, others read-only
-- Read-replicas: Primary region for writes, others for reads

-- Example: Global deployment strategy
CREATE TABLE global_config (
    region_code CHAR(2) PRIMARY KEY,
    endpoint_url TEXT NOT NULL,
    latency_ms INT NOT NULL,
    status VARCHAR(20) DEFAULT 'active'
);

-- Insert regional endpoints
INSERT INTO global_config VALUES 
('us-east', 'https://us-east.db.example.com', 10, 'active'),
('eu-west', 'https://eu-west.db.example.com', 45, 'active'),
('ap-south', 'https://ap-south.db.example.com', 85, 'active');
```

### Distributed Transactions
```sql
-- Two-phase commit (2PC) pattern
BEGIN TRANSACTION;
-- Phase 1: Prepare
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Phase 2: Commit
COMMIT;

-- Optimistic concurrency control
UPDATE accounts 
SET balance = balance - 100, 
    version = version + 1 
WHERE id = 1 AND version = 5;

-- Check if update succeeded (row count > 0)
-- If not, retry with new version
```

## Performance Optimization

### Query Routing
- **Smart routing**: Route queries to optimal nodes
- **Local execution**: Execute queries on nodes holding relevant data
- **Cross-shard joins**: Minimize when possible, use denormalization
- **Caching**: Local caches per node

### Indexing Strategies
- **Local indexes**: Indexes on individual shards
- **Global indexes**: Distributed index structures
- **Covering indexes**: Include frequently accessed columns
- **Partial indexes**: For hot data subsets

### Load Balancing
- **Request routing**: Distribute queries evenly
- **Connection pooling**: Efficient resource management
- **Auto-scaling**: Add/remove nodes based on load
- **Health checks**: Monitor node health and redirect traffic

## AI/ML Specific Considerations

### Large-Scale Training Data
- **Distributed storage**: Store training datasets across nodes
- **Parallel processing**: Process data in parallel across cluster
- **Data locality**: Co-locate computation with data
- **Streaming pipelines**: Real-time data ingestion for online learning

### Model Serving Infrastructure
- **Distributed inference**: Scale model serving across nodes
- **Model versioning**: Store multiple model versions
- **A/B testing**: Route traffic to different model versions
- **Canary deployments**: Gradual rollout of new models

### Real-time Analytics
- **Stream processing**: Process data as it arrives
- **Windowed aggregations**: Time-based calculations
- **Stateful processing**: Maintain state across events
- **Exactly-once semantics**: Ensure data processing reliability

## Implementation Examples

### Cassandra Schema Design
```cql
-- Time-series data with wide rows
CREATE TABLE sensor_readings (
    device_id UUID,
    reading_time TIMESTAMP,
    temperature DOUBLE,
    humidity DOUBLE,
    pressure DOUBLE,
    PRIMARY KEY (device_id, reading_time)
) WITH CLUSTERING ORDER BY (reading_time DESC);

-- Materialized view for fast time-range queries
CREATE MATERIALIZED VIEW sensor_readings_by_date AS
SELECT * FROM sensor_readings
WHERE device_id IS NOT NULL AND reading_time IS NOT NULL
PRIMARY KEY ((device_id, date(reading_time)), reading_time);

-- Secondary index for filtering by location
CREATE INDEX ON sensor_readings (location);
```

### CockroachDB Distributed Transactions
```sql
-- Distributed transaction with strong consistency
BEGIN;
-- This transaction will be distributed across multiple nodes
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
UPDATE transactions SET status = 'completed' WHERE id = 123;
COMMIT;

-- Read-only transaction with follower reads
BEGIN READ ONLY;
SELECT * FROM users WHERE last_login > NOW() - INTERVAL '1 day';
COMMIT;
```

### MongoDB Sharding Configuration
```javascript
// Enable sharding for database
sh.enableSharding("analytics_db");

// Shard collection by hashed _id
sh.shardCollection("analytics_db.events", { "_id": "hashed" });

// Shard collection by range (time-based)
sh.shardCollection("analytics_db.logs", { "timestamp": 1 });

// Add shards
sh.addShard("shard1.example.com:27017");
sh.addShard("shard2.example.com:27017");
sh.addShard("shard3.example.com:27017");

// Configure zones for geographic distribution
sh.addShardTag("shard1.example.com:27017", "us-east");
sh.addShardTag("shard2.example.com:27017", "eu-west");
sh.addShardTag("shard3.example.com:27017", "ap-south");

sh.updateZoneKeyRange(
    "analytics_db.events",
    { "region": "us-east" },
    { "region": "us-east" },
    "us-east"
);
```

## Best Practices

1. **Design for failure**: Assume nodes will fail and design accordingly
2. **Start with appropriate consistency**: Choose consistency model based on requirements
3. **Monitor cross-node operations**: Track latency and errors in distributed operations
4. **Implement proper backup**: Distributed backups across regions
5. **Test under failure conditions**: Simulate network partitions and node failures
6. **Consider hybrid approaches**: Combine distributed databases with caching layers

## Related Resources

- [CAP Theorem] - Deep dive into consistency trade-offs
- [Database Replication] - Advanced replication patterns
- [AI/ML System Design] - Distributed databases in ML architecture
- [Scalability Patterns] - General scalability best practices