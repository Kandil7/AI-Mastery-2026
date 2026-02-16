# Sharding Strategies

Sharding is a horizontal partitioning technique that distributes data across multiple database instances. It's essential for scaling databases beyond single-node limitations, particularly important for AI/ML applications handling large datasets.

## Overview

Sharding involves splitting a logical database into smaller, manageable pieces (shards) distributed across multiple physical servers. For senior AI/ML engineers, understanding sharding strategies is critical for building scalable data platforms.

## Sharding Approaches

### Key-Based Sharding (Hash Sharding)
- **Method**: Hash function on shard key determines shard assignment
- **Example**: `shard_id = hash(user_id) % num_shards`
- **Advantages**: Even data distribution, simple routing
- **Disadvantages**: Resharding complexity, difficult range queries

### Range-Based Sharding
- **Method**: Data partitioned by ranges of shard key values
- **Example**: Users 0000-9999 → Shard 1, 10000-19999 → Shard 2
- **Advantages**: Efficient range queries, predictable data placement
- **Disadvantages**: Hotspots possible, uneven distribution

### Directory-Based Sharding
- **Method**: Lookup table maps keys to shards
- **Example**: Central directory service or metadata table
- **Advantages**: Flexible routing, easy resharding
- **Disadvantages**: Single point of failure, additional latency

### Geohash-Based Sharding
- **Method**: Geographic coordinates converted to geohash for spatial partitioning
- **Example**: `geohash(latitude, longitude, precision)`
- **Advantages**: Locality-aware, good for location-based services
- **Disadvantages**: Complex implementation, uneven distribution

## Shard Key Selection

### Good Shard Keys
- **High cardinality**: Many unique values
- **Uniform distribution**: Even data spread
- **Query patterns**: Matches common access patterns
- **Stable**: Rarely changes

### Bad Shard Keys
- **Low cardinality**: Few unique values (e.g., gender)
- **Skewed distribution**: Most data in few shards
- **Frequently changing**: Causes data movement
- **Unknown at write time**: Hard to route writes

### AI/ML Specific Considerations
- **User ID**: Good for personalized ML models
- **Time-based**: Good for time-series ML workloads
- **Model ID**: Good for model registry systems
- **Feature groups**: Good for feature store partitioning

## Implementation Patterns

### Application-Level Sharding
- **Routing logic**: In application code
- **Connection management**: Application handles connections to shards
- **Transaction coordination**: Application manages cross-shard transactions
- **Advantages**: Full control, flexible
- **Disadvantages**: Complex application logic, harder to maintain

### Proxy-Based Sharding
- **Routing layer**: Database proxy (e.g., Vitess, Citus)
- **Transparent**: Applications unaware of sharding
- **Transaction management**: Proxy handles cross-shard operations
- **Advantages**: Simpler application code, easier migration
- **Disadvantages**: Additional infrastructure, potential bottleneck

### Database-Native Sharding
- **Built-in**: Database system handles sharding (e.g., MongoDB, Cassandra)
- **Configuration**: Declarative sharding setup
- **Management**: Database handles rebalancing, failover
- **Advantages**: Integrated solution, optimized performance
- **Disadvantages**: Vendor lock-in, limited flexibility

## Cross-Shard Operations

### Distributed Transactions
- **Two-Phase Commit (2PC)**: Strong consistency, high latency
- **Saga Pattern**: Eventual consistency, compensating transactions
- **CRDTs**: Conflict-free replicated data types
- **Application-level coordination**: Custom business logic

### Query Patterns
- **Single-shard queries**: Fast, efficient
- **Multi-shard queries**: Require aggregation, slower
- **Broadcast queries**: Send to all shards, expensive
- **Scatter-gather**: Query multiple shards, combine results

### Example - Multi-Shard Query Pattern
```sql
-- Application-level scatter-gather
async function get_user_orders(userId) {
    const shardId = calculateShard(userId);
    const orders = await db.query(`SELECT * FROM orders WHERE user_id = $1`, [userId]);
    
    // If we need cross-shard data, query multiple shards
    if (needsCrossShardData) {
        const allShards = await Promise.all(
            Array.from({length: numShards}, (_, i) => 
                db.query(`SELECT * FROM orders WHERE user_id % ${numShards} = ${i} AND user_id = $1`, [userId])
            )
        );
        return allShards.flat();
    }
    
    return orders;
}
```

## AI/ML Specific Sharding Patterns

### Feature Store Sharding
- **By entity type**: Users, products, sessions
- **By time**: Recent vs historical features
- **By feature group**: Numerical, categorical, embedding features
- **Hybrid**: Combination of above

### Model Registry Sharding
- **By model type**: Classification, regression, NLP
- **By team/department**: Isolation between teams
- **By environment**: Development, staging, production
- **By usage pattern**: High-frequency vs low-frequency models

### Training Data Sharding
- **By time window**: Daily, weekly partitions
- **By data source**: Different ingestion pipelines
- **By feature importance**: Critical vs non-critical features
- **By model version**: Separate shards for different model versions

## Performance Optimization

### Shard Size Management
- **Optimal size**: 50-100GB per shard (PostgreSQL), 10-50GB (MongoDB)
- **Too small**: Overhead from too many shards
- **Too large**: Loss of scalability benefits
- **Monitoring**: Track shard sizes and growth rates

### Query Optimization
- **Shard-aware indexing**: Indexes within each shard
- **Local joins**: Prefer joins within same shard
- **Materialized views**: Pre-aggregate cross-shard data
- **Caching**: Cache frequently accessed cross-shard results

## Challenges and Solutions

### Resharding
- **Online resharding**: Move data without downtime
- **Consistent hashing**: Minimize data movement
- **Double-write**: Write to both old and new shards during transition
- **Proxy routing**: Handle old/new shard mappings

### Data Consistency
- **Eventual consistency**: Accept temporary inconsistencies
- **Quorum reads/writes**: Ensure majority agreement
- **Version vectors**: Track causal relationships
- **Conflict resolution**: Application-level merging

## Best Practices

1. **Start simple**: Begin with single node, add sharding when needed
2. **Choose shard key carefully**: Most important decision
3. **Monitor shard balance**: Prevent hotspots
4. **Plan for resharding**: Design with future growth in mind
5. **Test cross-shard operations**: Ensure they work correctly
6. **Consider hybrid approaches**: Vertical scaling + horizontal sharding

## Related Resources

- [Partitioning Techniques] - Complementary partitioning strategies
- [Distributed Transactions] - Handling transactions across shards
- [CAP Theorem Applications] - Consistency trade-offs in sharded systems
- [Database Scaling Patterns] - Comprehensive scaling guide