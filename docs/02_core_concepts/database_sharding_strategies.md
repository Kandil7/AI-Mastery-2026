# Database Sharding Strategies: Advanced Partitioning for AI/ML Systems

## Overview

Database sharding is the process of horizontally partitioning data across multiple database instances to improve scalability, performance, and availability. In AI/ML systems, sharding is essential for handling massive datasets, high-throughput inference requests, and distributed training workloads.

## Core Sharding Concepts

### Sharding vs Partitioning
- **Sharding**: Physical separation across different database instances/servers
- **Partitioning**: Logical separation within a single database instance

### Key Considerations for AI/ML Workloads
- **Data locality**: Co-locate related data (e.g., user features with model predictions)
- **Query patterns**: Design shards based on common query access patterns
- **Skew management**: Prevent hotspots in ML training data access

## Horizontal Sharding Strategies

### 1. Range-Based Sharding

#### Architecture
- Data partitioned by key ranges (e.g., user_id 0-1000000 → shard 1)
- Simple to implement but prone to hotspots

#### Implementation Example
```sql
-- PostgreSQL example with declarative partitioning
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    data JSONB
) PARTITION BY RANGE (user_id);

CREATE TABLE users_0_1M PARTITION OF users
    FOR VALUES FROM (0) TO (1000000);

CREATE TABLE users_1M_2M PARTITION OF users
    FOR VALUES FROM (1000000) TO (2000000);
```

#### AI/ML Use Cases
- **User segmentation**: Shard by user ID ranges for personalized recommendations
- **Time-series data**: Shard by timestamp ranges for model monitoring metrics
- **Model versions**: Shard by model version numbers

#### Challenges
- **Hotspot creation**: Popular ranges (e.g., recent timestamps) become bottlenecks
- **Rebalancing complexity**: Moving data between shards is expensive

### 2. Hash-Based Sharding

#### Architecture
- Apply hash function to shard key → determine target shard
- Uniform distribution but poor range queries

#### Consistent Hashing Implementation
```python
class ConsistentHashRing:
    def __init__(self, num_virtual_nodes=100):
        self.ring = {}
        self.sorted_keys = []
        self.num_virtual_nodes = num_virtual_nodes
    
    def add_node(self, node_name):
        for i in range(self.num_virtual_nodes):
            key = hash(f"{node_name}:{i}")
            self.ring[key] = node_name
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key):
        hash_val = hash(key)
        # Binary search for closest node
        idx = bisect.bisect_left(self.sorted_keys, hash_val)
        if idx == len(self.sorted_keys):
            idx = 0
        return self.ring[self.sorted_keys[idx]]
```

#### AI/ML Applications
- **Feature store**: Hash by feature namespace + entity ID
- **Model artifacts**: Hash by model name + version
- **Training logs**: Hash by job ID + timestamp

#### Advantages
- **Uniform distribution**: Minimizes hotspots
- **Scalability**: Adding/removing nodes affects minimal data
- **Load balancing**: Automatic distribution across shards

### 3. Directory-Based Sharding

#### Architecture
- Central directory maps keys to shards
- Most flexible but introduces single point of failure

#### Implementation Pattern
```json
{
  "shard_directory": {
    "user_123456": "shard-3",
    "model_v2.1": "shard-1",
    "feature_user_profile": "shard-2"
  }
}
```

#### Use Cases for AI/ML
- **Hybrid sharding**: Combine with other strategies for complex workloads
- **Dynamic rebalancing**: Update directory without moving data
- **Multi-tenant systems**: Isolate customer data while maintaining flexibility

## Advanced Sharding Patterns

### Composite Sharding
Combine multiple sharding strategies:
- **Geographic + Hash**: Region first, then hash within region
- **Time + Range**: Time buckets, then range within bucket
- **Entity + Feature**: User ID + feature type

### Multi-Level Sharding
Hierarchical sharding for extremely large datasets:
```
Level 1: Region (us-east, eu-west)
Level 2: Data center (dc-1, dc-2)
Level 3: Shard group (sg-1, sg-2)
Level 4: Individual shard (shard-1, shard-2)
```

### AI-Specific Sharding Strategies

#### Model-Centric Sharding
- **By model type**: Separate shards for different model architectures
- **By training phase**: Training data vs inference data separation
- **By data sensitivity**: PII data in separate, more secure shards

#### Feature Store Sharding
- **Namespace-based**: Different feature namespaces in different shards
- **Access pattern-based**: Frequently accessed features in hot shards
- **Temporal sharding**: Recent features in fast storage, historical in cold storage

## Real-World Production Examples

### Twitter's Snowflake ID + Sharding
- Uses 64-bit IDs with timestamp, worker ID, sequence
- Shards by worker ID component for horizontal scaling
- Handles 500K+ QPS for tweet storage

### Uber's Feature Store
- Hybrid sharding: geographic + hash-based
- Real-time features in memory shards, batch features in disk shards
- Achieves <10ms p99 latency for feature retrieval

### Google's Bigtable
- Row-key based sharding with automatic load balancing
- Uses tablet servers that split when size thresholds exceeded
- Scales to petabytes of data across thousands of servers

## Performance Metrics & Benchmarks

| Strategy | Scalability | Query Performance | Rebalancing Cost | Complexity |
|----------|-------------|-------------------|------------------|------------|
| Range    | Medium      | Excellent (range queries) | High             | Low        |
| Hash     | High        | Good (point queries) | Low              | Medium     |
| Directory| Very High   | Variable          | Very Low         | High       |
| Composite| Highest     | Optimized         | Medium           | Very High  |

*Tested on 100-node Cassandra cluster with 1B records*

## Debugging Sharding Issues

### Common Failure Modes
1. **Shard skew**: Uneven data distribution causing hotspots
2. **Cross-shard queries**: Expensive joins across shards
3. **Transaction coordination**: Distributed transactions across shards
4. **Rebalancing failures**: Data loss during shard migration

### Diagnostic Techniques
- **Shard utilization monitoring**: Track read/write ratios per shard
- **Query routing analysis**: Identify cross-shard query patterns
- **Latency profiling**: Measure shard-specific latencies
- **Data distribution visualization**: Heatmaps of data distribution

### Tools and Frameworks
- **Prometheus + Grafana**: Custom metrics for shard health
- **OpenTelemetry**: Distributed tracing across shards
- **Chaos engineering**: Simulate shard failures and network partitions

## AI/ML Specific Optimization Techniques

### Training Data Sharding
```python
class MLDataSharder:
    def __init__(self, strategy="consistent_hash"):
        self.strategy = strategy
        self.shard_map = {}
    
    def shard_key(self, dataset_id, sample_id):
        if self.strategy == "consistent_hash":
            return self._consistent_hash(f"{dataset_id}:{sample_id}")
        elif self.strategy == "time_range":
            return self._time_range_shard(sample_id)
    
    def _consistent_hash(self, key):
        # Use MurmurHash for better distribution
        return mmh3.hash(key) % self.num_shards
    
    def optimize_for_ml_workloads(self):
        # Ensure related samples (same user, same session) stay together
        # Implement locality-aware sharding for training batches
        pass
```

### Inference Optimization
- **Caching strategy**: Cache frequently accessed shards in memory
- **Prefetching**: Predict and prefetch related shards for batch inference
- **Shard affinity**: Route similar inference requests to same shards

## Best Practices for Senior Engineers

1. **Start simple**: Begin with range or hash sharding before complex strategies
2. **Monitor skew continuously**: Implement automated skew detection
3. **Design for rebalancing**: Build systems that can migrate data with minimal downtime
4. **Consider query patterns**: Optimize sharding for your most frequent queries
5. **Implement fallback mechanisms**: Handle shard unavailability gracefully

## Related Resources
- [Database Case Study: Netflix Recommendation Sharding](../06_case_studies/netflix_sharding_strategy.md)
- [System Design: Large-Scale Feature Store Architecture](../03_system_design/feature_store_architecture.md)
- [Debugging Patterns: Sharding Skew Analysis](../05_interview_prep/database_debugging_patterns.md)

---
*Last updated: February 2026 | Target audience: Senior AI/ML Engineers*