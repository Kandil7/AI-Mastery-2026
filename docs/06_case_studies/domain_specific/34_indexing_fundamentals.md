# Indexing Fundamentals: B-tree, Hash, and LSM-tree for AI/ML Systems

## Executive Summary

Indexing is the cornerstone of database performance optimization, directly impacting the efficiency of ML data pipelines, feature stores, and model serving systems. For AI/ML engineers, understanding different indexing strategies—B-tree, hash, and LSM-tree—is essential for designing high-performance data infrastructure. This case study explores the mechanics, performance characteristics, and practical applications of these fundamental indexing techniques in ML contexts.

## Technical Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                          Database Indexing Layer              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 Storage Engine Architecture              │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │ │
│  │  │ B-tree Index│   │ Hash Index  │   │ LSM-tree    │    │ │
│  │  │ (Balanced)  │   │ (Direct)    │   │ (Log-Structured)│  │ │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬────────┘   │ │
│  │         │                 │                 │            │ │
│  │  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼────────┐  │ │
│  │  │ Page-based  │   │ Bucket-based│   │ Memtable + SST│  │ │
│  │  │ Storage     │   │ Hash Table  │   │ Files + Compaction││ │
│  │  └─────────────┘   └─────────────┘   └───────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   Query Processing Flow                  │ │
│  │  Client Query → Parser → Optimizer → Executor → Index   │ │
│  │                         ↑          ↓                    │ │
│  │                Index Selection Strategy                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Details

### B-tree Indexes: Balanced Search Trees

**Concept**: Multi-level balanced tree structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time.

**Structure**:
```
Root: [10, 20, 30]
       /    |    |    \
Level1: [5,8] [15,18] [25,28] [35,40]
       /|\    /|\    /|\    /|\
Leaves: 1,3,7 9,12,14 16,19,22 24,26,29 31,33,37 39,41,45
```

**Real-World Example - PostgreSQL B-tree**:
```sql
-- Create B-tree index (default)
CREATE INDEX idx_user_id ON predictions (user_id);

-- Composite B-tree for ML feature queries
CREATE INDEX idx_model_timestamp ON predictions (model_id, timestamp DESC);
```

**ML-Specific Use Cases**:
- **Time-series queries**: Range queries on timestamps for feature engineering
- **Model version filtering**: Fast lookup by model_id and version
- **Sorted feature retrieval**: Ordered access to features for training batches

**Performance Characteristics**:
- **Search**: O(log n)
- **Insert/Delete**: O(log n)
- **Range Queries**: Excellent (sequential access)
- **Memory Usage**: Moderate (internal nodes + leaf pages)

### Hash Indexes: Direct Addressing

**Concept**: Uses hash function to map keys directly to storage locations, providing O(1) average-case lookup.

**Structure**:
```
Hash Function: h(key) = key % bucket_count
Bucket 0: [user_1001 → offset_12345]
Bucket 1: [user_1002 → offset_67890]
Bucket 2: [user_1003 → offset_24680]
...
Bucket N: []
```

**Real-World Example - Redis Hash Index**:
```python
# Redis hash for fast feature lookup
redis.hset(f"features:{entity_id}", "age", 35)
redis.hset(f"features:{entity_id}", "location", "NY")
redis.hget(f"features:{entity_id}", "age")  # O(1) lookup
```

**ML-Specific Use Cases**:
- **Real-time feature serving**: Single-entity lookups for inference
- **Caching layers**: Feature cache with TTL-based expiration
- **Entity ID mapping**: Fast conversion between external IDs and internal keys

**Performance Characteristics**:
- **Search**: O(1) average, O(n) worst-case (collisions)
- **Insert/Delete**: O(1) average
- **Range Queries**: Poor (no ordering)
- **Memory Usage**: High (empty buckets, collision handling)

### LSM-tree (Log-Structured Merge-tree): Write-Optimized

**Concept**: Optimized for write-heavy workloads by batching writes and merging sorted files.

**Structure**:
```
Memtable (in-memory, sorted)
↓
Immutable Memtable → SSTable (on disk, sorted)
↓
Compaction: Merge SSTables → New SSTable
↓
Level 0: Recent writes (small SSTables)
Level 1+: Older data (larger, merged SSTables)
```

**Real-World Example - Apache Cassandra LSM-tree**:
```cql
-- Cassandra uses LSM-tree internally
CREATE TABLE predictions (
    model_id TEXT,
    timestamp TIMESTAMP,
    entity_id TEXT,
    prediction DOUBLE,
    PRIMARY KEY ((model_id), timestamp, entity_id)
);
```

**ML-Specific Use Cases**:
- **High-volume data ingestion**: Sensor data, telemetry, logs
- **Time-series databases**: Efficient time-based queries
- **Feature logging**: Append-only feature stores

**Performance Characteristics**:
- **Write**: O(1) amortized (batched writes)
- **Read**: O(log n + k) where k = number of SSTables to check
- **Range Queries**: Good (sorted SSTables)
- **Memory Usage**: Variable (memtable size + SSTable overhead)

## Performance Metrics and Trade-offs

| Index Type | Write Performance | Read Performance | Range Queries | Memory Efficiency | Best ML Use Cases |
|------------|-------------------|------------------|---------------|-------------------|-------------------|
| B-tree | Medium (O(log n)) | High (O(log n)) | Excellent | Medium | Training data, model registry |
| Hash | Excellent (O(1)) | Excellent (O(1)) | Poor | Low | Real-time inference, caching |
| LSM-tree | Excellent (O(1) amortized) | Medium (O(log n + k)) | Good | High (for writes) | Data ingestion, time-series |

**Latency Comparison (1M records)**:
- B-tree: 0.1ms search, 0.2ms insert
- Hash: 0.05ms search, 0.05ms insert
- LSM-tree: 0.02ms write, 0.3ms read (worst-case)

**Throughput Comparison**:
- B-tree: 50K-100K ops/sec
- Hash: 100K-500K ops/sec
- LSM-tree: 200K-1M+ ops/sec (write-heavy)

## Key Lessons for AI/ML Systems

1. **Workload Pattern Dictates Index Choice**:
   - Read-heavy ML workloads → B-tree
   - Real-time inference → Hash
   - Data ingestion pipelines → LSM-tree

2. **Composite Indexes Are Powerful**: Combine multiple columns for ML-specific query patterns.

3. **Index Maintenance Matters**: Consider the cost of index updates during bulk operations.

4. **Memory vs Disk Trade-offs**: Hash indexes consume more memory; LSM-trees optimize for disk I/O.

5. **ML-Specific Indexing Strategies**:
   - Time-partitioned indexes for time-series ML data
   - Multi-column indexes for feature combinations
   - Partial indexes for filtered ML datasets

## Real-World Industry Examples

**Google Bigtable**: LSM-tree based, used for ML training data storage at scale

**Redis**: Hash-based indexing, used for real-time feature serving in recommendation systems

**PostgreSQL**: B-tree default, extended with GiST, GIN for specialized ML use cases

**Apache Cassandra**: LSM-tree with SSTable compaction, used for IoT and telemetry data in ML pipelines

**TimescaleDB**: Hybrid approach - B-tree for metadata, LSM-tree for time-series data

## Measurable Outcomes

- **LSM-tree for Ingestion**: 5-10x higher write throughput compared to B-tree for ML data pipelines
- **Hash for Inference**: 3-5x lower P99 latency for real-time feature lookups
- **B-tree for Analytics**: 2-3x faster range queries for training data exploration
- **Storage Efficiency**: LSM-tree reduces storage costs by 20-40% for append-only ML workloads

**ML Impact Metrics**:
- Feature serving latency: Reduced from 50ms to 10ms with hash indexes
- Data pipeline throughput: Increased from 10K to 100K records/sec with LSM-tree
- Training data preparation: 40% faster with optimized B-tree indexes

## Practical Guidance for AI/ML Engineers

1. **Profile Your Query Patterns**: Use database query analyzers to identify hot paths.

2. **Start with Default Indexes**: Most databases have good defaults (B-tree), optimize later.

3. **Use Covering Indexes**: Include all needed columns in indexes to avoid table lookups.

4. **Consider Time-Series Specific Indexes**: For temporal ML data, use time-partitioned indexes.

5. **Monitor Index Statistics**: Track index hit ratios, fragmentation, and maintenance costs.

6. **Hybrid Indexing Strategies**: Combine index types (e.g., B-tree for metadata, LSM-tree for data).

7. **Test with Real ML Workloads**: Benchmark with actual feature engineering and inference queries.

Understanding indexing fundamentals enables AI/ML engineers to design data systems that deliver optimal performance for their specific ML workloads, balancing the trade-offs between read speed, write throughput, and storage efficiency.