# NoSQL Database Internals: How Cassandra/MongoDB Handle Distributed Operations

## Executive Summary

NoSQL databases power many AI/ML systems due to their scalability and flexibility, but understanding their internal architecture is crucial for building reliable ML infrastructure. This system design explores the fundamental internals of Apache Cassandra and MongoDB, revealing how these distributed databases achieve high availability, horizontal scalability, and eventual consistency—key requirements for modern ML workloads.

## Technical Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                          Apache Cassandra Architecture        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Client Applications                                     │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │ │
│  │  │ Driver      │   │ Driver      │   │ Driver      │    │ │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │ │
│  │         │                 │                 │           │ │
│  │  ┌──────▼─────────────────▼─────────────────▼────────┐ │ │
│  │  │                Cassandra Cluster                 │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │ │
│  │  │  │ Node 1      │ │ Node 2      │ │ Node 3      │ │ │ │
│  │  │  │ (Coordinator)│ │ (Replica)   │ │ (Replica)   │ │ │ │
│  │  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ │ │ │
│  │  │         │               │               │        │ │ │
│  │  │  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐ │ │ │
│  │  │  │ Memtable    │ │ Commit Log  │ │ SSTables    │ │ │ │
│  │  │  │ (in-memory) │ │ (WAL)       │ │ (on-disk)  │ │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ │ │ │
│  │  └───────────────────────────────────────────────────┘ │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │                Data Distribution                │ │ │
│  │  │  Partition Key → Token → Virtual Node Ring     │ │ │
│  │  │  Replication Factor = 3 → 3 replicas per partition│ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Implementation Details

### Core Architecture Components

#### 1. Distributed Data Model

**Cassandra Partitioning**:
- **Partition Key**: Determines which node stores the data
- **Token Ring**: Consistent hashing maps partitions to nodes
- **Virtual Nodes**: Multiple tokens per physical node for better distribution

**MongoDB Sharding**:
- **Shard Key**: Determines data distribution across shards
- **Config Servers**: Store cluster metadata
- **Mongos Routers**: Query routing and aggregation

#### 2. Storage Engine Internals

**Cassandra LSM-tree Architecture**:
- **Memtable**: In-memory sorted structure (write buffer)
- **Commit Log**: Write-ahead log for durability
- **SSTable**: Sorted String Table (immutable on-disk files)
- **Compaction**: Merge SSTables to remove tombstones and duplicates

**MongoDB WiredTiger Engine**:
- **B+ Tree**: Primary storage structure
- **Document-level locking**: Fine-grained concurrency control
- **Checkpointing**: Periodic snapshots for crash recovery
- **Compression**: Snappy/Zstd compression for storage efficiency

#### 3. Consistency and Replication

**Cassandra Consistency Levels**:
- **ONE**: Acknowledge by one replica
- **QUORUM**: Majority of replicas (RF=3 → 2 replicas)
- **ALL**: All replicas acknowledge
- **LOCAL_QUORUM**: Quorum within local data center

**MongoDB Read Preferences**:
- **primary**: Read from primary only
- **secondary**: Read from secondaries
- **nearest**: Read from closest node
- **majority**: Read from majority of replica set

#### 4. Query Processing

**Cassandra CQL Execution**:
```
Client Query → Coordinator Node → Partition Key Hashing → 
→ Replica Selection → Consistency Check → 
→ Read Repair / Hinted Handoff → Result
```

**MongoDB Query Optimization**:
- **Index Selection**: B-tree indexes, compound indexes
- **Query Planner**: Cost-based optimization
- **Aggregation Pipeline**: Multi-stage processing for complex queries

### Performance Optimization Mechanisms

#### Compaction Strategies (Cassandra)
- **Size-Tiered Compaction**: Merge similar-sized SSTables
- **Leveled Compaction**: Overlapping SSTables in levels (better read performance)
- **Time-Window Compaction**: Optimized for time-series data

#### Caching Layers
- **Cassandra**: Key cache, row cache, counter cache
- **MongoDB**: WiredTiger cache (in-memory working set)

#### Indexing Strategies
- **Cassandra**: Secondary indexes (local), materialized views
- **MongoDB**: Single field, compound, multikey, text, geospatial indexes

## Performance Metrics and Trade-offs

| Component | Cassandra | MongoDB | ML Impact |
|-----------|-----------|---------|-----------|
| Write Throughput | Very High (100K-1M+ ops/sec) | High (50K-200K ops/sec) | Cassandra better for high-volume ML ingestion |
| Read Latency | Medium-High (5-50ms) | Low-Medium (1-20ms) | MongoDB better for real-time inference |
| Scalability | Linear horizontal scaling | Good horizontal scaling | Both excellent for ML scale-out |
| Consistency Model | Tunable (eventual to strong) | Strong (replica sets) | Cassandra more flexible for ML trade-offs |
| Storage Efficiency | High (compression + compaction) | Medium-High | Cassandra better for large ML datasets |

**Latency Comparison (1M records)**:
- Cassandra write: 2-5ms (with QUORUM)
- MongoDB write: 5-15ms (with w=majority)
- Cassandra read: 10-30ms (with ONE)
- MongoDB read: 2-8ms (with primary)

**Throughput Comparison**:
- Cassandra bulk insert: 200K-500K records/sec
- MongoDB bulk insert: 50K-150K records/sec
- Cassandra concurrent reads: 50K-200K ops/sec
- MongoDB concurrent reads: 20K-100K ops/sec

## Key Lessons for AI/ML Systems

1. **Data Modeling Drives Performance**: In NoSQL, query patterns determine schema design (denormalization is often necessary).

2. **Consistency Level Selection**: Choose based on ML workload requirements (strong for model registry, eventual for telemetry).

3. **Compaction Strategy Matters**: Time-window compaction is ideal for time-series ML data.

4. **Indexing Limitations**: Secondary indexes in Cassandra have performance penalties; prefer denormalized models.

5. **ML-Specific Patterns**:
   - Time-series data → Cassandra with time-window compaction
   - Document-rich features → MongoDB with rich indexing
   - High-throughput ingestion → Cassandra LSM-tree

## Real-World Industry Examples

**Netflix**: Cassandra for user activity tracking and recommendation data (100TB+)

**Uber**: Cassandra for real-time ride matching and location data

**Facebook**: Cassandra for inbox search and messaging infrastructure

**Google**: MongoDB for various internal ML services and analytics

**Tesla**: Cassandra for vehicle telemetry and sensor data storage

**Airbnb**: MongoDB for listing data and personalized recommendations

## Measurable Outcomes

- **Cassandra Ingestion**: 5-10x higher throughput than relational databases for ML telemetry data
- **MongoDB Query Performance**: 3-5x faster for document-based feature queries
- **Storage Efficiency**: Cassandra reduces storage costs by 30-50% for append-only ML workloads
- **Scalability**: Both systems scale linearly to 100+ nodes for ML infrastructure

**ML Impact Metrics**:
- Feature store latency: Reduced from 100ms to 20ms with optimized NoSQL design
- Data pipeline throughput: Increased from 10K to 200K records/sec with Cassandra
- Model training data availability: 99.99% with proper replication strategies

## Practical Guidance for AI/ML Engineers

1. **Start with Data Access Patterns**: Design schema around how ML systems will query the data.

2. **Use Appropriate Consistency Levels**: 
   - Model metadata → QUORUM or ALL
   - Telemetry data → ONE for maximum throughput

3. **Optimize Compaction**: For time-series ML data, use time-window compaction.

4. **Implement Proper Indexing**: In MongoDB, create compound indexes for common ML query patterns.

5. **Monitor Key Metrics**: Track read/write latency, compaction backlog, and cache hit ratios.

6. **Consider Hybrid Approaches**: Use Cassandra for ingestion, MongoDB for serving layers.

7. **Leverage Built-in Features**: 
   - Cassandra: Materialized views, lightweight transactions
   - MongoDB: Aggregation pipelines, change streams for ML monitoring

Understanding NoSQL database internals enables AI/ML engineers to design distributed data systems that scale horizontally while maintaining the performance and reliability required for modern ML workloads.