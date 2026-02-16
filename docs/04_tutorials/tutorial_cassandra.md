# Cassandra Tutorial: Distributed NoSQL for High-Scale AI Workloads

> **Target Audience**: Senior AI/ML Engineers, Data Architects, and Systems Engineers  
> **Prerequisites**: Basic understanding of distributed systems, NoSQL concepts, and database fundamentals  
> **Estimated Reading Time**: 45 minutes

Cassandra is a highly scalable, distributed NoSQL database designed for handling large amounts of data across many commodity servers while providing high availability with no single point of failure. For AI/ML workloads requiring massive scale, low-latency access, and fault tolerance, Cassandra offers unique advantages over traditional relational databases.

## Introduction

Apache Cassandra is a column-oriented, wide-column store that excels in write-heavy workloads and provides linear scalability. Its architecture is inspired by Amazon's Dynamo paper and Google's Bigtable, making it ideal for time-series data, feature stores, and real-time analytics pipelines common in modern ML systems.

Key characteristics relevant to AI/ML:
- **Linear scalability**: Add nodes to increase capacity without downtime
- **Tunable consistency**: Balance between availability and consistency based on use case
- **High write throughput**: Optimized for append-only patterns common in feature logging
- **Multi-datacenter replication**: Essential for global ML deployments
- **Schema flexibility**: Supports evolving ML feature schemas

This tutorial covers Cassandra fundamentals with a focus on AI/ML integration patterns, performance optimization, and practical implementation strategies.

## Core Concepts

### Architecture Overview

Cassandra follows a peer-to-peer architecture with no master nodes. Key components:

- **Node**: A single instance of Cassandra
- **Cluster**: Collection of nodes working together
- **Data Center**: Group of nodes in the same physical location
- **Replication Factor (RF)**: Number of copies of data across nodes
- **Consistency Level (CL)**: Number of replicas that must acknowledge reads/writes

### Data Modeling Principles

Unlike relational databases, Cassandra data modeling starts with query patterns, not entities. Key principles:

1. **Denormalization is mandatory**: Store redundant data to optimize read performance
2. **Partition keys define data distribution**: Choose carefully to avoid hotspots
3. **Clustering columns order data within partitions**: Enable efficient range queries
4. **Avoid secondary indexes for high-cardinality data**: Use materialized views or application-level indexing instead

### Consistency Levels

Cassandra offers tunable consistency levels:

| Level | Description | Use Case |
|-------|-------------|----------|
| `ONE` | Wait for 1 replica | High availability, low latency |
| `QUORUM` | Wait for majority (RF/2 + 1) | Balanced consistency/availability |
| `ALL` | Wait for all replicas | Strong consistency (rarely used in production) |
| `LOCAL_QUORUM` | Quorum within local DC | Multi-DC deployments |

For AI/ML workloads, `QUORUM` is typically optimal for feature stores, while `ONE` may be acceptable for telemetry/logging.

### Replication Strategies

- **SimpleStrategy**: For single DC deployments
- **NetworkTopologyStrategy**: For multi-DC deployments (recommended for production)

## Hands-On Examples

### Installation and Setup

#### Docker-based Development Environment

```bash
# Start a 3-node Cassandra cluster
docker run --name cassandra1 -d \
  -p 9042:9042 \
  -e CASSANDRA_CLUSTER_NAME=test-cluster \
  -e CASSANDRA_NUM_TOKENS=256 \
  -e CASSANDRA_SEEDS=cassandra1 \
  -e CASSANDRA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch \
  cassandra:4.1

# Connect via cqlsh
docker exec -it cassandra1 cqlsh
```

#### Local Installation (Linux/macOS)

```bash
# Download and install
wget https://downloads.apache.org/cassandra/4.1/apache-cassandra-4.1-bin.tar.gz
tar -xzf apache-cassandra-4.1-bin.tar.gz
cd apache-cassandra-4.1
bin/cassandra -f  # Run in foreground for development
```

### Creating a Feature Store Schema

For ML feature storage, we need efficient time-based access patterns:

```cql
-- Create keyspace for feature store
CREATE KEYSPACE ml_features 
WITH replication = {'class': 'NetworkTopologyStrategy', 'dc1': 3}
AND durable_writes = true;

USE ml_features;

-- Feature store table for model features
CREATE TABLE feature_store (
    model_id text,
    feature_name text,
    timestamp timestamp,
    value double,
    metadata map<text, text>,
    PRIMARY KEY ((model_id, feature_name), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC)
   AND compaction = {'class': 'TimeWindowCompactionStrategy',
                     'compaction_window_size': '1',
                     'compaction_window_unit': 'DAYS'}
   AND gc_grace_seconds = 864000;  -- 10 days
```

### Inserting and Querying Features

```cql
-- Insert feature values
INSERT INTO feature_store (model_id, feature_name, timestamp, value, metadata)
VALUES ('model_v2', 'user_engagement_score', '2026-02-16 10:30:00+0000', 0.87,
        {'user_id': 'u123', 'session_id': 's456', 'version': 'v2.1'});

-- Query latest feature value for a model
SELECT * FROM feature_store 
WHERE model_id = 'model_v2' 
  AND feature_name = 'user_engagement_score'
ORDER BY timestamp DESC 
LIMIT 1;

-- Query features over time window
SELECT timestamp, value, metadata 
FROM feature_store 
WHERE model_id = 'model_v2' 
  AND feature_name = 'user_engagement_score'
  AND timestamp > '2026-02-15 00:00:00+0000'
  AND timestamp < '2026-02-16 23:59:59+0000'
ORDER BY timestamp DESC;
```

### Time-Series Data Pattern

For monitoring ML model performance metrics:

```cql
CREATE TABLE model_metrics (
    model_id text,
    metric_name text,
    timestamp timestamp,
    value double,
    tags map<text, text>,
    PRIMARY KEY ((model_id, metric_name), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC)
   AND compaction = {'class': 'TimeWindowCompactionStrategy',
                     'compaction_window_size': '1',
                     'compaction_window_unit': 'HOURS'}
   AND default_time_to_live = 2592000;  -- 30 days TTL
```

## AI/ML Integration Patterns

### Feature Store Implementation

Cassandra serves as an excellent feature store backend due to its high write throughput and time-series capabilities:

```python
# Python client example using cassandra-driver
from cassandra.cluster import Cluster
from datetime import datetime, timedelta

class FeatureStore:
    def __init__(self, hosts=['127.0.0.1']):
        self.cluster = Cluster(hosts)
        self.session = self.cluster.connect('ml_features')
    
    def write_feature(self, model_id, feature_name, value, metadata=None):
        """Write a feature value with timestamp"""
        timestamp = datetime.utcnow()
        query = """
        INSERT INTO feature_store (model_id, feature_name, timestamp, value, metadata)
        VALUES (%s, %s, %s, %s, %s)
        """
        self.session.execute(query, 
                            (model_id, feature_name, timestamp, value, metadata or {}))
    
    def get_latest_feature(self, model_id, feature_name):
        """Get latest feature value"""
        query = """
        SELECT value, metadata FROM feature_store 
        WHERE model_id = %s AND feature_name = %s
        ORDER BY timestamp DESC LIMIT 1
        """
        result = self.session.execute(query, (model_id, feature_name))
        return result.one() if result else None
    
    def get_time_series(self, model_id, feature_name, start_time, end_time):
        """Get feature values over time window"""
        query = """
        SELECT timestamp, value, metadata FROM feature_store 
        WHERE model_id = %s AND feature_name = %s
        AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp DESC
        """
        return self.session.execute(query, (model_id, feature_name, start_time, end_time))
```

### Real-Time Analytics Pipeline

For real-time ML inference scoring:

```cql
-- Table for real-time inference results
CREATE TABLE inference_results (
    model_id text,
    request_id uuid,
    timestamp timestamp,
    input_hash text,
    prediction double,
    confidence double,
    features list<frozen<map<text, double>>>,
    PRIMARY KEY ((model_id), timestamp, request_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, request_id ASC)
   AND compaction = {'class': 'TimeWindowCompactionStrategy',
                     'compaction_window_size': '1',
                     'compaction_window_unit': 'MINUTES'};
```

### Model Versioning and A/B Testing

```cql
-- Track model versions and A/B test results
CREATE TABLE model_versions (
    experiment_id text,
    model_id text,
    version text,
    created_at timestamp,
    metrics map<text, double>,
    ab_test_group text,
    PRIMARY KEY ((experiment_id), created_at, model_id)
) WITH CLUSTERING ORDER BY (created_at DESC);
```

## Performance Optimization

### Tuning Parameters

#### Write Path Optimization

```yaml
# cassandra.yaml tuning for ML workloads
concurrent_writes: 32
concurrent_counter_writes: 32
memtable_flush_writers: 8
commitlog_sync: batch
commitlog_sync_batch_window_in_ms: 2
commitlog_segment_size_in_mb: 32
```

#### Read Path Optimization

```yaml
# For read-heavy feature serving
concurrent_reads: 64
file_cache_size_in_mb: 1024
key_cache_size_in_mb: 100
row_cache_size_in_mb: 0  # Disable row cache for time-series workloads
```

### Compaction Strategy Selection

| Workload Type | Recommended Strategy | Rationale |
|---------------|----------------------|-----------|
| Time-series data | TimeWindowCompactionStrategy | Efficient for time-bounded data, automatic TTL management |
| Static feature data | SizeTieredCompactionStrategy | Better for infrequently updated data |
| High-write throughput | LeveledCompactionStrategy | Lower read amplification, higher write overhead |

### Indexing Strategies

For high-cardinality filtering (e.g., user_id in feature store):

```cql
-- Materialized view for efficient user-based queries
CREATE MATERIALIZED VIEW feature_by_user AS
SELECT user_id, model_id, feature_name, timestamp, value
FROM feature_store
WHERE user_id IS NOT NULL AND model_id IS NOT NULL 
  AND feature_name IS NOT NULL AND timestamp IS NOT NULL
PRIMARY KEY ((user_id), model_id, feature_name, timestamp)
WITH CLUSTERING ORDER BY (model_id ASC, feature_name ASC, timestamp DESC);
```

### Batch Operations

For bulk feature ingestion:

```python
def bulk_insert_features(session, features):
    """Insert multiple features efficiently"""
    insert_query = session.prepare("""
        INSERT INTO feature_store (model_id, feature_name, timestamp, value, metadata)
        VALUES (?, ?, ?, ?, ?)
    """)
    
    # Use async execution for better throughput
    futures = []
    for feature in features:
        future = session.execute_async(
            insert_query, 
            (feature['model_id'], feature['feature_name'], 
             feature['timestamp'], feature['value'], feature['metadata'])
        )
        futures.append(future)
    
    # Wait for completion
    for future in futures:
        future.result()
```

## Common Pitfalls and Solutions

### 1. Hot Partition Keys

**Problem**: Uneven data distribution causing node overload.

**Solution**: 
- Use composite partition keys with high cardinality
- Apply salting techniques for low-cardinality dimensions
- Monitor partition sizes with `nodetool tablestats`

```cql
-- Bad: Single low-cardinality partition key
PRIMARY KEY (model_id, timestamp)

-- Good: Composite partition key with salt
PRIMARY KEY ((model_id, bucket), timestamp)
-- Where bucket = hash(user_id) % 100
```

### 2. Tombstone Overhead

**Problem**: Excessive deleted data causing read performance degradation.

**Solution**:
- Set appropriate TTL for time-series data
- Use `gc_grace_seconds` appropriately (default 10 days)
- Regularly run `nodetool repair` and `nodetool cleanup`

### 3. Large Partitions

**Problem**: Partitions exceeding 100MB causing GC pressure and timeouts.

**Solution**:
- Limit partition size to < 100MB
- Use time-bucketing for time-series data
- Monitor with `nodetool cfstats`

### 4. Consistency Level Mismatch

**Problem**: Inconsistent reads due to inappropriate consistency levels.

**Solution**:
- Use `QUORUM` for feature stores where consistency matters
- Use `ONE` for telemetry where availability is prioritized
- Test consistency behavior in your specific deployment

## Advanced Topics for AI/ML Engineers

### Integration with ML Platforms

#### TensorFlow Extended (TFX) Integration

```python
# Custom TFX component for Cassandra feature retrieval
class CassandraFeatureRetrievalComponent:
    def __init__(self, host, keyspace, table):
        self.host = host
        self.keyspace = keyspace
        self.table = table
    
    def resolve_features(self, examples):
        """Retrieve features for given examples"""
        # Implement feature lookup logic
        pass
```

#### PyTorch Lightning Integration

```python
class CassandraDataset(torch.utils.data.Dataset):
    def __init__(self, cassandra_session, query_template, params):
        self.session = cassandra_session
        self.query_template = query_template
        self.params = params
        self.rows = self._fetch_data()
    
    def _fetch_data(self):
        query = self.query_template % self.params
        return list(self.session.execute(query))
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        # Convert to tensor format
        return torch.tensor(row.features), torch.tensor(row.label)
```

### Monitoring and Observability

Critical metrics for Cassandra in ML deployments:

- **Read/write latency percentiles** (p95, p99)
- **Pending compactions**
- **Memtable size and flush rates**
- **Hinted handoff rates**
- **GC pause times**

Use tools like Prometheus + Grafana with Cassandra exporter for comprehensive monitoring.

## Conclusion

Cassandra provides a robust foundation for high-scale AI/ML workloads requiring distributed storage, high availability, and linear scalability. By following the data modeling principles outlined in this tutorial and applying the performance optimization techniques, you can build feature stores, time-series databases, and real-time analytics systems that scale to handle the demands of modern ML infrastructure.

The key to success with Cassandra is embracing its distributed nature and designing data models around query patterns rather than traditional normalization principles. For AI/ML engineers, this means thinking in terms of feature access patterns, time windows, and consistency requirements rather than entity relationships.

## Further Reading

- [Apache Cassandra Documentation](https://cassandra.apache.org/doc/latest/)
- "Designing Data-Intensive Applications" by Martin Kleppmann (Chapter 3)
- Cassandra Summit presentations on ML use cases
- Netflix's Cassandra usage patterns for recommendation systems