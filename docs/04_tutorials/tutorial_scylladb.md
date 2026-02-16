# ScyllaDB Tutorial: High-Performance Cassandra-Compatible Database

> **Target Audience**: Senior AI/ML Engineers, Systems Architects, and Performance Engineers  
> **Prerequisites**: Cassandra knowledge, understanding of distributed systems, CQL proficiency  
> **Estimated Reading Time**: 45 minutes

ScyllaDB is a drop-in replacement for Apache Cassandra, written in C++ with a shared-nothing architecture that delivers significantly higher performance and lower latency. Built on the Seastar framework, ScyllaDB achieves near-linear scalability across CPU cores and provides predictable low-latency performance essential for real-time ML inference and feature serving.

## Introduction

ScyllaDB was created to address Cassandra's Java-based limitations, particularly around GC pauses, CPU efficiency, and latency predictability. For AI/ML workloads requiring ultra-low latency, high throughput, and consistent performance, ScyllaDB offers compelling advantages:

- **10x+ higher throughput** than Cassandra for the same hardware
- **Sub-millisecond p99 latencies** (vs 10-100ms for Cassandra)
- **Predictable performance** without GC pauses
- **Cassandra compatibility** (CQL, drivers, tools)
- **Shard-per-core architecture** for optimal CPU utilization

ScyllaDB is particularly well-suited for:
- Real-time feature serving for ML models
- Low-latency inference scoring
- High-frequency time-series data
- Multi-tenant ML platforms requiring isolation

This tutorial covers ScyllaDB fundamentals with a focus on AI/ML integration patterns, performance optimization, and practical implementation strategies.

## Core Concepts

### Architecture Overview

ScyllaDB's key architectural innovations:

- **Shard-per-core model**: Each CPU core runs an independent "shard" with its own memory, network stack, and I/O
- **Seastar framework**: Asynchronous, non-blocking C++ framework eliminating traditional OS overhead
- **Shared-nothing design**: No coordination between shards, enabling linear scalability
- **Zero-copy networking**: Direct memory access for network operations
- **Lock-free data structures**: Eliminate contention bottlenecks

### Data Modeling Principles

ScyllaDB follows Cassandra's data modeling principles but with enhanced performance characteristics:

1. **Partition keys remain critical**: Poor partitioning still causes hotspots
2. **Denormalization is essential**: Optimize for read patterns
3. **Clustering columns enable efficient range queries**
4. **Secondary indexes are still discouraged** for high-cardinality data

### Consistency and Replication

ScyllaDB maintains Cassandra's consistency model but with improved performance:

| Consistency Level | Performance Impact | Use Case |
|-------------------|-------------------|----------|
| `ONE` | Lowest latency, highest throughput | Real-time inference, telemetry |
| `QUORUM` | Balanced performance | Feature stores, model metadata |
| `LOCAL_QUORUM` | Multi-DC optimized | Global ML deployments |
| `ALL` | Highest consistency | Critical metadata operations |

### Key Differences from Cassandra

| Feature | Cassandra | ScyllaDB |
|---------|-----------|----------|
| Language | Java | C++ |
| Architecture | JVM-based | Shard-per-core |
| Latency (p99) | 10-100ms | 0.5-5ms |
| Throughput | ~100K ops/sec/node | ~1M+ ops/sec/node |
| Memory overhead | High (JVM) | Low (direct memory) |
| GC pauses | Yes | None |
| CPU utilization | Variable | Near 100% |
| Startup time | Seconds | Milliseconds |

## Hands-On Examples

### Installation and Setup

#### Docker-based Development Environment

```bash
# Start ScyllaDB cluster (3 nodes)
docker run --name scylla1 -d \
  -p 9042:9042 \
  -p 7199:7199 \
  -e SCYLLA_CLUSTER_NAME=test-cluster \
  -e SCYLLA_SEEDS=scylla1 \
  -e SCYLLA_ENDPOINT_SNITCH=GossipingPropertyFileSnitch \
  scylladb/scylla:5.4

# Connect via cqlsh
docker exec -it scylla1 cqlsh
```

#### Local Installation (Linux)

```bash
# Ubuntu/Debian
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 278B1B81
echo "deb https://downloads.scylladb.com/deb/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/scylladb.list
sudo apt-get update
sudo apt-get install scylla-server scylla-jmx
sudo systemctl start scylla-server
```

### Creating High-Performance Feature Store

#### Low-Latency Feature Serving Table

```cql
-- Create keyspace optimized for low-latency access
CREATE KEYSPACE ml_features 
WITH replication = {'class': 'NetworkTopologyStrategy', 'dc1': 3}
AND durable_writes = true;

USE ml_features;

-- Feature store for real-time inference
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
                     'compaction_window_unit': 'MINUTES'}
   AND gc_grace_seconds = 864000
   AND speculative_retry = 'PERCENTILE:99.0'
   AND read_repair_chance = 0.0;
```

#### Real-Time Inference Scoring Table

```cql
-- Ultra-low latency inference results
CREATE TABLE inference_scores (
    model_id text,
    request_id uuid,
    timestamp timestamp,
    prediction double,
    confidence double,
    features list<frozen<map<text, double>>>,
    PRIMARY KEY ((model_id), timestamp, request_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, request_id ASC)
   AND compaction = {'class': 'TimeWindowCompactionStrategy',
                     'compaction_window_size': '1',
                     'compaction_window_unit': 'SECONDS'}
   AND default_time_to_live = 300;  -- 5 minute TTL for real-time data
```

### Inserting and Querying Data

#### High-Throughput Insert Example

```cql
-- Batch insert for feature updates
BEGIN BATCH
INSERT INTO feature_store (model_id, feature_name, timestamp, value, metadata)
VALUES ('model_v3', 'user_engagement', '2026-02-16 10:35:00+0000', 0.92,
        {'user_id': 'u456', 'session_id': 's789', 'version': 'v3.0'});
INSERT INTO feature_store (model_id, feature_name, timestamp, value, metadata)
VALUES ('model_v3', 'click_through_rate', '2026-02-16 10:35:00+0000', 0.18,
        {'user_id': 'u456', 'session_id': 's789', 'version': 'v3.0'});
APPLY BATCH;
```

#### Low-Latency Query Examples

```cql
-- Get latest feature value (optimized for p99 latency)
SELECT value, metadata FROM feature_store 
WHERE model_id = 'model_v3' 
  AND feature_name = 'user_engagement'
ORDER BY timestamp DESC 
LIMIT 1
USING TIMESTAMP 1708079700000000;  -- Explicit timestamp for consistency

-- Real-time inference lookup
SELECT prediction, confidence, features 
FROM inference_scores 
WHERE model_id = 'model_v3' 
  AND timestamp > '2026-02-16 10:34:00+0000'
  AND timestamp < '2026-02-16 10:35:00+0000'
ORDER BY timestamp DESC 
LIMIT 1;
```

## AI/ML Integration Patterns

### Real-Time Feature Serving

For ML model inference requiring sub-millisecond feature lookup:

```python
# Python client with connection pooling for low latency
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
import time

class LowLatencyFeatureStore:
    def __init__(self, hosts=['127.0.0.1'], port=9042):
        self.cluster = Cluster(
            hosts=hosts,
            port=port,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='dc1'),
            protocol_version=4,
            # Optimized for low latency
            connect_timeout=2.0,
            execution_profiles={
                'low_latency': ExecutionProfile(
                    request_timeout=1.0,
                    retry_policy=RetryPolicy(),
                    consistency_level=ConsistencyLevel.ONE
                )
            }
        )
        self.session = self.cluster.connect('ml_features')
    
    def get_feature(self, model_id, feature_name, timeout=0.5):
        """Get feature with strict latency SLA"""
        start_time = time.time()
        try:
            query = """
            SELECT value, metadata FROM feature_store 
            WHERE model_id = %s AND feature_name = %s
            ORDER BY timestamp DESC LIMIT 1
            """
            result = self.session.execute(
                query, 
                (model_id, feature_name),
                execution_profile='low_latency'
            )
            latency = time.time() - start_time
            if latency > timeout:
                raise TimeoutError(f"Query exceeded {timeout}s SLA: {latency:.3f}s")
            return result.one()
        except Exception as e:
            # Fallback to cached values or defaults
            return self._get_fallback_value(model_id, feature_name)
```

### Multi-Tenant ML Platform

For SaaS ML platforms requiring tenant isolation:

```cql
-- Tenant-scoped feature store
CREATE TABLE tenant_features (
    tenant_id text,
    model_id text,
    feature_name text,
    timestamp timestamp,
    value double,
    metadata map<text, text>,
    PRIMARY KEY ((tenant_id, model_id, feature_name), timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC)
   AND compaction = {'class': 'TimeWindowCompactionStrategy',
                     'compaction_window_size': '1',
                     'compaction_window_unit': 'HOURS'}
   AND speculative_retry = 'PERCENTILE:99.9';
```

### Model Versioning and Canary Deployments

```cql
-- Track model versions with canary percentages
CREATE TABLE model_versions (
    tenant_id text,
    model_id text,
    version text,
    created_at timestamp,
    canary_percentage int,
    metrics map<text, double>,
    status text,
    PRIMARY KEY ((tenant_id, model_id), created_at, version)
) WITH CLUSTERING ORDER BY (created_at DESC, version DESC)
   AND compaction = {'class': 'SizeTieredCompactionStrategy'};
```

## Performance Optimization

### ScyllaDB-Specific Tuning

#### Server Configuration (`scylla.yaml`)

```yaml
# Critical settings for ML workloads
cpu_affinity: true
memory: 64GB
commitlog_directory: /mnt/ssd/commitlog
data_file_directories:
  - /mnt/ssd/data
  - /mnt/ssd/data2
endpoint_snitch: GossipingPropertyFileSnitch
concurrent_reads: 128
concurrent_writes: 128
memtable_flush_writers: 16
compaction_throughput_mb_per_sec: 100
streaming_socket_timeout_in_ms: 300000
```

#### Advanced CQL Settings

```cql
-- Per-table optimizations
ALTER TABLE feature_store 
WITH speculative_retry = 'PERCENTILE:99.9'
AND read_repair_chance = 0.0
AND dclocal_read_repair_chance = 0.0
AND crc_check_chance = 1.0;

-- Enable compression for feature data
ALTER TABLE feature_store 
WITH compression = { 
    'class': 'LZ4Compressor',
    'level': '1'  -- Fast compression for low-latency workloads
};
```

### Shard-Aware Design Patterns

#### Connection Pooling Strategy

```python
# Shard-aware connection management
class ShardAwareClient:
    def __init__(self, hosts, num_shards):
        self.hosts = hosts
        self.num_shards = num_shards
        self.sessions = {}
        
        # Create one session per shard
        for shard_id in range(num_shards):
            cluster = Cluster(
                hosts=hosts,
                # Pin to specific shard using scylla-specific options
                execution_profiles={
                    'shard_aware': ExecutionProfile(
                        request_timeout=0.5,
                        consistency_level=ConsistencyLevel.ONE
                    )
                }
            )
            self.sessions[shard_id] = cluster.connect('ml_features')
    
    def get_session_for_key(self, partition_key):
        """Route requests to appropriate shard based on partition key hash"""
        shard_id = hash(partition_key) % self.num_shards
        return self.sessions[shard_id]
```

### Hardware Optimization

#### Storage Configuration

- **NVMe SSDs**: Essential for low-latency I/O
- **RAID 0 for commit logs**: Separate fast storage for write path
- **Multiple data directories**: Distribute I/O across devices
- **Direct I/O**: Bypass page cache for predictable latency

#### Network Optimization

```bash
# Kernel tuning for low-latency networking
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' >> /etc/sysctl.conf
sysctl -p
```

## Common Pitfalls and Solutions

### 1. Over-Partitioning

**Problem**: Too many small partitions causing overhead.

**Solution**:
- Target 10-100MB per partition
- Use time-bucketing for time-series data
- Monitor with `nodetool tablestats`

```cql
-- Good: Time-bucketed partitions
PRIMARY KEY ((model_id, toYYYYMM(timestamp)), feature_name, timestamp)

-- Better: Composite with high-cardinality dimension
PRIMARY KEY ((model_id, user_bucket, toYYYYMM(timestamp)), feature_name, timestamp)
-- Where user_bucket = hash(user_id) % 100
```

### 2. High GC Pressure (Cassandra Legacy)

**Problem**: Even with ScyllaDB, some Java-based tools may cause GC issues.

**Solution**:
- Use ScyllaDB-native tools instead of Cassandra tools
- Avoid Java-based monitoring agents
- Use `scylla-manager` for operations instead of `nodetool`

### 3. Network Bottlenecks

**Problem**: Network saturation limiting throughput.

**Solution**:
- Use 10Gbps+ networking
- Enable TCP BBR congestion control
- Configure proper MTU settings
- Use multiple network interfaces for sharding

### 4. Inconsistent Read Performance

**Problem**: Variable latency despite ScyllaDB's predictability.

**Solution**:
- Ensure proper partition sizing
- Use `speculative_retry` appropriately
- Monitor shard utilization with `scylla-monitoring`
- Avoid large partitions (> 100MB)

## Advanced Topics for AI/ML Engineers

### Integration with Real-Time ML Systems

#### Kafka Integration for Streaming Features

```cql
-- ScyllaDB CDC (Change Data Capture) for real-time pipelines
CREATE TABLE feature_updates_cdc (
    model_id text,
    feature_name text,
    timestamp timestamp,
    old_value double,
    new_value double,
    PRIMARY KEY ((model_id, feature_name), timestamp)
) WITH cdc = {'enabled': 'true'};
```

#### gRPC Service Integration

```protobuf
// Feature service definition
service FeatureService {
  rpc GetFeature(GetFeatureRequest) returns (GetFeatureResponse);
  rpc BatchGetFeatures(BatchGetFeaturesRequest) returns (BatchGetFeaturesResponse);
}

message GetFeatureRequest {
  string model_id = 1;
  string feature_name = 2;
  string user_id = 3;
}

message GetFeatureResponse {
  double value = 1;
  map<string, string> metadata = 2;
  double latency_ms = 3;
}
```

### Performance Benchmarking

#### Comparative Benchmarks

| Operation | Cassandra (4.1) | ScyllaDB (5.4) | Improvement |
|-----------|----------------|----------------|-------------|
| 1M writes/sec | 120K | 1.2M | 10x |
| p99 read latency | 45ms | 1.8ms | 25x |
| CPU efficiency | 30% | 95% | 3x |
| Memory overhead | 2GB/node | 0.5GB/node | 4x |

#### Benchmarking Script

```python
import time
import cassandra
from cassandra.cluster import Cluster

def benchmark_scylla_performance():
    """Measure ScyllaDB performance for ML workloads"""
    
    # Test 1: Single feature lookup
    start = time.time()
    for _ in range(10000):
        result = session.execute("""
            SELECT value FROM feature_store 
            WHERE model_id = 'model_v1' 
              AND feature_name = 'engagement_score'
            ORDER BY timestamp DESC LIMIT 1
        """)
    single_lookup_time = (time.time() - start) / 10000
    
    # Test 2: Batch feature retrieval
    start = time.time()
    for _ in range(1000):
        results = session.execute("""
            SELECT feature_name, value FROM feature_store 
            WHERE model_id = 'model_v1' 
              AND timestamp > %s
            ORDER BY timestamp DESC LIMIT 10
        """, (time.time() - 300,))
    batch_time = (time.time() - start) / 1000
    
    return {
        'single_lookup_p99': single_lookup_time * 1000,  # ms
        'batch_retrieval_p99': batch_time * 1000,  # ms
        'throughput_ops_sec': 10000 / (single_lookup_time * 10000)
    }
```

### Monitoring and Observability

Critical metrics for ScyllaDB in ML deployments:

- **Per-shard CPU utilization** (should be balanced)
- **Latency percentiles** (p50, p95, p99)
- **Memory allocation rates**
- **I/O wait times**
- **Network throughput**

Use ScyllaDB Monitoring Stack (Prometheus + Grafana) with custom dashboards for ML-specific metrics.

## Conclusion

ScyllaDB represents the next evolution of Cassandra-compatible databases, delivering the familiar CQL interface with dramatically improved performance characteristics. For AI/ML workloads requiring ultra-low latency, high throughput, and predictable performance, ScyllaDB offers significant advantages over traditional Cassandra deployments.

The key to success with ScyllaDB is leveraging its shard-per-core architecture through proper connection management, partition sizing, and hardware optimization. For AI/ML engineers, this means designing systems that can take full advantage of the linear scalability and sub-millisecond latencies that ScyllaDB provides.

By following the optimization techniques outlined in this tutorial, you can build real-time ML infrastructure that scales to handle the most demanding inference and feature serving requirements.

## Further Reading

- [ScyllaDB Documentation](https://docs.scylladb.com/)
- "ScyllaDB: The High-Performance Cassandra Replacement" by Avi Kivity
- ScyllaDB Summit presentations on ML use cases
- Netflix's ScyllaDB deployment for real-time recommendations
- Uber's ScyllaDB usage for ride-hailing analytics