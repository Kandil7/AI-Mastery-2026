# Replication Patterns

Database replication is a fundamental technique for achieving high availability, scalability, and disaster recovery. Understanding different replication patterns is essential for building robust AI/ML systems.

## Overview

Replication involves maintaining multiple copies of data across different nodes or locations. For senior AI/ML engineers, choosing the right replication pattern impacts system reliability, performance, and consistency guarantees.

## Replication Types

### Synchronous Replication
- **Definition**: Primary node waits for replicas to acknowledge writes
- **Consistency**: Strong consistency (ACID)
- **Latency**: Higher due to network round trips
- **Availability**: Lower (if replica fails, primary may block)

### Asynchronous Replication
- **Definition**: Primary node acknowledges writes immediately, replicates later
- **Consistency**: Eventual consistency
- **Latency**: Lower (no waiting for replicas)
- **Availability**: Higher (replica failures don't block primary)

### Semi-Synchronous Replication
- **Definition**: Primary waits for at least one replica to acknowledge
- **Consistency**: Stronger than async, weaker than sync
- **Latency**: Moderate
- **Availability**: Good balance

## Common Replication Architectures

### Master-Slave (Primary-Replica)
- **Architecture**: One primary node, multiple read-only replicas
- **Use Case**: Read-heavy workloads, reporting, analytics
- **Advantages**: Simple, good read scalability
- **Disadvantages**: Single point of failure, write bottleneck

```sql
-- PostgreSQL streaming replication setup
-- Primary configuration
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET max_wal_senders = 10;
ALTER SYSTEM SET hot_standby = on;

-- Replica configuration
ALTER SYSTEM SET primary_conninfo = 'host=primary_host port=5432 user=replicator';
```

### Multi-Master Replication
- **Architecture**: Multiple nodes can accept writes
- **Use Case**: Geographically distributed applications
- **Advantages**: High write availability, low latency writes
- **Disadvantages**: Complex conflict resolution, eventual consistency

### Active-Passive (Failover) Replication
- **Architecture**: Primary active, standby passive (hot/warm/cold)
- **Use Case**: Disaster recovery, high availability
- **Advantages**: Fast failover, strong consistency
- **Disadvantages**: Resource underutilization, failover complexity

## Replication Strategies for AI/ML Workloads

### Training Data Replication
- **Strategy**: Async replication with periodic consistency checks
- **Rationale**: Training data can tolerate some inconsistency
- **Implementation**: 
  - Primary: Real-time data ingestion
  - Replicas: Batch processing and model training
  - Consistency: Checkpoint-based validation

### Inference Data Replication
- **Strategy**: Synchronous or semi-synchronous for critical paths
- **Rationale**: Real-time inference requires consistent data
- **Implementation**:
  - Primary: Low-latency serving
  - Replicas: Geographic distribution for edge inference
  - Consistency: Quorum-based writes for critical data

### Model Registry Replication
- **Strategy**: Multi-master with conflict resolution
- **Rationale**: Model versions need global availability
- **Implementation**:
  - Conflict resolution: Last-write-wins or application-level merging
  - Version vectors: Track causal relationships
  - Validation: Schema and integrity checks

## Conflict Resolution Techniques

### Last-Write-Wins (LWW)
- Simple timestamp-based resolution
- Easy to implement but can lose data
- Suitable for non-critical data

### Vector Clocks
- Track causal relationships between operations
- More complex but preserves causality
- Suitable for collaborative applications

### Application-Level Merging
- Custom logic to merge conflicting updates
- Most flexible but requires domain knowledge
- Suitable for complex business logic

### Example - Conflict Resolution Implementation
```sql
-- LWW with timestamp
CREATE TABLE model_versions (
    version_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    version_number INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    -- Conflict resolution: latest updated_at wins
    CONSTRAINT unique_version UNIQUE (model_id, version_number)
);

-- Vector clock example (simplified)
CREATE TABLE model_versions_vc (
    version_id UUID PRIMARY KEY,
    model_id UUID NOT NULL,
    version_number INT NOT NULL,
    vector_clock JSONB NOT NULL,  -- {"node1": 1, "node2": 3}
    data JSONB NOT NULL
);
```

## Performance Considerations

### Replication Lag Monitoring
```sql
-- PostgreSQL replication lag monitoring
SELECT 
    application_name,
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    (sent_lsn - replay_lsn) AS lag_bytes
FROM pg_stat_replication;

-- Calculate lag in seconds
SELECT 
    application_name,
    EXTRACT(EPOCH FROM (NOW() - pg_last_xact_replay_timestamp())) AS replay_lag_seconds
FROM pg_stat_replication;
```

### Throughput Optimization
- **Batching**: Group multiple writes for replication
- **Compression**: Compress WAL logs for network efficiency
- **Parallel replication**: Multiple replication slots
- **Selective replication**: Replicate only critical tables

## AI/ML Specific Patterns

### Feature Store Replication
- **Hot/Cold separation**: Hot features (real-time) vs cold features (historical)
- **Geographic replication**: Edge feature stores for low-latency inference
- **Consistency requirements**: Vary by feature type (critical vs non-critical)

### Experiment Tracking Replication
- **Event sourcing**: Replicate events rather than state
- **Causal consistency**: Ensure experiment lineage integrity
- **Audit trails**: Replicate metadata for reproducibility

### Model Serving Replication
- **Canary deployments**: Gradual rollout with replication
- **A/B testing**: Replicated instances for different variants
- **Shadow traffic**: Replicate production traffic for testing

## Best Practices

1. **Monitor replication lag**: Set alerts for abnormal delays
2. **Test failover regularly**: Ensure DR plans work
3. **Choose appropriate consistency**: Balance needs vs performance
4. **Implement conflict resolution**: Before problems occur
5. **Document replication topology**: For operational clarity
6. **Consider network topology**: Latency between nodes affects choice

## Related Resources

- [High Availability] - Comprehensive HA strategies
- [Disaster Recovery] - DR planning and implementation
- [Distributed Transactions] - Handling transactions across replicas
- [Consistency Models] - Theoretical foundations for replication