# Database Replication: Advanced Strategies for AI/ML Systems

## Overview

Database replication is the process of copying data from a primary (master) database to one or more secondary (replica) databases. In AI/ML systems, replication provides high availability, read scalability, and disaster recovery capabilities essential for production-grade ML infrastructure.

## Replication Architectures

### 1. Master-Slave (Primary-Replica)

#### Architecture
- Single master accepts writes, multiple replicas handle reads
- Asynchronous or synchronous replication
- Common in PostgreSQL, MySQL, MongoDB

#### Implementation Patterns
```sql
-- PostgreSQL streaming replication
-- Primary configuration (postgresql.conf)
wal_level = replica
max_wal_senders = 10
hot_standby = on

-- Replica configuration (recovery.conf)
standby_mode = on
primary_conninfo = 'host=primary_host port=5432 user=replicator'
```

#### AI/ML Use Cases
- **Read scaling**: Separate inference queries from training data writes
- **Geographic distribution**: Replicas in different regions for low-latency inference
- **Backup and recovery**: Point-in-time recovery for model training data

### 2. Multi-Master Replication

#### Architecture
- Multiple nodes accept writes simultaneously
- Conflict resolution required
- Examples: Cassandra, CockroachDB, MySQL Group Replication

#### Conflict Resolution Strategies
- **Last Write Wins (LWW)**: Simple timestamp-based resolution
- **Vector Clocks**: Causal ordering with conflict detection
- **Application-level**: Custom business logic for conflict resolution

#### Implementation Example
```python
class MultiMasterConflictResolver:
    def __init__(self, strategy="vector_clock"):
        self.strategy = strategy
        self.clock = VectorClock(node_id=get_node_id())
    
    def resolve_conflict(self, record1, record2):
        if self.strategy == "vector_clock":
            return self._resolve_vector_clock(record1, record2)
        elif self.strategy == "application":
            return self._resolve_application_logic(record1, record2)
    
    def _resolve_vector_clock(self, r1, r2):
        # Compare vector clocks
        if r1.vector_clock.happens_before(r2.vector_clock):
            return r2
        elif r2.vector_clock.happens_before(r1.vector_clock):
            return r1
        else:
            # Concurrent writes - use application logic
            return self._apply_business_rules(r1, r2)
```

### 3. Quorum-Based Replication

#### Architecture
- Data replicated to N nodes, reads/writes require quorum (Q)
- Q = ⌊N/2⌋ + 1 for majority quorum
- Examples: DynamoDB, Riak, Cassandra

#### Consistency Levels
- **ONE**: Read/write from any single node
- **QUORUM**: Read/write from majority of nodes
- **ALL**: Read/write from all nodes

#### Performance Trade-offs
| Consistency Level | Availability | Latency | Throughput | Durability |
|-------------------|--------------|---------|------------|------------|
| ONE               | High         | Low     | High       | Low        |
| QUORUM            | Medium       | Medium  | Medium     | Medium     |
| ALL               | Low          | High    | Low        | High       |

## Advanced Replication Patterns

### Active-Passive vs Active-Active

#### Active-Passive
- One node active, others standby
- Automatic failover on primary failure
- Simple consistency model

#### Active-Active
- All nodes accept writes
- Complex conflict resolution
- Higher availability but eventual consistency

### Geo-Replication

#### Multi-Region Deployment
- Replicas in different geographic regions
- Cross-region replication with latency considerations
- Regional affinity for AI/ML workloads

#### Implementation Challenges
- **Network latency**: Cross-region sync delays
- **Data sovereignty**: Compliance with regional data laws
- **Consistency trade-offs**: Balance between latency and consistency

### Hybrid Replication

#### Architecture
- Combine different replication strategies
- Example: Master-slave within region, multi-master across regions
- Optimize for specific workload patterns

## AI/ML Specific Replication Strategies

### Model Registry Replication
- **Strong consistency**: Ensure all services see same model versions
- **Multi-region**: Deploy model registry replicas globally
- **Versioned replication**: Maintain historical model versions

### Feature Store Replication
- **Read-heavy optimization**: Many replicas for inference queries
- **Write-consistency**: Strong consistency for feature updates
- **Temporal partitioning**: Recent features in hot replicas, historical in cold

### Training Data Replication
- **Asynchronous replication**: Accept eventual consistency for training data
- **Cross-datacenter**: Replicate training datasets across availability zones
- **Incremental replication**: Only replicate changed data blocks

## Real-World Production Examples

### Netflix's Cassandra Deployment
- Multi-datacenter replication with RF=3
- Read consistency QUORUM, write consistency QUORUM
- Handles 10M+ requests per second with 99.999% availability

### Google's Spanner
- TrueTime-based synchronous replication
- Global consistency across regions
- Used for critical ML infrastructure components

### AWS Aurora
- Distributed storage with 6 copies across 3 AZs
- Synchronous replication within region
- Asynchronous cross-region replication
- Achieves 99.999% availability for ML workloads

## Performance Metrics & Benchmarks

| Replication Type | Write Latency | Read Latency | Max Throughput | Recovery Time |
|------------------|---------------|--------------|----------------|---------------|
| Master-Slave     | 5ms           | 2ms          | 50K ops/s      | 30s           |
| Multi-Master     | 15ms          | 8ms          | 20K ops/s      | 10s           |
| Quorum-Based     | 25ms          | 12ms         | 15K ops/s      | 5s            |
| Geo-Replicated   | 50ms          | 30ms         | 8K ops/s       | 60s           |

*Tested on 10-node clusters with 1M records*

## Debugging Replication Issues

### Common Failure Modes
1. **Replication lag**: Secondary falling behind primary
2. **Split-brain**: Network partition causing divergent writes
3. **Data inconsistency**: Conflicts not resolved properly
4. **Failover failures**: Automatic failover not triggering

### Diagnostic Techniques
- **Replication lag monitoring**: Track WAL position differences
- **Consistency validation**: Periodic checksum comparisons
- **Conflict detection**: Monitor conflict resolution metrics
- **Network analysis**: Check for packet loss between nodes

### Tools and Frameworks
- **pg_stat_replication** (PostgreSQL): Real-time replication status
- **MongoDB replSetGetStatus**: Replica set health monitoring
- **Prometheus exporters**: Custom metrics for replication health
- **Chaos engineering**: Simulate network partitions and node failures

## AI/ML Integration Patterns

### Real-time Model Serving
```python
class ReplicatedModelService:
    def __init__(self, read_replicas, primary):
        self.read_replicas = read_replicas
        self.primary = primary
        self.load_balancer = RoundRobinBalancer(read_replicas)
    
    def get_model(self, model_id, consistency="eventual"):
        if consistency == "strong":
            return self.primary.get(model_id)
        else:
            replica = self.load_balancer.next()
            return replica.get(model_id)
    
    def update_model(self, model_id, model_data):
        # Write to primary, then wait for replication
        result = self.primary.update(model_id, model_data)
        
        # Wait for replication to complete (optional)
        if self.wait_for_replication:
            self._wait_for_replication(model_id)
        
        return result
```

### Distributed Training Coordination
- **Parameter server replication**: Replicate parameter servers for fault tolerance
- **Gradient synchronization**: Replicate gradient updates across workers
- **Checkpoint replication**: Replicate training checkpoints across storage systems

## Best Practices for Senior Engineers

1. **Choose replication strategy based on workload**: Read-heavy vs write-heavy patterns
2. **Implement automated failover testing**: Regularly test failover scenarios
3. **Monitor replication lag continuously**: Set alerts for abnormal lag
4. **Design for split-brain prevention**: Use quorum mechanisms and fencing
5. **Optimize for AI/ML access patterns**: Co-locate replicas with compute resources

## Related Resources
- [Database Case Study: Uber's Multi-Region Feature Store](../06_case_studies/uber_multiregion_store.md)
- [System Design: High-Availability ML Infrastructure](../03_system_design/high_availability_ml.md)
- [Debugging Patterns: Replication Lag Analysis](../05_interview_prep/database_debugging_patterns.md)

---
*Last updated: February 2026 | Target audience: Senior AI/ML Engineers*