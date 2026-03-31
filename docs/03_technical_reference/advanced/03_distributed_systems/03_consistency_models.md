# Consistency Models

Consistency models define how and when changes to data become visible to readers in distributed systems. Understanding consistency models is essential for senior AI/ML engineers building reliable, scalable applications.

## Overview

In distributed systems, the CAP theorem forces trade-offs between consistency, availability, and partition tolerance. Consistency models provide different guarantees about data visibility and ordering, allowing architects to choose the right model for their use case.

## Core Consistency Models

### Strong Consistency
- **Definition**: All nodes see the same data at the same time
- **Guarantees**: Linearizability, sequential consistency
- **Examples**: Traditional RDBMS, CockroachDB, Spanner
- **Use cases**: Financial transactions, inventory management

### Eventual Consistency
- **Definition**: Updates propagate eventually; no guarantee on timing
- **Guarantees**: Convergence, no conflicts (with conflict resolution)
- **Examples**: DynamoDB, Cassandra, Riak
- **Use cases**: User profiles, content delivery, IoT telemetry

### Causal Consistency
- **Definition**: Preserve causal relationships between operations
- **Guarantees**: If operation A causes operation B, all nodes see A before B
- **Examples**: Amazon DynamoDB (with appropriate configuration), FoundationDB
- **Use cases**: Social networks, collaborative editing, messaging

### Session Consistency
- **Definition**: Per-session guarantees (read-your-writes, monotonic reads)
- **Guarantees**: Within a session, certain consistency properties hold
- **Examples**: Most cloud databases with session affinity
- **Use cases**: Web applications, user sessions, interactive applications

## Model Comparison

| Model | Read-Your-Writes | Monotonic Reads | Monotonic Writes | Bounded Staleness |
|-------|------------------|-----------------|------------------|-------------------|
| Strong | ✓ | ✓ | ✓ | ✓ |
| Causal | ✓ | ✓ | ✓ | ✗ |
| Session | ✓ | ✓ | ✓ | ✗ |
| Eventual | ✗ | ✗ | ✓ | ✗ |

## Implementation Patterns

### Vector Clocks
```python
class VectorClock:
    def __init__(self, node_id):
        self.clock = {node_id: 0}
        self.node_id = node_id
    
    def increment(self):
        self.clock[self.node_id] += 1
    
    def merge(self, other_clock):
        for node, time in other_clock.clock.items():
            self.clock[node] = max(self.clock.get(node, 0), time)
    
    def is_after(self, other_clock):
        # Check if this clock is after another clock
        # (all components >= and at least one >)
        greater = False
        for node, time in other_clock.clock.items():
            if self.clock.get(node, 0) < time:
                return False
            if self.clock.get(node, 0) > time:
                greater = True
        return greater
    
    def to_dict(self):
        return self.clock
```

### Conflict-Free Replicated Data Types (CRDTs)
```python
# Counter CRDT (G-Counter)
class GCounter:
    def __init__(self, node_id):
        self.counters = {node_id: 0}
        self.node_id = node_id
    
    def increment(self):
        self.counters[self.node_id] += 1
    
    def merge(self, other):
        for node, count in other.counters.items():
            self.counters[node] = max(self.counters.get(node, 0), count)
    
    def value(self):
        return sum(self.counters.values())

# Set CRDT (OR-Set)
class ORSet:
    def __init__(self, node_id):
        self.adds = {}
        self.removes = {}
        self.node_id = node_id
    
    def add(self, element):
        self.adds[element] = self.adds.get(element, 0) + 1
    
    def remove(self, element):
        self.removes[element] = self.removes.get(element, 0) + 1
    
    def merge(self, other):
        for element, count in other.adds.items():
            self.adds[element] = max(self.adds.get(element, 0), count)
        for element, count in other.removes.items():
            self.removes[element] = max(self.removes.get(element, 0), count)
    
    def value(self):
        return [e for e in self.adds.keys() if self.adds[e] > self.removes.get(e, 0)]
```

### Quorum-Based Consistency
```sql
-- Dynamo-style quorum configuration
-- R (read quorum): minimum replicas to read from
-- W (write quorum): minimum replicas to write to
-- N (replication factor): total replicas

-- Configuration examples:
-- Strong consistency: R = W = N (all replicas)
-- High availability: R = 1, W = N (fast reads, strong writes)
-- Balanced: R = W = N/2 + 1 (majority)

-- Example: N=3, R=2, W=2
-- Write: Write to 2 out of 3 replicas
-- Read: Read from 2 out of 3 replicas
-- Conflict resolution: Last write wins or vector clocks
```

## Database-Specific Implementations

### Cassandra Tunable Consistency
```cql
-- Read consistency levels
SELECT * FROM users WHERE id = 'user1' 
USING CONSISTENCY ONE;  -- Fast, eventual consistency

SELECT * FROM users WHERE id = 'user1' 
USING CONSISTENCY QUORUM;  -- Balanced, majority

SELECT * FROM users WHERE id = 'user1' 
USING CONSISTENCY ALL;  -- Strong consistency, all replicas

-- Write consistency levels
INSERT INTO users (id, name) VALUES ('user1', 'Alice') 
USING CONSISTENCY QUORUM;
```

### DynamoDB Consistency Options
```python
# Strongly consistent read
response = dynamodb.get_item(
    TableName='users',
    Key={'id': {'S': 'user1'}},
    ConsistentRead=True  # Strong consistency
)

# Eventually consistent read (default)
response = dynamodb.get_item(
    TableName='users',
    Key={'id': {'S': 'user1'}}
)
```

### CockroachDB Consistency
```sql
-- CockroachDB uses strong consistency by default
-- But can configure for performance
SET TRANSACTION PRIORITY LOW;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Read-only transactions can use follower reads
BEGIN READ ONLY;
SELECT * FROM accounts WHERE balance > 1000;
COMMIT;
```

## AI/ML Specific Considerations

### Model Training Consistency
- **Parameter synchronization**: Ensure consistent parameter updates across workers
- **Gradient aggregation**: Consistent gradient averaging
- **Checkpoint coordination**: Atomic checkpoint creation
- **Data shuffling**: Consistent data distribution across workers

### Real-time Inference Consistency
- **Model version consistency**: Ensure all replicas serve the same model version
- **State consistency**: Maintain consistent session state across replicas
- **A/B testing**: Atomic switching between model versions
- **Canary deployments**: Coordinated rollout with consistency guarantees

### Data Pipeline Consistency
- **ETL consistency**: End-to-end data consistency guarantees
- **Stream processing**: Exactly-once vs at-least-once semantics
- **CDC synchronization**: Consistent change capture across systems
- **Data validation**: Atomic validation and ingestion

## Performance Trade-offs

### Latency vs Consistency
- **Strong consistency**: Higher latency due to coordination
- **Eventual consistency**: Lower latency, but potential staleness
- **Causal consistency**: Moderate latency, preserves important relationships
- **Session consistency**: Low latency for individual users

### Throughput vs Consistency
- **Strong consistency**: Lower throughput due to coordination overhead
- **Eventual consistency**: Higher throughput, minimal coordination
- **Quorum-based**: Tunable throughput based on R/W configuration
- **Optimistic concurrency**: High throughput under low contention

### Availability vs Consistency
- **Strong consistency**: Lower availability during partitions
- **Eventual consistency**: Higher availability, may serve stale data
- **Causal consistency**: Good availability with meaningful consistency
- **Session consistency**: High availability for individual sessions

## Best Practices

1. **Choose based on business requirements**: Not all systems need strong consistency
2. **Document consistency guarantees**: Make expectations clear to developers
3. **Test under failure conditions**: Verify behavior during network partitions
4. **Monitor consistency violations**: Track staleness and conflicts
5. **Implement proper error handling**: Handle consistency-related errors gracefully
6. **Consider hybrid approaches**: Different consistency models for different data

## Related Resources

- [Distributed Transactions] - How consistency models affect transaction design
- [CAP Theorem] - Theoretical foundation for consistency trade-offs
- [Distributed Databases] - Database implementations of consistency models
- [AI/ML System Design] - Consistency in ML system architecture