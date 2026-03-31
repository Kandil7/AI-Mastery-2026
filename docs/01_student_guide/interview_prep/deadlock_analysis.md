# Deadlock Analysis: Detection, Prevention, and Resolution

## Overview

Deadlocks occur when two or more transactions are waiting for each other to release locks, creating a circular dependency that prevents progress. In AI/ML systems, deadlocks can severely impact training pipelines, real-time inference, and feature engineering workflows, making understanding and resolving them critical for senior engineers.

## Deadlock Fundamentals

### The Four Necessary Conditions (Coffman Conditions)
1. **Mutual Exclusion**: Resources cannot be shared
2. **Hold and Wait**: Processes hold resources while waiting for others
3. **No Preemption**: Resources cannot be forcibly taken from processes
4. **Circular Wait**: A circular chain of processes waiting for resources

### Deadlock Types in Database Systems
- **Row-level deadlocks**: Concurrent updates to same rows
- **Table-level deadlocks**: Schema changes conflicting with DML operations
- **Index deadlocks**: Concurrent index maintenance operations
- **Metadata deadlocks**: System catalog contention

## Deadlock Detection Mechanisms

### Database-Specific Detection

#### PostgreSQL Deadlock Detection
```sql
-- View current deadlocks
SELECT 
    pid,
    query,
    wait_event_type,
    wait_event,
    state,
    backend_start,
    xact_start
FROM pg_stat_activity 
WHERE wait_event_type = 'Lock';

-- Deadlock statistics
SELECT 
    datname,
    deadlocks
FROM pg_stat_database;
```

#### MySQL InnoDB Deadlock Detection
```sql
-- Show latest deadlock information
SHOW ENGINE INNODB STATUS;

-- Deadlock monitoring
SELECT 
    variable_name,
    variable_value
FROM information_schema.global_status
WHERE variable_name LIKE '%INNODB_DEADLOCKS%';
```

#### MongoDB Deadlock Detection
```javascript
// Check for lock contention
db.serverStatus().locks

// Monitor operation latency
db.currentOp({ "secs_running": { "$gt": 10 } })
```

## Advanced Deadlock Analysis Techniques

### Lock Dependency Graph Analysis

#### Building the Wait-for Graph
```python
class DeadlockAnalyzer:
    def __init__(self):
        self.wait_graph = {}
    
    def build_wait_graph(self, lock_info):
        """Build wait-for graph from lock information"""
        for transaction in lock_info['transactions']:
            for waiting_lock in transaction['waiting_for']:
                holder = self._find_lock_holder(waiting_lock)
                if holder:
                    self.wait_graph.setdefault(transaction['id'], []).append(holder)
    
    def detect_cycles(self):
        """Detect cycles in wait-for graph using DFS"""
        visited = set()
        recursion_stack = set()
        
        def dfs(node):
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in self.wait_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    return True
            
            recursion_stack.remove(node)
            return False
        
        for node in self.wait_graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False
```

### Time-Based Deadlock Analysis

#### Deadlock Timeline Reconstruction
```
Time T0: Transaction A acquires lock on Row X
Time T1: Transaction B acquires lock on Row Y  
Time T2: Transaction A requests lock on Row Y (waits)
Time T3: Transaction B requests lock on Row X (waits)
Time T4: Deadlock detected, Transaction A rolled back
```

#### Diagnostic Queries
```sql
-- PostgreSQL: Analyze deadlock history
SELECT 
    datname,
    query,
    age(query_start) as duration,
    wait_event_type,
    wait_event
FROM pg_stat_activity 
WHERE state = 'active' AND wait_event_type = 'Lock'
ORDER BY duration DESC;

-- Identify common deadlock patterns
SELECT 
    regexp_replace(query, '\s+', ' ', 'g') as normalized_query,
    COUNT(*) as occurrence_count
FROM pg_stat_activity 
WHERE state = 'active' AND wait_event_type = 'Lock'
GROUP BY normalized_query
ORDER BY occurrence_count DESC;
```

## AI/ML Specific Deadlock Scenarios

### Scenario 1: Model Training Pipeline Deadlocks

#### Symptoms
- Training jobs hanging indefinitely
- Resource utilization spikes followed by crashes
- Intermittent failures in distributed training

#### Common Patterns
- **Parameter server contention**: Multiple workers updating same model parameters
- **Checkpoint coordination**: Concurrent checkpoint writes and reads
- **Feature cache updates**: Real-time feature updates conflicting with batch processing

#### Example Deadlock Chain
```
Worker 1: Holds lock on model_params_v1 → Requests lock on checkpoint_v2
Worker 2: Holds lock on checkpoint_v2 → Requests lock on model_params_v1
Result: Circular wait → Deadlock
```

### Scenario 2: Real-time Feature Store Deadlocks

#### Symptoms
- Inference requests timing out
- Feature updates failing intermitt2ly
- High lock wait times during peak traffic

#### Root Causes
- **Concurrent feature updates**: Multiple services updating same user features
- **Batch vs real-time conflicts**: Batch processing jobs conflicting with real-time updates
- **Index maintenance**: Automatic index rebuilds during high write load

## Deadlock Prevention Strategies

### Design-Level Prevention

#### 1. Lock Ordering Protocol
- **Consistent ordering**: Always acquire locks in the same order
- **Hierarchical locking**: Acquire parent locks before child locks
- **Timeout-based acquisition**: Use lock timeouts to prevent indefinite waits

#### 2. Optimistic Concurrency Control
- **Version checking**: Compare version numbers before updates
- **Retry logic**: Implement exponential backoff for conflicts
- **Application-level coordination**: Central coordinator for critical operations

#### Implementation Example
```python
class OptimisticLockManager:
    def __init__(self):
        self.retry_limit = 3
        self.base_delay = 0.1
    
    def update_with_optimistic_lock(self, entity_id, data, version):
        for attempt in range(self.retry_limit):
            try:
                # Check current version
                current = self.db.get(entity_id)
                if current.version != version:
                    raise ConcurrentModificationError(
                        f"Version mismatch: expected {version}, got {current.version}"
                    )
                
                # Update with new version
                new_version = version + 1
                data['version'] = new_version
                self.db.update(entity_id, data)
                return {'success': True, 'new_version': new_version}
            
            except ConcurrentModificationError:
                if attempt == self.retry_limit - 1:
                    raise
                time.sleep(self.base_delay * (2 ** attempt))
```

### Database Configuration Prevention

#### PostgreSQL Settings
```sql
-- Increase deadlock timeout (default 1s)
SET deadlock_timeout = '5s';

-- Optimize lock escalation thresholds
SET lock_timeout = '30s';
SET idle_in_transaction_session_timeout = '300s';
```

#### MySQL InnoDB Settings
```ini
# innodb_deadlock_detect = ON (default)
# Reduce lock contention
innodb_thread_concurrency = 32
innodb_read_io_threads = 8
innodb_write_io_threads = 8
```

## Deadlock Resolution Techniques

### Automatic Resolution
- **Victim selection**: Database chooses transaction to rollback
- **Priority-based**: Higher priority transactions survive
- **Cost-based**: Rollback cheapest transaction to undo

### Manual Resolution Strategies
1. **Query rewriting**: Break large transactions into smaller ones
2. **Batch processing**: Process in smaller chunks to reduce lock duration
3. **Read-only optimizations**: Use snapshot isolation for read-heavy workloads

#### AI/ML Specific Resolution
```python
class MLDeadlockResolver:
    def __init__(self):
        self.deadlock_patterns = {
            'training_parameter_update': self._resolve_training_deadlock,
            'feature_store_update': self._resolve_feature_deadlock,
            'model_registry_update': self._resolve_registry_deadlock
        }
    
    def resolve_deadlock(self, deadlock_info):
        # Classify deadlock pattern
        pattern = self._classify_deadlock(deadlock_info)
        
        if pattern in self.deadlock_patterns:
            return self.deadlock_patterns[pattern](deadlock_info)
        
        return self._generic_resolution(deadlock_info)
    
    def _resolve_training_deadlock(self, info):
        # Strategy: Separate parameter updates by worker ID
        # Use sharded parameter storage
        return {
            'action': 'shard_parameters',
            'description': 'Split parameter updates by worker ID to eliminate contention',
            'expected_improvement': '95% deadlock reduction'
        }
    
    def _resolve_feature_deadlock(self, info):
        # Strategy: Implement optimistic locking for feature updates
        return {
            'action': 'optimistic_locking',
            'description': 'Add version numbers to features and use optimistic concurrency',
            'expected_improvement': '80% deadlock reduction'
        }
```

## Production Debugging Workflows

### Step-by-Step Deadlock Investigation

#### Phase 1: Immediate Response
1. **Identify affected transactions**: From database logs
2. **Capture deadlock graph**: Database-specific commands
3. **Rollback victims**: Allow system to recover
4. **Alert stakeholders**: Communicate impact and status

#### Phase 2: Root Cause Analysis
1. **Reproduce scenario**: Create test case with similar workload
2. **Analyze lock patterns**: Identify resource contention points
3. **Review application code**: Find inconsistent lock ordering
4. **Examine deployment patterns**: Check for race conditions in deployments

#### Phase 3: Permanent Fix
1. **Implement prevention**: Lock ordering, optimistic concurrency
2. **Add monitoring**: Alert on deadlock frequency
3. **Update runbooks**: Document resolution procedures
4. **Test thoroughly**: Validate fixes under load

## Performance Impact Analysis

### Deadlock Cost Metrics
| Metric | Low Severity | Medium Severity | High Severity |
|--------|--------------|-----------------|---------------|
| Frequency | < 1/hour | 1-10/hour | > 10/hour |
| Resolution Time | < 1s | 1-10s | > 10s |
| Transaction Loss | < 0.1% | 0.1-1% | > 1% |
| Throughput Impact | < 5% | 5-20% | > 20% |

### AI/ML Workload Impact
- **Training jobs**: Deadlocks cause job failures and wasted compute resources
- **Inference services**: Increased latency and error rates
- **Feature engineering**: Data pipeline delays and inconsistencies

## Best Practices for Senior Engineers

1. **Design for deadlock avoidance**: Implement consistent lock ordering from day one
2. **Monitor deadlock rates**: Set alerts for abnormal increases
3. **Use appropriate isolation levels**: Balance consistency with concurrency
4. **Test under realistic loads**: Simulate production-like contention
5. **Document patterns**: Maintain knowledge base of common deadlock scenarios

## Related Resources
- [System Design: High-Concurrency ML Infrastructure](../03_system_design/high_concurrency_ml.md)
- [Debugging Patterns: Database Performance Bottlenecks](../05_interview_prep/database_debugging_patterns.md)
- [Case Study: Distributed Training Deadlock Resolution](../06_case_studies/training_deadlock_case_study.md)

---
*Last updated: February 2026 | Target audience: Senior AI/ML Engineers*