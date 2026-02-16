# Database Debugging Patterns: Common Issues and Resolution Strategies

## Overview

Database debugging is a systematic process of identifying, diagnosing, and resolving issues in database systems. For senior AI/ML engineers, effective debugging requires understanding both traditional database issues and AI-specific challenges like distributed training data consistency, real-time feature store reliability, and model serving performance.

## Core Debugging Methodology

### The 5-Step Debugging Framework
1. **Reproduce**: Create a minimal test case that reproduces the issue
2. **Isolate**: Narrow down to specific components (network, query, storage)
3. **Analyze**: Collect and interpret diagnostic data
4. **Hypothesize**: Formulate potential root causes
5. **Validate**: Test hypotheses and verify fixes

### Diagnostic Data Collection Strategy
- **Time-series metrics**: Latency, throughput, error rates
- **Log analysis**: Error logs, slow query logs, audit logs
- **Tracing data**: Distributed traces with context propagation
- **System metrics**: CPU, memory, I/O, network statistics

## Common Database Issue Categories

### 1. Performance Bottlenecks

#### Symptoms
- High query latency (> p95 thresholds)
- Low throughput despite available resources
- Resource saturation (CPU, I/O, memory)

#### Root Cause Analysis
```python
class PerformanceDebugger:
    def __init__(self):
        self.metrics = {
            'cpu_utilization': None,
            'io_wait_time': None,
            'query_queue_depth': None,
            'cache_hit_ratio': None
        }
    
    def diagnose_performance_issue(self, symptoms):
        if symptoms['high_latency'] and symptoms['low_cpu']:
            return "I/O bottleneck - check disk subsystem"
        elif symptoms['high_latency'] and symptoms['high_cpu']:
            return "Query optimization needed - analyze execution plans"
        elif symptoms['high_queue_depth'] and symptoms['low_throughput']:
            return "Connection pool exhaustion - increase pool size"
```

#### AI/ML Specific Patterns
- **Training data loading bottlenecks**: Large sequential scans for training datasets
- **Feature store hotspots**: Popular user features causing shard imbalance
- **Model registry contention**: Concurrent model version updates

### 2. Data Consistency Issues

#### Symptoms
- Inconsistent reads across replicas
- Stale data in ML predictions
- Transaction rollback failures

#### Diagnostic Techniques
- **Vector clock analysis**: Trace causal relationships between operations
- **Consistency validation queries**: Compare data across replicas
- **Transaction log inspection**: Analyze commit/rollback patterns

#### Common Scenarios
- **Split-brain scenarios**: Network partitions causing divergent writes
- **Eventual consistency violations**: Applications assuming strong consistency
- **CDC pipeline failures**: Missing or duplicate change events

### 3. Connection and Resource Issues

#### Symptoms
- Connection timeouts
- "Too many connections" errors
- Memory allocation failures

#### Root Cause Analysis Matrix
| Symptom | Likely Cause | Diagnostic Command |
|---------|--------------|-------------------|
| Connection timeout | Network issues, firewall | `telnet db-host 5432` |
| Too many connections | Pool exhaustion, leaks | `SHOW PROCESSLIST` |
| Out of memory | Large queries, poor indexing | `EXPLAIN ANALYZE` |
| Deadlocks | Contention, long transactions | `SHOW ENGINE INNODB STATUS` |

## Advanced Debugging Patterns

### Pattern 1: The Query Execution Path Analysis

#### Step-by-Step Approach
1. **Capture the exact query** causing issues
2. **Run EXPLAIN ANALYZE** to get execution plan
3. **Compare with baseline performance**
4. **Identify expensive operations** (sequential scans, sorts, joins)
5. **Test index modifications** incrementally

#### Example Workflow
```sql
-- Step 1: Capture problematic query
SELECT * FROM feature_store 
WHERE user_id = '123456' 
AND timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC 
LIMIT 100;

-- Step 2: Analyze execution plan
EXPLAIN (ANALYZE, BUFFERS) <query>;

-- Step 3: Identify issues
-- Look for: Seq Scan instead of Index Scan, high Buffers: shared hit/miss ratio

-- Step 4: Test fixes
CREATE INDEX idx_feature_user_timestamp ON feature_store (user_id, timestamp DESC);
```

### Pattern 2: Distributed System Trace Correlation

#### Cross-Service Debugging
- **Trace ID propagation**: Ensure trace IDs flow through all services
- **Latency correlation**: Match database latency with application latency
- **Error correlation**: Link database errors to application failures

#### Implementation Example
```python
# OpenTelemetry trace correlation
def debug_distributed_issue(trace_id):
    # Get all spans with this trace ID
    spans = otel_query(f"spans WHERE trace_id = '{trace_id}'")
    
    # Analyze database spans
    db_spans = [s for s in spans if s.service == "database"]
    
    # Correlate with application spans
    app_spans = [s for s in spans if s.service == "model-server"]
    
    # Calculate total latency breakdown
    total_latency = sum(s.duration for s in spans)
    db_latency = sum(s.duration for s in db_spans)
    app_latency = sum(s.duration for s in app_spans)
    
    return {
        'total_latency': total_latency,
        'db_percentage': (db_latency / total_latency) * 100,
        'bottleneck': 'database' if db_latency > app_latency else 'application'
    }
```

### Pattern 3: Time-Based Issue Diagnosis

#### Chronological Analysis
- **Before/after comparison**: Compare metrics before and after issue onset
- **Seasonal patterns**: Check for time-based correlations (daily peaks, batch jobs)
- **Change correlation**: Link issues to deployments, configuration changes

#### Diagnostic Queries
```sql
-- Find when issue started
SELECT 
    date_trunc('minute', timestamp) as minute,
    COUNT(*) as error_count,
    AVG(latency_ms) as avg_latency
FROM query_logs
WHERE error_code IS NOT NULL
GROUP BY minute
ORDER BY minute DESC
LIMIT 20;

-- Correlate with deployments
SELECT 
    d.deploy_time,
    COUNT(*) as errors_after_deploy
FROM deployments d
JOIN query_logs q ON q.timestamp > d.deploy_time AND q.timestamp < d.deploy_time + INTERVAL '1 hour'
WHERE q.error_code IS NOT NULL
GROUP BY d.deploy_time;
```

## AI/ML Specific Debugging Scenarios

### Scenario 1: Real-time Feature Store Inconsistency

#### Symptoms
- Model predictions vary unexpectedly
- Feature values differ between requests
- Stale features in inference responses

#### Debugging Steps
1. **Verify feature freshness**: Check timestamp of retrieved features
2. **Trace CDC pipeline**: Ensure changes are propagating correctly
3. **Check replica consistency**: Compare primary vs replica feature values
4. **Analyze cache invalidation**: Verify cache eviction policies

#### Diagnostic Tools
```python
class FeatureStoreDebugger:
    def __init__(self, feature_store, cdc_system):
        self.feature_store = feature_store
        self.cdc_system = cdc_system
    
    def diagnose_inconsistency(self, user_id, feature_name):
        # Get current feature value
        current_value = self.feature_store.get(f"user:{user_id}:{feature_name}")
        
        # Get latest CDC event
        latest_event = self.cdc_system.get_latest_event(user_id, feature_name)
        
        # Compare timestamps
        if current_value.timestamp < latest_event.timestamp - timedelta(minutes=5):
            return f"Feature staleness: {current_value.timestamp} vs CDC {latest_event.timestamp}"
        
        # Check replica consistency
        replicas = self.feature_store.get_replicas()
        for replica in replicas:
            replica_value = replica.get(f"user:{user_id}:{feature_name}")
            if abs(current_value.value - replica_value.value) > 0.01:
                return f"Inconsistency between replicas: {current_value.value} vs {replica_value.value}"
```

### Scenario 2: Training Data Pipeline Failures

#### Symptoms
- Training jobs failing with data errors
- Incomplete datasets for training
- Feature engineering jobs timing out

#### Root Cause Analysis
- **Data validation failures**: Schema mismatches, null values
- **Resource constraints**: Memory, disk space, I/O limits
- **Concurrency issues**: Race conditions in distributed processing

#### Debugging Framework
```python
class TrainingPipelineDebugger:
    def __init__(self):
        self.data_validators = [
            self._check_schema_consistency,
            self._check_null_values,
            self._check_data_distribution
        ]
    
    def debug_training_failure(self, job_id):
        # Get job logs and metrics
        logs = self._get_job_logs(job_id)
        metrics = self._get_job_metrics(job_id)
        
        # Run validators
        issues = []
        for validator in self.data_validators:
            result = validator(logs, metrics)
            if result:
                issues.append(result)
        
        # Check resource usage
        if metrics['memory_usage'] > 90:
            issues.append("Memory pressure - consider increasing resources")
        
        return issues
```

## Production Debugging Tools and Techniques

### Automated Root Cause Analysis
```python
class AutoDebugger:
    def __init__(self):
        self.knowledge_base = {
            'high_latency': self._diagnose_high_latency,
            'connection_errors': self._diagnose_connection_errors,
            'data_inconsistency': self._diagnose_data_inconsistency
        }
    
    def auto_diagnose(self, incident_data):
        # Classify incident type
        incident_type = self._classify_incident(incident_data)
        
        # Run appropriate diagnosis
        if incident_type in self.knowledge_base:
            return self.knowledge_base[incident_type](incident_data)
        
        return "Unknown incident type - manual investigation required"
    
    def _classify_incident(self, data):
        if data.get('latency_p99') > 1000 and data.get('error_rate') < 0.01:
            return 'high_latency'
        elif data.get('connection_errors') > 100:
            return 'connection_errors'
        elif data.get('inconsistent_reads') > 0.1:
            return 'data_inconsistency'
        return 'unknown'
```

### Chaos Engineering Integration
- **Failure injection**: Simulate network partitions, node failures
- **Automated verification**: Validate system behavior under stress
- **Post-mortem automation**: Generate incident reports automatically

## Best Practices for Senior Engineers

1. **Build debugging into architecture**: Design systems that are inherently debuggable
2. **Maintain knowledge base**: Document common issues and solutions
3. **Invest in observability**: Rich telemetry enables faster diagnosis
4. **Practice regularly**: Conduct debugging drills and war games
5. **Share learnings**: Create post-mortem templates and runbooks

## Related Resources
- [System Design: Debugging Infrastructure](../03_system_design/debugging_infrastructure.md)
- [Observability: Database Monitoring Fundamentals](../03_system_design/observability/database_monitoring_fundamentals.md)
- [Case Study: Production Database Incident Response](../06_case_studies/incident_response_case_study.md)

---
*Last updated: February 2026 | Target audience: Senior AI/ML Engineers*