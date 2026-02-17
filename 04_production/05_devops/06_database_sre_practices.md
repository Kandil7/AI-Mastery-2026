# Database Site Reliability Engineering (SRE) Practices

*Production-grade guide for senior AI/ML engineers building reliable database systems*

**Last Updated**: February 17, 2026  
**Version**: 1.2  
**Target Audience**: Senior AI/ML Engineers, Database Reliability Engineers, Platform Engineers

---

## 1. Introduction to Database SRE Principles

Database SRE is the application of Site Reliability Engineering principles specifically to database systems. Unlike traditional database administration, database SRE focuses on:

- **Reliability as a product feature**: Databases are not just infrastructure but critical components of the service reliability
- **Automation-first approach**: Manual interventions should be exceptions, not the norm
- **Quantitative reliability measurement**: Using SLOs/SLIs instead of qualitative "it feels stable"
- **Error budget-driven development**: Making explicit trade-offs between velocity and reliability

### Core Philosophy

> "A database that works 99.9% of the time is unreliable for AI/ML workloads that require continuous data processing and model inference."

For AI/ML systems, database reliability is non-negotiable because:
- Model training pipelines depend on consistent data availability
- Real-time inference requires low-latency, predictable database responses
- Data drift detection and monitoring rely on accurate, timely data access
- Feature stores must maintain consistency across training and serving environments

### Key Principles

1. **Design for failure**: Assume every component will fail; build resilience into the architecture
2. **Measure everything**: If you can't measure it, you can't improve it
3. **Automate recovery**: Human intervention should be the last resort
4. **Shift left on reliability**: Test reliability early in the development cycle
5. **Blameless post-mortems**: Focus on systemic improvements, not individual accountability

---

## 2. Database SLOs and SLIs

Service Level Objectives (SLOs) define the target reliability, while Service Level Indicators (SLIs) are the measurable metrics that indicate whether you're meeting those objectives.

### Core Database SLIs

| Category | Metric | Measurement Method | Target |
|----------|--------|-------------------|--------|
| **Availability** | Uptime % | Heartbeat checks, connection attempts | ≥99.95% (for critical systems) |
| **Latency** | p95/p99 query latency | Application instrumentation, database logs | ≤100ms (p95), ≤500ms (p99) |
| **Throughput** | Queries per second (QPS) | Database metrics, proxy metrics | Defined by workload profile |
| **Consistency** | Replication lag | Replica lag monitoring | ≤5s (for synchronous replication) |
| **Error Rate** | Failed queries % | Application error logging | ≤0.1% |

### Example SLO Definitions

#### For ML Feature Store:
- **SLO**: 99.99% availability for read operations, 99.95% for write operations
- **SLI**: `successful_read_requests / total_read_requests`
- **Measurement window**: 30-day rolling period

#### For Training Data Pipeline:
- **SLO**: 99.9% of queries complete within 200ms (p95)
- **SLI**: `count(queries where latency < 200ms) / total_queries`
- **Measurement window**: 1-hour sliding window

### Implementation Pattern (Prometheus)

```promql
# Availability SLI for PostgreSQL
(
  sum(rate(pg_up{job="postgres"}[5m])) 
  / 
  sum(rate(pg_up_total{job="postgres"}[5m]))
) * 100

# Latency SLI (p95)
histogram_quantile(0.95, sum(rate(pg_query_duration_seconds_bucket{job="postgres"}[5m])) by (le))
```

---

## 3. Error Budgets and Reliability Trade-offs

Error budgets quantify how much unreliability your system can tolerate before violating SLOs.

### Error Budget Calculation

For a 99.95% monthly availability SLO:
- Total minutes in month: ~43,200
- Allowed downtime: 43,200 × (1 - 0.9995) = 21.6 minutes/month

### Reliability vs. Velocity Trade-offs

| Activity | Impact on Error Budget | Decision Framework |
|----------|------------------------|-------------------|
| New feature deployment | Consumes error budget | Deploy only if remaining budget > 50% |
| Performance optimization | Improves error budget | Prioritize when budget < 20% |
| Technical debt reduction | Neutral/positive impact | Schedule during high-budget periods |
| Experimental features | High risk to budget | Use canary deployments with strict rollback |

### AI/ML Specific Considerations

- **Model retraining jobs**: Should have separate error budgets from serving systems
- **Data validation pipelines**: Critical path for data quality; allocate higher reliability targets
- **Feature engineering workflows**: Can tolerate more downtime than real-time serving

### Error Budget Dashboard Example

```json
{
  "slo_target": "99.95%",
  "current_availability": "99.97%",
  "remaining_budget_minutes": 18.4,
  "budget_consumption_rate": "0.3%/day",
  "warning_threshold": "10 minutes remaining",
  "critical_threshold": "2 minutes remaining"
}
```

---

## 4. Comprehensive Monitoring Strategy

### Key Metrics by Database Type

#### PostgreSQL
- `pg_up`: Database availability
- `pg_query_duration_seconds`: Query latency distribution
- `pg_locks`: Lock contention
- `pg_replication_lag_seconds`: Replication lag
- `pg_connections`: Connection pool utilization
- `pg_checkpoints_total`: Checkpoint frequency

#### MySQL
- `mysql_up`: Server availability
- `mysql_global_status_questions`: Query throughput
- `mysql_global_status_slow_queries`: Slow query count
- `mysql_slave_status_seconds_behind_master`: Replication lag
- `mysql_global_status_threads_connected`: Connection usage

#### MongoDB
- `mongodb_up`: Instance availability
- `mongodb_mongod_operation_total`: Operation counts
- `mongodb_mongod_operation_duration_seconds`: Operation latency
- `mongodb_mongod_repl_lag_seconds`: Replication lag
- `mongodb_mongod_memory_resident_bytes`: Memory usage

#### Redis
- `redis_up`: Server availability
- `redis_commands_processed_total`: Command throughput
- `redis_connected_clients`: Client connections
- `redis_used_memory_bytes`: Memory usage
- `redis_evicted_keys_total`: Eviction rate

### Alerting Thresholds and Escalation Policies

#### Tiered Alerting System

| Severity | Threshold | Response Time | Escalation Path |
|----------|-----------|---------------|----------------|
| **Critical** | SLO violation imminent (≤5 min budget) | 5 minutes | PagerDuty → On-call engineer → Engineering lead |
| **High** | p99 latency > 2× baseline | 15 minutes | Slack → On-call engineer |
| **Medium** | Replication lag > 30s | 1 hour | Email → Team channel |
| **Low** | Connection pool > 80% | 4 hours | Daily digest report |

#### Sample Alert Rules (Prometheus)

```yaml
# Critical: Availability drop
- alert: DatabaseDown
  expr: pg_up{job="postgres"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "PostgreSQL instance down"
    description: "PostgreSQL instance {{ $labels.instance }} has been down for more than 1 minute"

# High: Latency degradation
- alert: HighQueryLatency
  expr: histogram_quantile(0.99, sum(rate(pg_query_duration_seconds_bucket{job="postgres"}[5m])) by (le)) > 1.0
  for: 5m
  labels:
    severity: high
  annotations:
    summary: "High query latency detected"
    description: "p99 query latency exceeds 1s for {{ $labels.instance }}"

# Medium: Replication lag
- alert: ReplicationLagHigh
  expr: pg_replication_lag_seconds{job="postgres"} > 30
  for: 10m
  labels:
    severity: medium
  annotations:
    summary: "Replication lag exceeds threshold"
    description: "Replication lag is {{ $value }} seconds for {{ $labels.instance }}"
```

### Distributed Tracing for Database Operations

Implement tracing at multiple levels:

1. **Application-level tracing** (OpenTelemetry):
   ```python
   # Python example with OpenTelemetry
   from opentelemetry import trace
   from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

   Psycopg2Instrumentor().instrument()
   
   tracer = trace.get_tracer(__name__)
   with tracer.start_as_current_span("database_query") as span:
       span.set_attribute("db.system", "postgresql")
       span.set_attribute("db.statement", "SELECT * FROM features WHERE model_id = %s")
       span.set_attribute("db.operation", "query")
       # Execute query
   ```

2. **Database-level tracing**:
   - PostgreSQL: Enable `log_statement = 'all'` and `log_min_duration_statement = 100`
   - MySQL: Enable slow query log with `slow_query_log = ON`
   - Use tools like Jaeger or Zipkin to correlate traces

3. **Cross-service tracing**:
   - Propagate trace context through HTTP headers
   - Correlate database operations with ML model inference requests
   - Identify bottlenecks in end-to-end ML pipelines

---

## 5. Chaos Engineering for Databases

Chaos engineering proactively tests system resilience by injecting failures in controlled environments.

### Failure Injection Patterns

#### Network Failures
- **Latency injection**: Add 100-500ms network delay to simulate WAN conditions
- **Packet loss**: Inject 1-5% packet loss to test TCP retransmission
- **Connection drops**: Randomly terminate connections to test connection pooling

#### Resource Constraints
- **CPU throttling**: Limit CPU to 50% to simulate overloaded instances
- **Memory pressure**: Allocate memory until OOM killer activates
- **Disk I/O throttling**: Limit IOPS to test storage performance

#### Database-Specific Failures
- **Primary failover**: Force leader election in distributed databases
- **Replica isolation**: Disconnect replicas to test read-only fallback
- **Index corruption**: Simulate index corruption to test recovery
- **Transaction deadlocks**: Create artificial deadlock scenarios

### Resilience Testing Methodologies

#### GameDay Exercises
- **Quarterly**: Full system chaos tests with production-like workloads
- **Monthly**: Component-specific tests (e.g., replica failover)
- **Weekly**: Automated smoke tests with injected failures

#### Automated Chaos Experiments

```python
# Example: Automated chaos experiment for PostgreSQL
import chaosmonkey
from datetime import datetime, timedelta

def test_replica_failover():
    """Test automatic failover when primary becomes unavailable"""
    
    # 1. Establish baseline
    baseline_latency = measure_p99_latency()
    
    # 2. Inject failure
    chaosmonkey.network.disconnect(primary_instance, duration=300)
    
    # 3. Monitor recovery
    start_time = datetime.now()
    while datetime.now() - start_time < timedelta(seconds=300):
        if check_replica_promotion():
            break
        time.sleep(5)
    
    # 4. Verify recovery
    new_latency = measure_p99_latency()
    assert new_latency <= baseline_latency * 1.2, "Latency degraded beyond acceptable threshold"
    
    # 5. Cleanup
    chaosmonkey.network.reconnect(primary_instance)
```

### Chaos Engineering Tools Integration

| Tool | Use Case | Integration Example |
|------|----------|---------------------|
| **Chaos Monkey** | Random instance termination | Kubernetes pod deletion |
| **Gremlin** | Network and resource failures | AWS EC2 instance chaos |
| **Pumba** | Docker container chaos | Container kill, network delay |
| **Litmus** | Kubernetes-native chaos | Custom CRDs for database chaos |
| **Chaos Toolkit** | Custom chaos experiments | Python-based experiment definitions |

---

## 6. Incident Response Runbooks

### Common Database Failure Scenarios

#### Scenario 1: High Query Latency
**Symptoms**: p99 latency spikes, increased error rates, application timeouts

**Runbook**:
1. **Immediate triage (0-5 min)**:
   - Check database CPU, memory, I/O utilization
   - Identify top slow queries using `pg_stat_activity` or `SHOW PROCESSLIST`
   - Verify connection pool saturation

2. **Diagnosis (5-15 min)**:
   - Analyze query execution plans for missing indexes
   - Check for lock contention (`pg_locks`)
   - Review recent deployments or schema changes

3. **Mitigation (15-30 min)**:
   - Kill problematic queries
   - Scale read replicas if available
   - Apply emergency index creation

4. **Recovery (30+ min)**:
   - Implement permanent fix (index, query optimization)
   - Update monitoring thresholds
   - Document lessons learned

#### Scenario 2: Replication Lag
**Symptoms**: Replica lag > 60s, read inconsistencies, failover concerns

**Runbook**:
1. **Immediate triage**:
   - Check replication status: `SHOW SLAVE STATUS` or `pg_stat_replication`
   - Verify network connectivity between nodes
   - Check disk I/O on replica

2. **Diagnosis**:
   - Identify large transactions causing lag
   - Check for long-running queries on replica
   - Review binlog/ WAL size and rotation

3. **Mitigation**:
   - Pause non-critical writes temporarily
   - Optimize large batch operations
   - Scale replica resources

#### Scenario 3: Connection Pool Exhaustion
**Symptoms**: "Too many connections" errors, application timeouts

**Runbook**:
1. **Immediate triage**:
   - Check current connection count vs max_connections
   - Identify leaking connections using `pg_stat_activity`
   - Verify application connection pool configuration

2. **Diagnosis**:
   - Check for unclosed connections in application code
   - Review connection timeout settings
   - Analyze connection lifecycle patterns

3. **Mitigation**:
   - Increase max_connections temporarily
   - Implement connection recycling
   - Deploy circuit breaker pattern

### Post-Mortem Analysis Template

```markdown
## Incident Summary
- **Date/Time**: YYYY-MM-DD HH:MM UTC
- **Duration**: X hours Y minutes
- **Impact**: [Severity level] - [Affected services/users]
- **Root Cause**: [Primary cause]

## Timeline
| Time | Event | Owner |
|------|-------|-------|
| HH:MM | Alert triggered | Monitoring system |
| HH:MM | Initial response | On-call engineer |
| HH:MM | Mitigation applied | Engineering team |
| HH:MM | Service restored | SRE team |

## Contributing Factors
1. [Factor 1 - e.g., insufficient monitoring coverage]
2. [Factor 2 - e.g., lack of automated recovery]
3. [Factor 3 - e.g., configuration drift]

## Action Items
| Priority | Action | Owner | Due Date | Status |
|----------|--------|-------|----------|--------|
1. [Action] | [Owner] | [Date] | [Pending/In Progress/Completed] |

## Metrics Impact
- Error budget consumed: X%
- SLO compliance: Y%
- Customer impact: Z users affected
```

---

## 7. Capacity Planning and Scaling Strategies

### Capacity Planning Framework

1. **Workload characterization**:
   - Read/write ratio analysis
   - Query pattern profiling
   - Data growth forecasting
   - Peak vs average load analysis

2. **Resource forecasting**:
   - CPU: Based on query complexity and concurrency
   - Memory: Based on working set size and cache requirements
   - Storage: Based on data volume + growth rate + retention policies
   - Network: Based on replication traffic and client connections

### Scaling Strategies

#### Vertical Scaling
- **When to use**: Quick fixes, temporary capacity needs
- **Limitations**: Hardware limits, single point of failure
- **Best practices**: 
  - Monitor resource utilization trends
  - Plan for maximum hardware configuration
  - Test scaling limits before production

#### Horizontal Scaling
- **Read scaling**: Read replicas, caching layers
- **Write scaling**: Sharding, partitioning, distributed databases
- **Hybrid approaches**: Vitess, Citus, CockroachDB

#### AI/ML Specific Scaling Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Feature store sharding** | Large-scale ML feature storage | Shard by model_id or feature_group |
| **Training data partitioning** | Distributed training datasets | Partition by time window or data source |
| **Inference caching** | Real-time model serving | Redis/Memcached with TTL based on data freshness |
| **Batch processing queues** | Asynchronous data processing | Kafka/Pulsar with consumer scaling |

### Capacity Planning Calculator

```python
def calculate_database_capacity(workload):
    """
    Calculate required database resources based on workload characteristics
    """
    # Base requirements
    base_cpu = 4  # vCPUs for management overhead
    base_memory = 16  # GB for OS and database overhead
    
    # Workload-based scaling
    query_complexity_factor = workload['avg_query_complexity']  # 1.0-5.0 scale
    concurrency_factor = workload['peak_concurrency'] / 100  # normalized to 100 connections
    
    # CPU calculation
    required_cpu = base_cpu + (workload['qps'] * 0.01 * query_complexity_factor * concurrency_factor)
    
    # Memory calculation (working set + cache)
    working_set_size = workload['active_data_size_gb'] * 0.7  # 70% of active data in memory
    cache_size = workload['qps'] * 0.1  # 100MB per 1000 QPS for query cache
    required_memory = base_memory + working_set_size + cache_size
    
    # Storage calculation
    required_storage = (workload['data_volume_gb'] * 
                       (1 + workload['growth_rate_monthly'] * 12) * 1.5)  # 12-month forecast + 50% buffer
    
    return {
        'cpu_vcpus': math.ceil(required_cpu),
        'memory_gb': math.ceil(required_memory),
        'storage_gb': math.ceil(required_storage),
        'network_gbps': max(1, workload['qps'] * 0.001)  # 1Gbps per 1000 QPS minimum
    }
```

---

## 8. Performance Budgeting and Optimization

### Performance Budget Framework

Define performance budgets similar to error budgets, but for latency and throughput:

- **Latency budget**: Maximum allowable latency for critical paths
- **Throughput budget**: Minimum required throughput for business operations
- **Resource budget**: Maximum resource consumption (CPU, memory, I/O)

### Optimization Strategies

#### Query-Level Optimization
- **Index optimization**: Use covering indexes, partial indexes
- **Query rewriting**: Avoid N+1 queries, use batch operations
- **Parameterization**: Prevent plan cache bloat with parameterized queries

#### Schema Design Optimization
- **Denormalization**: For read-heavy ML workloads
- **Time-series optimization**: Partition by time, use appropriate data types
- **JSON/BSON optimization**: Index nested fields, avoid deep nesting

#### Database Configuration Optimization
- **Shared buffers**: 25-40% of available RAM for PostgreSQL
- **WAL settings**: Tune based on write workload
- **Checkpoint settings**: Balance between I/O load and crash recovery time

### Performance Testing Framework

```bash
# Example: Automated performance testing
./perf-test.sh \
  --database postgres \
  --concurrency 100 \
  --duration 300 \
  --query-file queries/ml_workload.sql \
  --metrics-output perf_results.json \
  --thresholds "p95_latency<200,p99_latency<500,error_rate<0.01"
```

### AI/ML Specific Performance Patterns

| Pattern | Optimization Technique | Example |
|---------|------------------------|---------|
| **Feature retrieval** | Materialized views, precomputed aggregates | Daily feature snapshots |
| **Model training data** | Columnar storage, compression | Parquet format in data lake |
| **Real-time inference** | In-memory caching, connection pooling | Redis feature cache |
| **Data validation** | Incremental processing, sampling | Validate 1% of data continuously |

---

## 9. Production Deployment Patterns

### Zero-Downtime Deployment Strategies

#### Blue-Green Deployments
- Maintain two identical environments (blue/green)
- Route traffic gradually from blue to green
- Rollback by switching back to blue
- **Database considerations**: Schema migration synchronization

#### Canary Deployments
- Deploy to small subset of instances first
- Monitor metrics before full rollout
- Automatic rollback on SLO violations
- **Database considerations**: Feature flags for schema changes

#### Rolling Updates
- Update instances one-by-one
- Health checks between updates
- Circuit breaker patterns for failed updates
- **Database considerations**: Online schema migration tools

### Database Migration Patterns

#### Online Schema Changes
- **pt-online-schema-change** (MySQL): Copy table, apply changes incrementally
- **pg_partman** (PostgreSQL): Partition-based schema evolution
- **Liquibase/Flyway**: Version-controlled migrations with rollback

#### Zero-Downtime Migration Checklist
1. [ ] Backward-compatible schema changes
2. [ ] Dual-write strategy for data migration
3. [ ] Feature flags for new functionality
4. [ ] Automated rollback procedures
5. [ ] Monitoring for migration progress
6. [ ] Validation of data consistency

### CI/CD Integration

```yaml
# Example GitHub Actions workflow for database deployments
name: Database Deployment
on:
  push:
    branches: [main]
    paths:
      - 'database/migrations/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Validate migrations
      run: ./scripts/validate-migrations.sh
      
    - name: Run pre-deployment tests
      run: ./scripts/run-db-tests.sh
      
    - name: Deploy to staging
      run: ./scripts/deploy-staging.sh
      
    - name: Run integration tests
      run: ./scripts/run-integration-tests.sh
      
    - name: Deploy to production (manual approval)
      if: github.ref == 'refs/heads/main'
      uses: softprops/action-gh-release@v1
      with:
        draft: true
        prerelease: true
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Production Readiness Checklist

| Category | Check | Status |
|----------|-------|--------|
| **Monitoring** | SLOs defined and measured | ☐ |
| **Alerting** | Tiered alerting configured | ☐ |
| **Chaos** | Automated chaos experiments | ☐ |
| **Incident** | Runbooks documented and tested | ☐ |
| **Scaling** | Capacity planning completed | ☐ |
| **Deployment** | Zero-downtime procedures | ☐ |
| **Security** | Encryption, authentication, auditing | ☐ |
| **Backup** | RTO/RPO validated | ☐ |

---

## Appendix: Real-World Case Studies

### Case Study 1: Feature Store Outage (2025)
**Problem**: 15-minute outage due to unbounded query on large feature table  
**Root Cause**: Missing index on timestamp column, combined with unbounded date range query  
**Resolution**: 
- Immediate: Added index, implemented query timeout
- Long-term: Query validation middleware, automated index recommendations
- Result: 99.999% availability achieved, error budget consumption reduced by 80%

### Case Study 2: ML Training Pipeline Degradation
**Problem**: Training jobs slowed by 300% due to database contention  
**Root Cause**: Concurrent feature extraction jobs competing for same resources  
**Resolution**:
- Implemented job queuing with priority levels
- Added dedicated read replicas for training workloads
- Optimized query patterns for batch processing
- Result: Training time reduced by 40%, resource utilization optimized

### Case Study 3: Multi-Region Replication Failure
**Problem**: Cross-region replication lag exceeded 10 minutes  
**Root Cause**: Network congestion during peak hours, insufficient bandwidth allocation  
**Resolution**:
- Increased replication bandwidth allocation
- Implemented compression for replication traffic
- Added regional failover testing
- Result: Replication lag consistently < 30s, improved multi-region reliability

---

## References and Further Reading

1. Google SRE Book - Chapter 4: Service Level Objectives
2. "Designing Data-Intensive Applications" - Martin Kleppmann
3. AWS Database Best Practices
4. PostgreSQL Performance Tuning Guide
5. MySQL 8.0 Reference Manual - Performance Schema
6. CNCF Chaos Engineering Whitepaper
7. OpenTelemetry Specification for Database Instrumentation

*This document is maintained by the AI/ML Platform Engineering team. Contributions and feedback welcome.*