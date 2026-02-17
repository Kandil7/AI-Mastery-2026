# Database SRE Practices

This comprehensive guide covers Site Reliability Engineering (SRE) practices for database systems, designed for senior AI/ML engineers responsible for production database reliability.

## Introduction to Database SRE

Site Reliability Engineering applies software engineering principles to infrastructure and operations. For databases, this means:

- **Reliability as code**: Treating reliability as a first-class requirement
- **Data-driven decisions**: Using metrics and monitoring to drive improvements
- **Automation first**: Automating repetitive tasks and recovery procedures
- **Blameless post-mortems**: Learning from incidents without assigning blame

### Core SRE Principles for Databases

1. **Service Level Objectives (SLOs)**: Define what "good" looks like
2. **Error Budgets**: Quantify acceptable unreliability
3. **Toil reduction**: Eliminate manual, repetitive work
4. **Progressive delivery**: Safe, incremental changes
5. **Observability**: Comprehensive monitoring and logging

## Database SLOs and SLIs

### Key Service Level Indicators (SLIs)

| Category | Metric | Target | Measurement |
|----------|--------|--------|-------------|
| **Availability** | Uptime percentage | 99.95% (26.3 min/month downtime) | Heartbeat checks, connection attempts |
| **Latency** | p95 query latency | ≤ 100ms for OLTP, ≤ 500ms for analytics | Query timing, application metrics |
| **Throughput** | Queries per second | Varies by workload | Database metrics, application counters |
| **Consistency** | Data consistency | Strong consistency for critical paths | Application validation, checksums |
| **Durability** | Data loss probability | ≤ 0.001% | Backup verification, replication lag |

### Error Budgets and Reliability Trade-offs

```text
Monthly error budget = 100% - SLO target
For 99.95% availability: 0.05% × 720 hours = 26.3 minutes/month

Usage examples:
- 15 minutes for planned maintenance
- 5 minutes for emergency fixes  
- 6.3 minutes for unexpected failures
```

**Trade-off decisions:**
- Higher consistency → Lower availability
- Better durability → Higher cost
- Faster queries → More resources
- Stronger security → Higher latency

## Comprehensive Monitoring Strategy

### Key Metrics by Database Type

#### PostgreSQL/MySQL (Relational)
- **Connection metrics**: Active connections, connection pool usage
- **Query metrics**: Query duration, slow queries, cache hit ratio
- **Replication metrics**: Replication lag, WAL write rate
- **Resource metrics**: CPU, memory, disk I/O, buffer cache hit ratio

#### MongoDB (Document)
- **Operation metrics**: Operations/sec, queue depth
- **Memory metrics**: Resident memory, virtual memory
- **Replication metrics**: Oplog window, replication lag
- **Index metrics**: Index misses, index size

#### Redis (Key-Value)
- **Memory metrics**: Used memory, memory fragmentation
- **Performance metrics**: Commands/sec, latency percentiles
- **Persistence metrics**: RDB/AOF sync time, last save time
- **Client metrics**: Connected clients, blocked clients

#### Vector Databases
- **Search metrics**: Query latency, recall@k, precision@k
- **Index metrics**: Index build time, index size, quantization ratio
- **Resource metrics**: GPU utilization, memory pressure
- **Quality metrics**: Embedding similarity distribution

### Alerting Strategy

**Tiered alerting system:**
- **Critical (P0)**: Database unavailable, data loss, severe performance degradation
- **High (P1)**: Significant performance issues, high error rates
- **Medium (P2)**: Warning conditions, resource pressure
- **Low (P3)**: Informational, capacity planning

**Example Prometheus alerts:**
```yaml
# Critical: Database unreachable
- alert: DatabaseDown
  expr: up{job="database"} == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Database instance {{ $labels.instance }} is down"

# High: High query latency
- alert: HighQueryLatency
  expr: histogram_quantile(0.95, sum(rate(pg_stat_activity_query_duration_seconds_bucket[5m])) by (le))
  > 1.0
  for: 5m
  labels:
    severity: high
  annotations:
    summary: "High query latency on {{ $labels.instance }}"
```

### Distributed Tracing for Database Operations

Implement tracing for end-to-end database operations:
- **Trace spans**: Connection, query execution, result processing
- **Context propagation**: Pass trace IDs through service boundaries
- **Sampling strategy**: Adaptive sampling based on error rates
- **Correlation**: Link database traces with application traces

## Chaos Engineering for Databases

### Failure Injection Patterns

| Failure Type | Injection Method | Purpose |
|--------------|------------------|---------|
| **Network partition** | iptables rules, network emulation | Test split-brain scenarios |
| **Latency injection** | tc qdisc, chaos mesh | Simulate network congestion |
| **CPU/memory pressure** | stress-ng, custom scripts | Test resource exhaustion |
| **Disk failure** | dd, fio, disk removal | Test durability and recovery |
| **Process kill** | kill -9, chaos mesh | Test process restart and recovery |
| **Data corruption** | byte manipulation, checksum errors | Test data integrity mechanisms |

### Resilience Testing Methodology

1. **Hypothesis-driven testing**: Define expected behavior before injecting failure
2. **Controlled experiments**: Start small, increase scope gradually
3. **Automated verification**: Validate system behavior automatically
4. **Post-experiment analysis**: Document findings and improvements

### Automated Chaos Experiments

```python
# Example chaos experiment using Chaos Mesh
class DatabaseChaosExperiment:
    def __init__(self, client):
        self.client = client
    
    def test_network_partition(self):
        """Test database behavior during network partition"""
        # Inject network partition between primary and replica
        self.client.inject_network_partition(
            source="db-primary",
            target="db-replica",
            duration="5m"
        )
        
        # Monitor key metrics during partition
        start_time = time.time()
        while time.time() - start_time < 300:
            metrics = self._collect_metrics()
            if self._detect_failure(metrics):
                return {"status": "failed", "metrics": metrics}
            time.sleep(10)
        
        # Verify recovery after partition ends
        return self._verify_recovery()
```

## Incident Response Runbooks

### Common Database Failure Scenarios

#### 1. Database Unavailability
**Symptoms**: Connection timeouts, 500 errors, high error rates
**Diagnosis**:
- Check database process status
- Verify network connectivity
- Review logs for startup errors
- Check resource constraints (CPU, memory, disk)

**Resolution steps**:
1. Attempt graceful restart
2. If fails, force restart with clean state
3. Restore from backup if necessary
4. Verify data consistency
5. Implement preventive measures

#### 2. High Latency Queries
**Symptoms**: Slow application responses, high p95/p99 latency
**Diagnosis**:
- Identify slow queries using EXPLAIN ANALYZE
- Check for missing indexes
- Monitor resource utilization
- Review query patterns and caching

**Resolution steps**:
1. Optimize problematic queries
2. Add missing indexes
3. Implement query caching
4. Consider read replicas for read-heavy workloads
5. Scale vertically/horizontally as needed

#### 3. Replication Lag
**Symptoms**: Data inconsistency, delayed reads, failover issues
**Diagnosis**:
- Check replication lag metrics
- Review binary log/transaction log sizes
- Monitor network bandwidth between nodes
- Check for long-running transactions

**Resolution steps**:
1. Optimize slow transactions
2. Increase replication bandwidth
3. Tune replication parameters
4. Consider asynchronous vs synchronous replication trade-offs
5. Implement monitoring for early detection

### Post-Mortem Analysis Template

```markdown
## Incident Summary
- **Timeline**: When it happened, duration, impact
- **Root Cause**: Technical cause and contributing factors
- **Impact**: Business impact, affected users/services
- **Response**: Actions taken during incident
- **Lessons Learned**: What we learned
- **Action Items**: Concrete improvements to prevent recurrence
- **Verification**: How we'll verify fixes work
```

## Capacity Planning and Scaling Strategies

### Growth Forecasting Methodology

1. **Historical analysis**: Analyze past growth patterns
2. **Business projections**: Incorporate product roadmap and business goals
3. **Workload modeling**: Model different usage scenarios
4. **Stress testing**: Validate capacity assumptions

### Scaling Patterns

#### Vertical Scaling
- **Pros**: Simple, maintains consistency
- **Cons**: Limited by hardware, single point of failure
- **Best for**: Small to medium workloads, stateful services

#### Horizontal Scaling
- **Sharding**: Partition data across multiple instances
- **Read replicas**: Scale read operations
- **Multi-region**: Geographic distribution for low latency

#### Auto-scaling Strategies
- **Metric-based**: Scale based on CPU, memory, QPS
- **Predictive**: Use ML to forecast demand
- **Scheduled**: Pre-scale for known traffic patterns

### Resource Optimization

**Compute optimization**:
- Right-size instances based on workload patterns
- Use spot instances for non-critical workloads
- Implement connection pooling

**Storage optimization**:
- Tiered storage (hot/warm/cold)
- Compression and deduplication
- Proper indexing to reduce I/O

**Network optimization**:
- Connection reuse and keep-alive
- Local caching for frequently accessed data
- CDN for static content

## Performance Budgeting and Optimization

### Performance Budget Definition

Define performance budgets for different operations:
- **User-facing operations**: ≤ 100ms p95
- **Background jobs**: ≤ 5s p95
- **Batch processing**: ≤ 30min for daily jobs
- **Data exports**: ≤ 1h for large datasets

### Optimization Methodology

1. **Baseline measurement**: Establish current performance
2. **Bottleneck identification**: Use profiling tools
3. **Hypothesis formulation**: Propose optimizations
4. **Implementation**: Apply changes
5. **Validation**: Measure impact
6. **Documentation**: Record findings

### Common Optimization Techniques

#### Query Optimization
- **Index optimization**: Covering indexes, composite indexes
- **Query rewriting**: Avoid N+1 queries, use JOINs efficiently
- **Caching strategies**: Application-level, query-level, result-level
- **Pagination optimization**: Keyset pagination vs offset pagination

#### Schema Optimization
- **Normalization vs denormalization**: Balance consistency and performance
- **Data types**: Choose appropriate types for storage efficiency
- **Partitioning**: Horizontal and vertical partitioning
- **Materialized views**: Pre-compute expensive aggregations

#### Infrastructure Optimization
- **Connection pooling**: Reduce connection overhead
- **Read replicas**: Offload read traffic
- **Caching layers**: Redis/Memcached for hot data
- **CDN integration**: For static content and API responses

## Production Deployment Patterns

### Zero-Downtime Deployments

**Strategies**:
- **Blue-green deployment**: Switch traffic after validation
- **Canary releases**: Gradual rollout with monitoring
- **Feature flags**: Enable/disable features without deployment
- **Database migrations**: Online schema changes, backward-compatible changes

### CI/CD Integration

**Database-specific CI/CD practices**:
- **Migration testing**: Test migrations in staging environment
- **Schema validation**: Verify schema compatibility
- **Data validation**: Check data integrity after migrations
- **Rollback automation**: Automated rollback procedures

### Automated Compliance Validation

```yaml
# Example compliance check in CI/CD pipeline
- name: Database Compliance Check
  run: |
    # Check for encryption at rest
    ./scripts/check-encryption.sh
    
    # Verify backup configuration
    ./scripts/check-backups.sh
    
    # Validate access controls
    ./scripts/check-rbac.sh
    
    # Verify audit logging
    ./scripts/check-audit-logging.sh
```

## Case Studies and Real-World Examples

### Example: E-commerce Database SRE Implementation

**Challenge**: 99.99% availability requirement for checkout database
**Solution**:
- Multi-AZ deployment with automatic failover
- Read replicas for catalog queries
- Connection pooling and circuit breakers
- Automated chaos testing weekly
- Real-time monitoring with anomaly detection

**Results**:
- Achieved 99.992% availability over 12 months
- Reduced incident response time from 45min to 8min
- 60% reduction in database-related P1 incidents

## Best Practices Summary

1. **Start with SLOs**: Define what reliability means for your system
2. **Monitor comprehensively**: Collect metrics across all layers
3. **Automate everything**: From deployments to recovery
4. **Test failures**: Regular chaos engineering exercises
5. **Learn from incidents**: Blameless post-mortems
6. **Scale proactively**: Don't wait for problems to occur
7. **Document everything**: Runbooks, architecture decisions
8. **Review regularly**: Quarterly SRE reviews and adjustments

## Next Steps

- Implement baseline monitoring for your database systems
- Define SLOs and error budgets for critical services
- Create incident response runbooks for common failure modes
- Schedule regular chaos engineering exercises
- Establish quarterly SRE review process

This guide provides a foundation for implementing database SRE practices. Adapt these practices to your specific environment and requirements.