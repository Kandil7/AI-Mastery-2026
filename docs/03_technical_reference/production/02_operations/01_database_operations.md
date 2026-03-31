# Database Operations

Database operations encompass the day-to-day management, monitoring, and maintenance of production database systems. For senior AI/ML engineers, understanding operational practices is essential for building reliable, maintainable AI systems.

## Overview

Database operations ensure that database systems remain available, performant, and secure in production environments. This includes monitoring, backup/recovery, performance tuning, and incident response.

## Core Operational Areas

### Monitoring and Alerting
- **Health metrics**: CPU, memory, disk I/O, network
- **Performance metrics**: Query latency, throughput, connection counts
- **Business metrics**: Error rates, success rates, SLA compliance
- **Alerting**: Threshold-based, anomaly detection, escalation policies

### Backup and Recovery
- **Backup strategies**: Full, incremental, differential
- **Recovery objectives**: RPO (Recovery Point Objective), RTO (Recovery Time Objective)
- **Testing**: Regular backup restoration testing
- **Geographic redundancy**: Multi-region backups

### Performance Tuning
- **Query optimization**: Indexing, query rewriting
- **Configuration tuning**: Memory, cache, connection pools
- **Hardware optimization**: Storage, CPU, network
- **Capacity planning**: Growth forecasting and scaling

### Incident Response
- **Runbooks**: Standardized procedures for common incidents
- **Escalation paths**: Clear ownership and escalation procedures
- **Post-mortems**: Blameless analysis of incidents
- **Continuous improvement**: Implement lessons learned

## Monitoring Implementation

### Key Metrics to Monitor

#### System-Level Metrics
- **CPU utilization**: >80% sustained indicates capacity issues
- **Memory usage**: >90% indicates potential OOM risks
- **Disk I/O**: High latency (>10ms) indicates storage bottlenecks
- **Network throughput**: Saturation indicates network issues

#### Database-Specific Metrics
- **Query latency**: P95, P99 response times
- **Connection pool usage**: >80% indicates connection exhaustion
- **Cache hit ratio**: <90% indicates cache inefficiency
- **Replication lag**: >60 seconds indicates replication issues

#### Application-Level Metrics
- **Error rates**: 5xx errors, query failures
- **Throughput**: Queries per second, transactions per second
- **Queue depths**: Connection queues, query queues
- **Timeout rates**: Query timeouts, connection timeouts

### Monitoring Stack Example
```yaml
# Prometheus configuration
scrape_configs:
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']  # postgres_exporter
  - job_name: 'mysql'
    static_configs:
      - targets: ['localhost:9104']  # mysqld_exporter

# Grafana dashboard example
panels:
  - title: "Database Health"
    metrics: ["up", "pg_up", "pg_connections"]
  - title: "Query Performance"
    metrics: ["pg_stat_activity_query_duration_seconds", "pg_stat_statements_mean_exec_time"]
  - title: "Replication Lag"
    metrics: ["pg_replication_lag_seconds"]
```

## Backup and Recovery Strategies

### Backup Types
```sql
-- Full backup (daily)
pg_dump -U postgres -Fc mydatabase > backup_$(date +%Y%m%d).dump

-- Incremental backup (hourly)
pg_basebackup -D /backup/incremental -Ft -z -P -R

-- WAL archiving (continuous)
-- Configure wal_level = replica and archive_mode = on
-- Archive WAL files to S3 or other storage
```

### Recovery Objectives
- **RPO (Recovery Point Objective)**: Maximum acceptable data loss
  - Real-time: 0 seconds (synchronous replication)
  - Near-real-time: 1-5 minutes (asynchronous replication)
  - Daily: 24 hours (daily backups)

- **RTO (Recovery Time Objective)**: Maximum acceptable downtime
  - Hot standby: < 1 minute (failover to replica)
  - Warm standby: 5-30 minutes (restore from backup)
  - Cold standby: 1-24 hours (full restore)

### Disaster Recovery Plan
1. **Detection**: Automated monitoring detects failure
2. **Assessment**: Determine failure type and impact
3. **Response**: Execute appropriate runbook
4. **Recovery**: Restore service using backups or failover
5. **Verification**: Confirm system functionality
6. **Post-mortem**: Analyze root cause and implement fixes

## Performance Optimization

### Query Optimization
```sql
-- Before optimization
SELECT * FROM users u 
JOIN orders o ON u.id = o.user_id 
WHERE u.created_at > '2024-01-01' 
ORDER BY o.created_at DESC 
LIMIT 10;

-- After optimization
-- Add composite index
CREATE INDEX idx_users_created ON users(created_at);
CREATE INDEX idx_orders_user_created ON orders(user_id, created_at DESC);

-- Rewrite query to use covering index
SELECT u.name, o.total 
FROM users u 
INNER JOIN orders o ON u.id = o.user_id 
WHERE u.created_at > '2024-01-01' 
ORDER BY o.created_at DESC 
LIMIT 10;
```

### Configuration Tuning
```ini
# PostgreSQL tuning parameters
shared_buffers = 256MB          # 25% of RAM
work_mem = 4MB                  # Per operation memory
maintenance_work_mem = 64MB     # Maintenance operations
effective_cache_size = 1GB      # OS cache estimate
random_page_cost = 1.1          # SSD storage
checkpoint_completion_target = 0.9  # Smoother checkpoints
```

### Capacity Planning
- **Growth forecasting**: Analyze historical growth patterns
- **Resource modeling**: Model resource requirements per transaction
- **Stress testing**: Simulate peak loads and capacity limits
- **Auto-scaling**: Configure based on metrics thresholds

## AI/ML Specific Operational Considerations

### Model Serving Infrastructure
- **Real-time monitoring**: Inference latency, error rates
- **Model version tracking**: Track deployed model versions
- **Canary deployments**: Gradual rollout with monitoring
- **A/B testing**: Traffic splitting with performance monitoring

### Training Pipeline Operations
- **Data pipeline monitoring**: ETL job status, data quality
- **Training job monitoring**: GPU/CPU utilization, training progress
- **Checkpoint management**: Automatic checkpointing and recovery
- **Resource allocation**: Dynamic resource allocation based on priority

### Data Quality Monitoring
- **Schema validation**: Ensure data conforms to expected schema
- **Statistical validation**: Check for anomalies in data distributions
- **Drift detection**: Monitor for concept drift in training data
- **Lineage tracking**: Track data provenance and transformations

## Incident Response Framework

### Common Database Incidents
- **High CPU/memory**: Query optimization, connection leaks
- **Slow queries**: Index missing, bad query plans
- **Replication lag**: Network issues, heavy write load
- **Connection exhaustion**: Connection pool misconfiguration
- **Disk full**: Log rotation, cleanup policies

### Runbook Example: High Query Latency
1. **Identify**: Alert triggered for high P99 query latency
2. **Diagnose**: 
   - Check pg_stat_activity for long-running queries
   - Review EXPLAIN ANALYZE for problematic queries
   - Check for missing indexes
3. **Mitigate**: 
   - Kill problematic queries if necessary
   - Add missing indexes
   - Optimize query patterns
4. **Verify**: Monitor metrics to confirm resolution
5. **Document**: Update runbook with findings

## Best Practices

1. **Automate monitoring**: Comprehensive coverage with automated alerts
2. **Test backups regularly**: Quarterly restore tests minimum
3. **Document runbooks**: Clear, step-by-step procedures
4. **Implement chaos engineering**: Regular failure injection testing
5. **Cross-train team members**: Prevent knowledge silos
6. **Review metrics weekly**: Proactive issue identification

## Related Resources

- [Database Security] - Security aspects of operations
- [Scalability Patterns] - Scaling considerations for operations
- [AI/ML System Design] - Operational aspects of ML systems
- [Observability] - Comprehensive observability practices