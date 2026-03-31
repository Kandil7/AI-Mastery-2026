# High Availability

High Availability (HA) ensures that database systems remain operational during failures, critical for AI/ML applications where downtime impacts model training, inference, and business operations.

## Overview

High Availability refers to system design that minimizes downtime through redundancy, failover mechanisms, and fault tolerance. For senior AI/ML engineers, HA is essential for building production-grade data platforms.

## HA Architectures

### Active-Passive (Primary-Standby)
- **Architecture**: One active node, one or more standby nodes
- **Failover**: Manual or automatic promotion of standby
- **Consistency**: Strong consistency (synchronous replication)
- **Use Case**: Traditional databases, financial systems

### Active-Active (Multi-Master)
- **Architecture**: Multiple nodes can accept writes
- **Failover**: Automatic, no single point of failure
- **Consistency**: Eventual consistency with conflict resolution
- **Use Case**: Global applications, distributed systems

### Read-Replica Architecture
- **Architecture**: One primary, multiple read-only replicas
- **Failover**: Primary failure requires manual intervention
- **Consistency**: Strong consistency (synchronous) or eventual (async)
- **Use Case**: Read-heavy workloads, analytics, reporting

## Implementation Patterns

### PostgreSQL HA with Streaming Replication
```sql
-- Primary configuration
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET max_wal_senders = 10;
ALTER SYSTEM SET hot_standby = on;
ALTER SYSTEM SET synchronous_commit = 'on';
ALTER SYSTEM SET synchronous_standby_names = 'standby1';

-- Standby configuration (recovery.conf)
standby_mode = 'on'
primary_conninfo = 'host=primary port=5432 user=replicator password=secret'
trigger_file = '/tmp/postgresql.trigger.5432'

-- Automatic failover with Patroni
# patroni.yml configuration
scope: postgresql-cluster
name: postgresql-node-1
restapi:
  listen: 127.0.0.1:8008
  connect_address: 127.0.0.1:8008
etcd:
  host: localhost:2379
postgresql:
  listen: 127.0.0.1:5432
  connect_address: 127.0.0.1:5432
  data_dir: /var/lib/postgresql/data
  parameters:
    wal_level: replica
    hot_standby: on
```

### MySQL HA with Group Replication
```sql
-- Enable group replication
SET GLOBAL group_replication_group_name = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
SET GLOBAL group_replication_start_on_boot = OFF;
SET GLOBAL group_replication_local_address = "127.0.0.1:33061";
SET GLOBAL group_replication_group_seeds = "127.0.0.1:33061,127.0.0.1:33062,127.0.0.1:33063";
SET GLOBAL group_replication_bootstrap_group = ON;
START GROUP_REPLICATION;

-- Check cluster status
SELECT * FROM performance_schema.replication_group_members;
```

### MongoDB Replica Sets
```javascript
// Initialize replica set
rs.initiate({
  _id: "ai_ml_rs",
  members: [
    { _id: 0, host: "node1:27017", priority: 2 },
    { _id: 1, host: "node2:27017", priority: 1 },
    { _id: 2, host: "node3:27017", priority: 1, arbiterOnly: true }
  ]
});

// Add members
rs.add("node2:27017");
rs.add("node3:27017");

// Check status
rs.status();
```

## AI/ML Specific HA Requirements

### Model Serving HA
- **Requirements**: Low-latency failover (< 30 seconds)
- **Architecture**: Load balancer + multiple serving instances
- **Data consistency**: Strong consistency for model metadata
- **State management**: Session state in Redis or similar

### Feature Store HA
- **Requirements**: High read availability, eventual consistency acceptable
- **Architecture**: Read replicas + caching layer
- **Data freshness**: Tolerate some staleness for performance
- **Recovery**: Fast restore from backups

### Training Data Pipeline HA
- **Requirements**: At-least-once processing guarantee
- **Architecture**: Distributed processing with checkpointing
- **Fault tolerance**: Retry failed batches, idempotent operations
- **Monitoring**: End-to-end pipeline health checks

## Failover Strategies

### Automatic Failover
- **Implementation**: Heartbeat monitoring + automated promotion
- **Tools**: Patroni, Consul, etcd, Kubernetes operators
- **Advantages**: Minimal downtime, no human intervention
- **Disadvantages**: Split-brain risk, complex configuration

### Manual Failover
- **Implementation**: Operator-triggered promotion
- **Advantages**: Controlled process, less risk of errors
- **Disadvantages**: Longer downtime, requires skilled personnel
- **Best for**: Critical systems where automatic failover risks are unacceptable

### Hybrid Approach
- **Implementation**: Automatic detection + manual approval
- **Workflow**: 
  1. System detects failure
  2. Alerts operators
  3. Operators approve failover
  4. Automated promotion executes

## Monitoring and Alerting

### HA Health Metrics
```sql
-- PostgreSQL HA monitoring
SELECT 
    application_name,
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    (sent_lsn - replay_lsn) AS lag_bytes,
    EXTRACT(EPOCH FROM (NOW() - pg_last_xact_replay_timestamp())) AS replay_lag_seconds
FROM pg_stat_replication;

-- MySQL group replication health
SELECT 
    MEMBER_HOST,
    MEMBER_PORT,
    MEMBER_STATE,
    MEMBER_ROLE,
    MEMBER_VERSION
FROM performance_schema.replication_group_members;
```

### Alerting Rules
- **Replication lag > 60 seconds**: Warning
- **Standby not connected**: Critical
- **Primary unavailable**: Critical
- **Quorum loss**: Critical
- **Disk space < 10%**: Warning

## Best Practices

1. **Test failover regularly**: Quarterly drills for critical systems
2. **Document runbooks**: Step-by-step recovery procedures
3. **Monitor end-to-end**: Application-level health checks
4. **Consider network topology**: Latency between nodes affects HA
5. **Implement circuit breakers**: Prevent cascading failures
6. **Review RTO/RPO**: Ensure HA meets business requirements

## Related Resources

- [Disaster Recovery] - Comprehensive DR planning
- [Backup and Recovery] - Data protection strategies
- [Database Security] - Secure HA implementations
- [Operational Excellence] - Production database operations