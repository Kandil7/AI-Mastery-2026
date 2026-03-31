# Database Performance Tuning

## Introduction

Database performance tuning encompasses the systematic process of optimizing database systems to achieve maximum efficiency, throughput, and responsiveness. This guide provides comprehensive coverage of performance tuning techniques for PostgreSQL and MySQL, the two most widely used open-source relational databases. Understanding how to tune these database systems is essential for building scalable applications that can handle increasing workloads without degradation.

Performance tuning involves multiple interconnected components, from low-level configuration parameters that control memory allocation and I/O operations, to higher-level architectural decisions about indexing strategies and query patterns. Each aspect of tuning affects the others, making a holistic approach critical. A database configured with optimal memory settings will still perform poorly if queries lack appropriate indexes, and well-designed indexes cannot compensate for misconfigured buffer pools.

This guide covers performance metrics and monitoring approaches to identify bottlenecks, detailed tuning parameters for both PostgreSQL and MySQL with practical examples, connection pooling strategies that reduce overhead, buffer pool optimization techniques, and common query optimization patterns. By following these guidelines and understanding the tradeoffs involved, you can significantly improve your database's performance and ensure it meets the demands of production workloads.

The techniques presented here are based on established best practices and real-world experience tuning databases for high-traffic applications. However, every application has unique characteristics, so use these recommendations as starting points and adjust based on actual performance measurements. The ultimate authority on whether a tuning change is beneficial is always benchmark testing in your specific environment.

---

## 1. Performance Metrics and Monitoring

### Key Performance Indicators

Understanding which metrics matter is fundamental to effective performance tuning. Database systems generate大量的 metrics, but focusing on key performance indicators (KPIs) helps you quickly identify issues and track improvements over time. The most important metrics fall into several categories: throughput, latency, resource utilization, and efficiency.

**Query throughput** measures the number of operations the database can handle per unit of time. Transactions per second (TPS) and queries per second (QPS) are common throughput metrics. High throughput with acceptable latency indicates a well-performing system, while throughput degradation often signals resource contention or configuration issues.

```sql
-- PostgreSQL: Calculate transactions per second
SELECT 
    sum(xact_commit + xact_rollback) / 
        EXTRACT(EPOCH FROM (now() - stats_reset)) AS tps
FROM pg_stat_database 
WHERE datname = current_database();

-- MySQL: Calculate queries per second
SHOW GLOBAL STATUS LIKE 'Questions';
-- Run twice and divide by elapsed seconds
SHOW GLOBAL STATUS LIKE 'Uptime';
```

**Query latency** measures how long individual operations take to complete. Latency is typically measured at multiple percentiles: p50 (median), p95, p99, and sometimes p99.9 for critical systems. Average latency can be misleading because it hides the impact of slow queries, so always examine tail latency.

```sql
-- PostgreSQL: Get latency distribution from pg_stat_statements
SELECT 
    substring(query, 1, 50) AS query_preview,
    calls,
    mean_exec_time,
    stddev_exec_time,
    min_exec_time,
    max_exec_time,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY total_exec_time / calls) AS p95_time
FROM pg_stat_statements
WHERE calls > 100
ORDER BY mean_exec_time DESC
LIMIT 20;
```

**Resource utilization** shows how effectively the database uses available hardware. CPU utilization, memory usage, disk I/O, and network bandwidth all contribute to overall performance. When any resource reaches saturation, performance degrades regardless of other factors.

```bash
# Linux: Monitor database resource usage
# CPU and memory
top -p $(pgrep -d',' -f postgres)

# Disk I/O
iostat -x 1

# Network
iftop -i eth0
```

**Cache efficiency** measures how well the database keeps frequently accessed data in memory. Buffer pool hit ratio is the primary cache metric: a ratio above 95-99% typically indicates good cache performance, while lower ratios suggest either insufficient memory allocation or poor access patterns.

```sql
-- PostgreSQL: Calculate buffer cache hit ratio
SELECT 
    sum(heap_blks_read) AS heap_read,
    sum(heap_blks_hit) AS heap_hit,
    round(sum(heap_blks_hit) * 100.0 / 
          (sum(heap_blks_hit) + sum(heap_blks_read)), 2) AS hit_ratio
FROM pg_statio_user_tables;

-- MySQL: InnoDB buffer pool hit ratio
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_reads';
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read_requests';
-- Calculate: (reads - requests) / reads * 100 = hit ratio
```

### Monitoring Tools and Approaches

Effective monitoring requires both real-time visibility and historical data for trend analysis. Modern database deployments typically combine database-native monitoring with external observability platforms to achieve comprehensive visibility.

```python
# Python: Prometheus exporter for PostgreSQL metrics
from prometheus_client import start_http_server, Gauge
import psycopg2
import time

buffer_hit_ratio = Gauge('postgres_buffer_hit_ratio', 'Buffer pool hit ratio')
active_connections = Gauge('postgres_active_connections', 'Active database connections')
query_latency = Gauge('postgres_query_latency_ms', 'Average query latency in ms')

def collect_metrics():
    conn = psycopg2.connect("dbname=mydb")
    cur = conn.cursor()
    
    # Buffer hit ratio
    cur.execute("""
        SELECT round(sum(heap_blks_hit) * 100.0 / 
                    (sum(heap_blks_hit) + sum(heap_blks_read)), 2)
        FROM pg_statio_user_tables
    """)
    buffer_hit_ratio.set(cur.fetchone()[0])
    
    # Active connections
    cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
    active_connections.set(cur.fetchone()[0])
    
    # Query latency
    cur.execute("SELECT mean_exec_time FROM pg_stat_statements ORDER BY calls DESC LIMIT 1")
    query_latency.set(cur.fetchone()[0] or 0)
    
    cur.close()
    conn.close()

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        collect_metrics()
        time.sleep(15)
```

PostgreSQL's `pg_stat_activity` view provides real-time visibility into current database activity, including active queries, connection states, and waiting processes. This is invaluable for diagnosing sudden performance spikes or identifying long-running queries that consume resources.

```sql
-- PostgreSQL: Real-time activity monitoring
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    wait_event_type,
    wait_event,
    query,
    query_start,
    EXTRACT(EPOCH FROM (now() - query_start)) AS duration_seconds
FROM pg_stat_activity
WHERE state != 'idle'
  AND query NOT LIKE '%pg_stat_activity%'
ORDER BY query_start;

-- Find blocking queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement,
    blocked_activity.application_name AS blocked_application,
    blocking_activity.application_name AS blocking_application
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity 
    ON blocked_locks.pid = blocked_activity.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity 
    ON blocking_locks.pid = blocking_activity.pid
WHERE NOT blocked_locks.granted;
```

MySQL provides similar real-time monitoring through the `INFORMATION_SCHEMA` and performance schema tables. The performance schema, introduced in MySQL 5.5, offers detailed instrumentation of server execution.

```sql
-- MySQL: Current queries and blocking
SELECT 
    p.id AS process_id,
    p.user,
    p.host,
    p.db,
    p.command,
    p.time,
    p.state,
    t.TEXT AS query
FROM information_schema.PROCESSLIST p
JOIN performance_schema.threads t ON p.id = t.PROCESSLIST_ID
WHERE p.command != 'Sleep'
ORDER BY p.time DESC;

-- MySQL: Recent expensive queries
SELECT 
    SCHEMA_NAME,
    DIGEST_TEXT AS query,
    COUNT_STAR AS executions,
    SUM_TIMER_WAIT / 1000000000 AS total_time_ms,
    AVG_TIMER_WAIT / 1000000 AS avg_time_ms
FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 20;
```

---

## 2. Database Tuning Parameters

### PostgreSQL Configuration

PostgreSQL offers extensive configuration options that control memory usage, I/O behavior, query planning, and many other aspects of performance. Understanding these parameters and setting them appropriately for your workload is essential for achieving optimal performance.

**Memory-related parameters** have the most significant impact on PostgreSQL performance. The `shared_buffers` parameter controls the size of the shared memory buffer pool where frequently accessed data is cached. As a general rule, set this to 25% of available RAM for a dedicated database server, though higher values (up to 40%) can be beneficial for read-heavy workloads.

```sql
-- PostgreSQL: Recommended memory settings for 8GB RAM server
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
```

The `effective_cache_size` parameter informs the query planner about the effective size of the disk cache, which affects its decision between index scans and sequential scans. While this doesn't allocate memory, it helps the optimizer make better choices. Set this to the amount of RAM available for caching plus the size of `shared_buffers`.

```sql
-- PostgreSQL: Query planner settings
ALTER SYSTEM SET random_page_cost = 1.1;      -- For SSDs
ALTER SYSTEM SET effective_io_concurrency = 200;  -- Parallel I/O
ALTER SYSTEM SET default_statistics_target = 200;  -- Better planner stats
```

The `work_mem` parameter controls memory used for sorting and hash operations in individual queries. Higher values allow queries to perform larger sorts in memory rather than writing to disk, significantly improving performance for complex queries. However, this setting is per-operation, not global, so a value that's too high can cause memory issues with many concurrent queries.

```sql
-- PostgreSQL: Check current settings
SHOW ALL;

-- Check specific important settings
SELECT name, setting, unit, context
FROM pg_settings
WHERE name IN (
    'shared_buffers', 'effective_cache_size', 'work_mem',
    'maintenance_work_mem', 'max_connections', 'random_page_cost'
);
```

### MySQL InnoDB Configuration

MySQL's InnoDB storage engine has its own set of performance-critical parameters. Understanding these settings and tuning them appropriately is essential for getting the best performance from MySQL-based applications.

**InnoDB buffer pool** is the memory area where InnoDB caches table data and indexes. This is the most critical configuration for read performance. The buffer pool should be set to 70-80% of available memory on a dedicated MySQL server.

```ini
# my.cnf / my.ini: InnoDB settings
[mysqld]
innodb_buffer_pool_size = 8G
innodb_buffer_pool_instances = 8
innodb_log_file_size = 1G
innodb_log_buffer_size = 64M
innodb_flush_log_at_trx_commit = 1
innodb_flush_method = O_DIRECT
innodb_file_per_table = 1
innodb_io_capacity = 2000
innodb_io_capacity_max = 6000
innodb_read_io_threads = 8
innodb_write_io_threads = 8
```

The `innodb_flush_log_at_trx_commit` parameter controls the durability guarantees of transactions. Setting it to 1 (the default) provides full ACID durability but adds disk I/O overhead. Setting it to 2 provides durability at the OS level but may lose one second of transactions in case of system crash. Setting it to 0 provides the best performance but may lose transactions during MySQL crash.

```ini
# Performance vs durability tradeoff
# Maximum performance (may lose 1 second of transactions in crash)
innodb_flush_log_at_trx_commit = 0

# Balanced (may lose 1 second of transactions in OS crash)
innodb_flush_log_at_trx_commit = 2

# Full durability (default, safest)
innodb_flush_log_at_trx_commit = 1
```

The `innodb_log_file_size` parameter controls the size of the redo log files. Larger log files allow more transactions to be batched before checkpoints, improving write performance. However, recovery time increases with larger logs, so balance performance needs against recovery time requirements.

```sql
-- MySQL: Check InnoDB status and metrics
SHOW ENGINE INNODB STATUS;

-- Buffer pool metrics
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool%';

-- Check log file usage
SHOW GLOBAL STATUS LIKE 'Innodb_log%';
```

### Connection and Session Tuning

Connection management significantly impacts database performance. Each connection consumes memory and resources, so connection pooling and proper configuration help maximize throughput while maintaining stability.

```sql
-- PostgreSQL: Connection-related settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET superuser_reserved_connections = 3;
ALTER SYSTEM SET tcp_keepalives_idle = 60;
ALTER SYSTEM SET tcp_keepalives_interval = 10;
ALTER SYSTEM SET tcp_keepalives_count = 10;
```

The `max_connections` parameter controls the maximum number of concurrent connections. While higher values allow more simultaneous users, each connection consumes memory and adds overhead. Instead of increasing max_connections, use connection pooling to efficiently share connections among many application threads.

```ini
# MySQL: Connection settings
[mysqld]
max_connections = 200
max_connect_errors = 100000
wait_timeout = 600
interactive_timeout = 600
connect_timeout = 10
```

The `wait_timeout` and `interactive_timeout` parameters control how long MySQL waits before closing idle connections. Setting appropriate values prevents connection exhaustion from accumulated idle connections while allowing legitimate idle connections to remain open.

---

## 3. Connection Pooling and Management

### Why Connection Pooling Matters

Database connections are expensive to establish. Each new connection requires TCP handshake, authentication, privilege setup, and memory allocation. For applications that frequently open and close connections, this overhead can dominate actual query time. Connection pooling addresses this by maintaining a pool of pre-established connections that can be reused across requests.

The performance benefit of connection pooling is dramatic for applications with many short-lived database operations. Without pooling, an application making 1000 database calls per second might create 1000 connections per second, each requiring significant overhead. With pooling, those 1000 calls reuse a small number of established connections, dramatically reducing latency and server load.

```python
# Python: SQLAlchemy connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool, NullPool

# Production: Use QueuePool with appropriate sizing
engine = create_engine(
    "postgresql://user:password@localhost:5432/mydb",
    poolclass=QueuePool,
    pool_size=10,              # Base connections
    max_overflow=20,           # Additional connections under load
    pool_timeout=30,          # Seconds to wait for available connection
    pool_recycle=1800,         # Recycle connections after 30 minutes
    pool_pre_ping=True,        # Verify connection before using
    echo=False                 # Set True to see SQL output
)

# Testing: Use NullPool to disable pooling
test_engine = create_engine(
    "postgresql://user:password@localhost:5432/testdb",
    poolclass=NullPool
)
```

### PgBouncer Configuration

PgBouncer is a lightweight connection pooler for PostgreSQL that sits between application clients and the database server. It maintains a pool of database connections and assigns them to clients as needed, reducing the overhead of many concurrent connections.

```ini
# pgbouncer.ini: PgBouncer configuration
[databases]
mydb = host=postgres port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

# Connection pool settings
pool_mode = transaction        # transaction, session, or statement
default_pool_size = 25
min_pool_size = 10
reserve_pool_size = 5
reserve_pool_timeout = 5
max_client_conn = 1000

# Timeouts
server_idle_timeout = 60
server_lifetime = 3600
server_round_robin = 1

# Logging
log_connections = 0
log_disconnections = 0
log_pooler_errors = 1
```

The `pool_mode` parameter determines how PgBouncer manages connections. Transaction mode, the most common choice, assigns a connection for the duration of a single transaction. Session mode maintains a connection for the entire client session, while statement mode creates a new connection for each statement. Transaction mode provides the best resource utilization for most applications.

```ini
# Pool mode comparison
# Transaction mode (recommended for most applications)
pool_mode = transaction

# Use case: applications that use BEGIN/COMMIT explicitly
# Pros: Best resource utilization
# Cons: Cannot use session-level features like prepared statements, SET

# Session mode
pool_mode = session

# Use case: applications needing session-level features
# Pros: Full PostgreSQL session semantics
# Cons: Higher resource usage

# Statement mode
pool_mode = statement

# Use case: autocommit mode, simple queries
# Pros: Minimal connection usage
# Cons: No transactions, very limited use cases
```

### Application-Side Pooling

Many application frameworks include built-in connection pooling that removes the need for external poolers in simpler deployments. Understanding how to configure these pools is essential for optimal application performance.

```python
# Django: Database connection pooling
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydb',
        'USER': 'myuser',
        'PASSWORD': 'mypassword',
        'HOST': 'localhost',
        'PORT': '5432',
        'CONN_MAX_AGE': 600,        # Reuse connections for 10 minutes
        'CONN_HEALTH_CHECKS': True, # Django 4.1+: verify connections
        'OPTIONS': {
            'connect_timeout': 10,
        },
    }
}

# For high-traffic sites, use django-db-geventpool
# This provides true connection pooling with gevent for async
```

```python
# Node.js: pg connection pool
const { Pool } = require('pg');

const pool = new Pool({
    host: 'localhost',
    port: 5432,
    database: 'mydb',
    user: 'myuser',
    password: 'mypassword',
    max: 20,                    // Maximum connections
    idleTimeoutMillis: 30000,   // Close idle connections after 30s
    connectionTimeoutMillis: 2000,  // Connection timeout
    max: 20,                   // Total max connections
});

// Use pool.query() for automatic connection management
const result = await pool.query(
    'SELECT * FROM users WHERE id = $1', 
    [userId]
);

// Manual connection handling for transactions
const client = await pool.connect();
try {
    await client.query('BEGIN');
    await client.query('INSERT INTO orders ...');
    await client.query('COMMIT');
} catch (e) {
    await client.query('ROLLBACK');
    throw e;
} finally {
    client.release();  // Return to pool
}
```

---

## 4. Buffer Pool Optimization

### Understanding Buffer Pool Behavior

The buffer pool is the heart of database caching, storing data pages in memory to avoid disk I/O. Understanding how the buffer pool works and optimizing its usage is fundamental to achieving good database performance. When data is accessed, PostgreSQL and MySQL check if the page exists in the buffer pool. If found (a buffer hit), the data is returned immediately. If not found (a buffer miss), the page must be read from disk, a significantly slower operation.

Buffer pool efficiency is measured by the hit ratio, the percentage of page requests that are served from memory. A high hit ratio (above 99%) indicates effective caching, while lower ratios suggest either insufficient buffer pool size or poor access patterns that prevent effective caching.

```sql
-- PostgreSQL: Detailed buffer statistics
SELECT 
    schemaname,
    relname,
    heap_blks_read,
    heap_blks_hit,
    round(heap_blks_hit * 100.0 / 
          NULLIF(heap_blks_hit + heap_blks_read, 0), 2) AS hit_ratio,
    idx_blks_read,
    idx_blks_hit,
    round(idx_blks_hit * 100.0 / 
          NULLIF(idx_blks_hit + idx_blks_read, 0), 2) AS idx_hit_ratio
FROM pg_statio_user_tables
ORDER BY (heap_blks_hit + heap_blks_read) DESC
LIMIT 20;
```

The key to effective buffer pool usage is ensuring that frequently accessed data fits in memory. When the buffer pool is too small, frequently accessed pages get evicted before they can be reused, causing thrashing between memory and disk. When access patterns are random or sequential scans dominate, even a large buffer pool provides limited benefit.

### Optimizing Buffer Pool Size

The optimal buffer pool size depends on your data size, access patterns, and available memory. The goal is to cache the working set, the portion of data that is frequently accessed, while leaving enough memory for the operating system cache and other processes.

```sql
-- PostgreSQL: Memory calculation for dedicated server
-- Assuming 32GB RAM, dedicated database server

-- shared_buffers: 25-30% of RAM = 8GB
ALTER SYSTEM SET shared_buffers = '8GB';

-- effective_cache_size: RAM minus OS and shared_buffers
-- 32GB - 8GB (buffers) - 4GB (OS) = 20GB
ALTER SYSTEM SET effective_cache_size = '20GB';

-- work_mem: RAM for sorting operations
-- Allocate 64-256MB per concurrent sort operation
ALTER SYSTEM SET work_mem = '128MB';

-- maintenance_work_mem: For VACUUM, CREATE INDEX, etc.
ALTER SYSTEM SET maintenance_work_mem = '2GB';

-- wal_buffers: For WAL writes
ALTER SYSTEM SET wal_buffers = '64MB';
```

```ini
# MySQL: Buffer pool configuration for 32GB RAM
[mysqld]
innodb_buffer_pool_size = 24G    # 70-80% of RAM
innodb_buffer_pool_instances = 8  # Divide pool for concurrency
innodb_buffer_pool_chunk_size = 1G  # Chunk size for online resizing
```

The `innodb_buffer_pool_instances` parameter in MySQL divides the buffer pool into independent instances, reducing contention for large workloads. Each instance has its own mutexes, allowing concurrent operations without blocking. The ideal number depends on workload characteristics, but typically ranges from 4 to 16 instances.

```sql
-- MySQL: Monitor buffer pool effectiveness
-- Buffer pool hit ratio
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read%';

-- Pages in buffer pool
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_pages%';

-- Check for page eviction
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_bytes%';
```

### Managing Buffer Pool for Large Tables

Large tables present particular challenges for buffer pool management. Sequential scans can quickly fill the buffer pool with pages that may not be accessed again, evicting frequently accessed pages. Managing this behavior is essential for maintaining performance.

```sql
-- PostgreSQL: Control sequential scans
-- Increase cost of sequential scan to encourage index usage
ALTER SYSTEM SET seq_page_cost = 10.0;  -- Default is 1.0

-- Per-session: Disable sequential scan for specific queries
SET enable_seqscan = OFF;

-- MySQL: Control large scans
-- Set per-session to prevent large scans
SET SESSION optimizer_switch = 'index_condition_pushdown=on';

-- Configure to limit large scans
SET SESSION max_execution_time = 10000;  # 10 seconds max
```

The key insight is that not all data has equal access frequency. In most applications, a small subset of data is accessed frequently (the hot set), while the majority of data is accessed rarely. Optimizing for the hot set maximizes buffer pool efficiency and overall performance.

---

## 5. Query Optimization Patterns

### Efficient Query Patterns

Writing efficient queries is the foundation of database performance. Even with perfect configuration, poorly written queries can cause significant performance problems. Understanding common patterns and their alternatives helps you write queries that perform well.

**Prefer indexed lookups over full scans:** Ensure WHERE clauses use indexed columns with appropriate operators. Range scans on indexed columns can use indexes efficiently, but functions or type conversions on indexed columns prevent index usage.

```sql
-- Efficient: Uses index on (status, created_at)
SELECT * FROM orders 
WHERE status = 'completed' 
  AND created_at > '2024-01-01';

-- Less efficient: Type conversion prevents index usage
SELECT * FROM orders 
WHERE status = 'completed' 
  AND created_at::date = '2024-01-01';

-- Efficient: Use parameters for date range
SELECT * FROM orders 
WHERE status = $1 
  AND created_at BETWEEN $2 AND $3;
```

**Batch operations instead of row-by-row:** Database operations are significantly faster when processing multiple rows in a single statement. This reduces round trips, allows the database to optimize execution, and reduces transaction overhead.

```sql
-- Row-by-row (slow)
INSERT INTO order_items (order_id, product_id, quantity)
VALUES (1, 101, 1);
INSERT INTO order_items (order_id, product_id, quantity)
VALUES (1, 102, 2);
INSERT INTO order_items (order_id, product_id, quantity)
VALUES (1, 103, 1);

-- Batch insert (fast)
INSERT INTO order_items (order_id, product_id, quantity)
VALUES 
    (1, 101, 1),
    (1, 102, 2),
    (1, 103, 1);

-- For large batches, use ON CONFLICT for upsert
INSERT INTO order_items (order_id, product_id, quantity)
VALUES (1, 101, 1), (1, 102, 2)
ON CONFLICT (order_id, product_id) 
DO UPDATE SET quantity = EXCLUDED.quantity;
```

**Use appropriate join types:** Different join algorithms have different performance characteristics. Nested loop joins are efficient when one table is small or has an index on the join column. Hash joins are efficient for large tables without indexes. Merge joins require sorted inputs.

```sql
-- Efficient: Small dimension table lookup via nested loop
SELECT o.*, c.name, c.email
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id
WHERE o.id = 12345;

-- For large table joins, consider hash or merge
-- Hash join: Both tables large, no useful index
SELECT o.*, p.*
FROM orders o
INNER JOIN order_items oi ON o.id = oi.order_id
INNER JOIN products p ON oi.product_id = p.id
WHERE o.created_at > '2024-01-01';

-- Materialize common table expression for reuse
WITH recent_orders AS (
    SELECT * FROM orders 
    WHERE created_at > '2024-01-01'
)
SELECT ro.*, c.name, p.name
FROM recent_orders ro
JOIN customers c ON ro.customer_id = c.id
JOIN order_items oi ON ro.id = oi.order_id
JOIN products p ON oi.product_id = p.id;
```

### Write Optimization Patterns

Write-heavy workloads require their own optimization strategies. Reducing write amplification, batching operations, and using appropriate isolation levels all contribute to write performance.

```sql
-- PostgreSQL: Bulk upsert pattern
INSERT INTO users (id, email, name, updated_at)
VALUES 
    (1, 'user1@example.com', 'User 1', NOW()),
    (2, 'user2@example.com', 'User 2', NOW()),
    (3, 'user3@example.com', 'User 3', NOW())
ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email,
    name = EXCLUDED.name,
    updated_at = EXCLUDED.updated_at;

-- MySQL: Bulk insert with ON DUPLICATE KEY
INSERT INTO users (id, email, name)
VALUES 
    (1, 'user1@example.com', 'User 1'),
    (2, 'user2@example.com', 'User 2'),
    (3, 'user3@example.com', 'User 3')
ON DUPLICATE KEY UPDATE
    email = VALUES(email),
    name = VALUES(name);

-- PostgreSQL: Partitioned writes for time-series data
INSERT INTO orders_2024 PARTITION OF orders
    (id, customer_id, total_amount, status, created_at)
VALUES 
    (DEFAULT, 1, 100.00, 'pending', NOW()),
    (DEFAULT, 2, 250.00, 'pending', NOW());
```

For very high write throughput, consider using batch processing and asynchronous writes. Many applications can tolerate slight delays in write confirmation, allowing you to batch multiple writes together and reduce per-operation overhead.

```python
# Python: Batch write optimization
import asyncio
from asyncpg import create_pool

async def batch_insert_orders(orders):
    """Insert orders in batches for better throughput"""
    BATCH_SIZE = 100
    
    async with create_pool("postgresql://user:pass@localhost/mydb") as pool:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for i in range(0, len(orders), BATCH_SIZE):
                    batch = orders[i:i + BATCH_SIZE]
                    await conn.executemany(
                        """
                        INSERT INTO orders (customer_id, total, status)
                        VALUES ($1, $2, $3)
                        """,
                        [(o['customer_id'], o['total'], o['status']) 
                         for o in batch]
                    )
```

---

## 6. Production Best Practices

### Capacity Planning

Planning for capacity ensures your database can handle growth without performance degradation. Capacity planning involves analyzing current usage patterns, projecting future growth, and planning infrastructure changes accordingly.

```python
# Capacity planning calculation
def calculate_database_capacity(
    daily_write_gb: float,
    retention_days: int,
    growth_rate: float,
    read_to_write_ratio: float = 10.0
) -> dict:
    """Calculate storage and performance requirements"""
    
    # Base storage for data
    base_storage = daily_write_gb * retention_days
    
    # Index overhead (typically 30-50% for write-heavy workloads)
    index_overhead = base_storage * 0.4
    
    # Write amplification factor
    # PostgreSQL: WAL + table btree typically 2-3x
    write_amplification = 2.5
    total_write_volume = daily_write_gb * write_amplification
    
    # Memory requirements based on working set
    # Assume 10% of data is frequently accessed
    working_set_gb = (base_storage + index_overhead) * 0.10
    
    # Add 30% buffer for growth during planning period
    planning_months = 12
    storage_buffer = (1 + growth_rate) ** planning_months
    
    return {
        'base_storage_tb': round(base_storage / 1000, 2),
        'index_storage_tb': round(index_overhead / 1000, 2),
        'total_storage_tb': round(
            (base_storage + index_overhead) * storage_buffer / 1000, 2
        ),
        'recommended_buffer_pool_gb': round(working_set_gb * 1.5, 0),
        'estimated_peak_iops': int(total_write_volume * 1000),
    }
```

### Performance Regression Prevention

Preventing performance regressions requires testing and monitoring in production-like environments. Establish performance baselines and alert on significant deviations.

```python
# Performance regression testing
import pytest
import time
from database import get_engine

class TestDatabasePerformance:
    @pytest.fixture
    def engine(self):
        return get_engine()
    
    @pytest.mark.parametrize("query,threshold_ms", [
        ("SELECT * FROM users WHERE id = 1", 10),
        ("SELECT * FROM orders WHERE status = 'pending'", 100),
        ("SELECT COUNT(*) FROM orders", 50),
    ])
    def test_query_performance(self, engine, query, threshold_ms):
        """Verify queries meet performance thresholds"""
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with engine.connect() as conn:
                conn.execute(text(query))
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        assert avg_time < threshold_ms, (
            f"Query exceeded threshold: {avg_time:.1f}ms > {threshold_ms}ms"
        )
```

### Troubleshooting Common Issues

Even with proper tuning, issues can arise in production. Knowing how to diagnose and resolve common problems quickly minimizes impact on users.

```sql
-- PostgreSQL: Diagnose slow queries
-- 1. Check currently running queries
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- 2. Find most frequently called slow queries
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
WHERE mean_time > 100
ORDER BY calls * mean_time DESC
LIMIT 10;

-- 3. Check for locks causing delays
SELECT * FROM pg_locks WHERE NOT granted;

-- 4. Check for missing indexes
SELECT schemaname, tablename, idx_scan, seq_scan
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan * 10;

-- 5. Check for bloat
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_stat_user_tables
WHERE (n_dead_tup * 100) / (n_live_tup + n_dead_tup) > 10;
```

```sql
-- MySQL: Diagnose slow queries
-- 1. Enable slow query log (if not already)
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';
SET GLOBAL long_query_time = 1;

-- 2. Analyze EXPLAIN output
EXPLAIN SELECT * FROM orders WHERE customer_id = 123;

-- 3. Check for table locks
SHOW ENGINE INNODB STATUS;

-- 4. Check open tables
SHOW OPEN TABLES WHERE In_use > 0;

-- 5. Check thread status
SHOW PROCESSLIST;
```

---

## Summary

Database performance tuning is both a science and an art. The science involves understanding how databases manage memory, process queries, and handle concurrent operations. The art involves applying this knowledge to your specific workload characteristics and business requirements.

The key principles to remember are: measure before making changes, tune incrementally and verify improvements, monitor continuously in production, and plan for growth. Configuration changes that work well for one workload may be inappropriate for another, so always validate changes in your specific environment.

Start with the fundamentals: ensure appropriate memory allocation for buffer pools, use connection pooling to reduce connection overhead, create targeted indexes for your query patterns, and write efficient queries that the database can optimize effectively. As your system grows and requirements evolve, deeper tuning may become necessary, but these fundamentals provide a solid foundation for most applications.

For advanced topics like distributed database tuning, refer to the research documentation and specialized guides. Remember that the best-tuned database cannot compensate for poor application architecture or inefficient queries, so always consider the full stack when diagnosing and resolving performance issues.
