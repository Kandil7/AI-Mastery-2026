# Database Performance Tuning

## Introduction

Database performance tuning encompasses the systematic process of optimizing database systems to achieve maximum efficiency, throughput, and responsiveness. This guide provides comprehensive coverage of performance tuning techniques for PostgreSQL and MySQL, the two most widely used open-source relational databases. Understanding how to tune these database systems is essential for building scalable applications that can handle increasing workloads without degradation.

Performance tuning involves multiple interconnected components, from low-level configuration parameters that control memory allocation and I/O operations, to higher-level architectural decisions about indexing strategies and query patterns. Each aspect of tuning affects the others, making a holistic approach critical. A database configured with optimal memory settings will still perform poorly if queries lack appropriate indexes, and well-designed indexes cannot compensate for misconfigured buffer pools.

This guide covers performance metrics and monitoring approaches to identify bottlenecks, detailed tuning parameters for both PostgreSQL and MySQL with practical examples, connection pooling strategies that reduce overhead, buffer pool optimization techniques, and common query optimization patterns. By following these guidelines and understanding the tradeoffs involved, you can significantly improve your database's performance and ensure it meets the demands of production workloads.

The techniques presented here are based on established best practices and real-world experience tuning databases for high-traffic applications. However, every application has unique characteristics, so use these recommendations as starting points and adjust based on actual performance measurements. The ultimate authority on whether a tuning change is beneficial is always benchmark testing in your specific environment.

---

## 1. Performance Metrics and Monitoring

### Key Performance Indicators

Understanding which metrics matter is fundamental to effective performance tuning. Database systems generate大量 metrics, but focusing on key performance indicators (KPIs) helps you quickly identify issues and track improvements over time. The most important metrics fall into several categories: throughput, latency, resource utilization, and efficiency.

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

**Prefer indexed lookups over full table scans**: Always ensure that WHERE clauses use indexed columns when possible. Indexes provide O(log n) lookup instead of O(n) full scans.

```sql
-- Bad: Full table scan
SELECT * FROM users WHERE email LIKE '%@gmail.com';

-- Good: Indexed lookup
SELECT * FROM users WHERE email = 'user@gmail.com';

-- Better: Partial index for common patterns
CREATE INDEX idx_users_gmail ON users(email) WHERE email LIKE '%@gmail.com';
```

**Use covering indexes**: Include all columns needed by the query in the index to avoid table lookups.

```sql
-- Covering index example
CREATE INDEX idx_orders_customer_date_total ON orders(customer_id, order_date DESC, total_amount);

-- Query can be satisfied entirely from index
SELECT customer_id, order_date, total_amount
FROM orders
WHERE customer_id = 123
ORDER BY order_date DESC
LIMIT 10;
```

**Avoid SELECT ***: Select only the columns you need to reduce network transfer and memory usage.

```sql
-- Bad
SELECT * FROM orders WHERE customer_id = 123;

-- Good
SELECT order_id, order_date, total_amount
FROM orders
WHERE customer_id = 123;
```

**Use parameterized queries**: Prevent SQL injection and allow query plan caching.

```python
# Bad - SQL injection vulnerable
query = f"SELECT * FROM users WHERE username = '{user_input}'"

# Good - Parameterized query
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, [user_input])
```

### Common Query Anti-Patterns

Certain query patterns consistently cause performance problems and should be avoided:

**N+1 queries**: Loading related data in separate queries instead of using JOINs.

```python
# Bad: N+1 pattern
users = db.query("SELECT * FROM users")
for user in users:
    orders = db.query("SELECT * FROM orders WHERE user_id = %s", [user.id])
    # Process orders

# Good: Single query with JOIN
users_with_orders = db.query("""
    SELECT u.*, o.order_id, o.total_amount
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.created_at > %s
""", [start_date])
```

**Cartesian products**: Missing JOIN conditions that create massive result sets.

```sql
-- Bad: Missing JOIN condition
SELECT * FROM users, orders WHERE users.created_at > '2024-01-01';

-- Good: Proper JOIN
SELECT * FROM users u
JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01';
```

**Functions in WHERE clauses**: Applying functions to columns prevents index usage.

```sql
-- Bad: Function on column prevents index usage
SELECT * FROM users WHERE LOWER(email) = 'user@gmail.com';

-- Good: Store normalized data or use functional indexes
SELECT * FROM users WHERE email = 'user@gmail.com';

-- Or create functional index
CREATE INDEX idx_users_lower_email ON users(LOWER(email));
```

### Advanced Query Optimization

For complex queries, additional techniques can significantly improve performance:

**Materialized views**: Pre-compute expensive aggregations for fast reads.

```sql
-- Create materialized view for dashboard
CREATE MATERIALIZED VIEW daily_sales_summary AS
SELECT
    DATE(order_date) as sale_date,
    COUNT(*) as total_orders,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_order_value
FROM orders
GROUP BY DATE(order_date);

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_sales_summary;
```

**Partitioning**: Split large tables into smaller, more manageable pieces.

```sql
-- PostgreSQL partitioning by date
CREATE TABLE sales (
    sale_id BIGSERIAL,
    sale_date DATE NOT NULL,
    amount DECIMAL(10,2),
    product_id INT
) PARTITION BY RANGE (sale_date);

-- Monthly partitions
CREATE TABLE sales_2024_01 PARTITION OF sales
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE sales_2024_02 PARTITION OF sales
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
```

**Query hints**: Force specific execution plans when the optimizer makes poor choices.

```sql
-- PostgreSQL: Use index hint
SELECT * FROM orders
WHERE customer_id = 123
AND order_date > '2024-01-01'
ORDER BY order_date DESC
/*+ IndexScan(orders idx_orders_customer_date) */;

-- MySQL: Force index usage
SELECT * FROM orders FORCE INDEX (idx_orders_customer_date)
WHERE customer_id = 123
AND order_date > '2024-01-01'
ORDER BY order_date DESC;
```

---

## Related Resources

- For database fundamentals, see [Database Fundamentals](./database_fundamentals.md)
- For indexing strategies, see [Indexing Fundamentals](./indexing_fundamentals.md)
- For query processing basics, see [Query Processing](./query_processing.md)
- For concurrency control, see [Concurrency Control](./concurrency_control.md)
- For scaling strategies, see [Scaling Strategies](./scaling_strategies.md)