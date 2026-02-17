# Database Troubleshooting and Debugging

## Overview

Database issues can bring down entire applications, and diagnosing them requires a systematic approach. This guide provides comprehensive techniques for identifying, analyzing, and resolving common database problems that AI/ML engineers and software developers encounter in production environments. The ability to troubleshoot database issues effectively is a critical skill that separates junior developers from senior engineers.

When database problems arise, the stakes are often high—users experience slow responses or complete failures, and the pressure to resolve issues quickly can lead to hasty diagnoses. This document teaches you to approach database debugging methodically, using the right tools and techniques to identify root causes rather than symptoms.

Understanding how to diagnose database issues requires knowledge across multiple domains: query execution, system resources, concurrency controls, and network communication. This guide covers each of these areas in depth, providing practical examples and real-world scenarios that prepare you for common production situations.

The examples in this guide use PostgreSQL as the primary database system because of its popularity and rich diagnostic features, but the concepts apply broadly to other relational databases. Where significant differences exist with other systems like MySQL, SQL Server, or MongoDB, those differences are explicitly noted.

## Common Database Performance Problems

### Identifying Slow Queries

The first step in database troubleshooting is identifying which queries are causing performance issues. Slow queries often hide in application code, executing hundreds or thousands of times without explicit attention. PostgreSQL's `pg_stat_statements` extension provides crucial visibility:

```sql
-- Enable the extension (run once as superuser)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View the slowest queries by total execution time
SELECT 
    substring(query, 1, 100) AS query_preview,
    calls,
    mean_exec_time,
    total_exec_time,
    rows,
    (mean_exec_time * calls / 1000)::integer AS total_seconds
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

This query reveals which statements consume the most total time. A query with a moderate execution time but high call count often indicates a bigger problem than a single slow query. Pay attention to the ratio between mean execution time and total time—a large gap suggests occasional outliers that might indicate data skew or parameter sniffing issues.

For MySQL, enable the slow query log to capture problematic queries:

```sql
-- Enable slow query log
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow-queries.log';
SET GLOBAL long_query_time = 1;  -- Log queries taking > 1 second
SET GLOBAL log_queries_not_using_indexes = 'ON';
```

### Query Analysis and Execution Plans

Understanding how your database executes a query is essential for troubleshooting performance issues. The `EXPLAIN` command shows the query plan without actually executing the query, while `EXPLAIN ANALYZE` executes the query and provides actual runtime statistics:

```sql
-- Basic execution plan analysis
EXPLAIN ANALYZE
SELECT o.id, o.created_at, c.name, SUM(oi.quantity * oi.price) AS total
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN order_items oi ON o.id = oi.order_id
WHERE o.created_at > NOW() - INTERVAL '30 days'
GROUP BY o.id, o.created_at, c.name
ORDER BY total DESC
LIMIT 100;
```

When analyzing execution plans, look for these warning signs:

**Sequential Scans**: Table scans indicate missing indexes or queries that return large portions of the table. For queries that should be selective, sequential scans suggest the optimizer chose this path because indexes wouldn't help or would be more expensive:

```
Seq Scan on orders  (cost=0.00..15420.00 rows=500000 width=40)
Filter: (created_at > '2024-01-01'::date)
```

**Nested Loop Joins**: These are efficient when the inner table has an index on the join key but become extremely slow without proper indexing. Watch for nested loops with high actual rows values:

```
Nested Loop  (cost=4.30..2847.50 rows=100 width=32)
  ->  Index Scan using idx_orders_customer on orders
  ->  Index Scan using idx_customer_id on customers
```

**High Cardinality Estimates**: If the estimated number of rows differs significantly from actual rows, statistics are stale or the query planner doesn't have enough information. This often leads to poor plan choices:

```
->  Bitmap Heap Scan on products  (cost=208.50..11420.00 rows=1 width=32)
    Recheck Cond: (category_id = 10)
    ->  Bitmap Index Scan on idx_products_category  (cost=0.00..208.50 rows=1 width=0)
```

**Missing Indexes**: The query planner explicitly mentions when indexes would help. Look for "Seq Scan on" followed by "Filter" conditions on columns that could benefit from indexing.

### Index Usage Analysis

Indexes dramatically improve query performance when used correctly, but they add overhead for write operations and consume storage. Analyze index usage patterns:

```sql
-- Find unused indexes (PostgreSQL)
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- Find indexes with high write overhead
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_tup_insert + idx_tup_update + idx_tup_delete AS total_writes
FROM pg_stat_user_indexes
ORDER BY total_writes DESC;
```

This analysis reveals two important patterns. Unused indexes consume storage and slow down INSERT, UPDATE, and DELETE operations without providing benefits—these should be dropped. Indexes with high write counts but low read counts might be candidates for removal if the table has high write throughput.

For MySQL, analyze index usage with:

```sql
-- MySQL index statistics
SELECT 
    TABLE_NAME,
    INDEX_NAME,
    NON_CARDINALITY,
    SEQ_IN_INDEX,
    COLUMN_NAME,
    CARDINALITY
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = 'your_database'
ORDER BY_NAME, SEQ_IN_INDEX;
 TABLE_NAME, INDEX```

## Locking and Concurrency Issues

### Understanding Lock Types

Database systems use various lock types to ensure data consistency during concurrent access. Understanding these locks helps diagnose blocking and deadlock issues. PostgreSQL provides several lock modes:

| Lock Mode | Purpose | Conflict Modes |
|-----------|---------|-----------------|
| ACCESS SHARE | Read operations | ACCESS EXCLUSIVE |
| ROW SHARE | SELECT FOR UPDATE | EXCLUSIVE, ACCESS EXCLUSIVE |
| ROW EXCLUSIVE | INSERT, UPDATE, DELETE | SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE | CREATE INDEX | ROW EXCLUSIVE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| SHARE ROW EXCLUSIVE | | ROW EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| EXCLUSIVE | | ROW SHARE, ROW EXCLUSIVE, SHARE, SHARE ROW EXCLUSIVE, EXCLUSIVE, ACCESS EXCLUSIVE |
| ACCESS EXCLUSIVE | ALTER TABLE, DROP TABLE | All |

### Diagnosing Blocking Queries

When transactions block each other, identify the blocking and blocked queries:

```sql
-- PostgreSQL: Find blocking queries
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocking_locks.pid AS blocking_pid,
    blocked_activity.usename AS blocked_user,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_query,
    blocking_activity.query AS blocking_query,
    blocked_activity.application_name AS blocked_application,
    blocking_activity.application_name AS blocking_application,
    blocked_activity.wait_event_type || ':' || blocked_activity.wait_event AS blocked_wait,
    blocking_activity.wait_event_type || ':' || blocking_activity.wait_event AS blocking_wait
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocked_locks.lock = blocking_locks.lock
    AND blocked_locks.transactionid = blocking_locks.transactionid
    AND blocked_locks.pid != blocking_locks.pid
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_locks.pid = blocked_activity.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_locks.pid = blocking_activity.pid
WHERE NOT blocked_locks.granted;
```

For SQL Server, use the `sys.dm_exec_requests` view:

```sql
-- SQL Server: Find blocking queries
SELECT 
    blocked.session_id AS blocked_session_id,
    blocked.status AS blocked_status,
    blocked.cpu_time AS blocked_cpu_time,
    blocked.reads AS blocked_reads,
    blocked.writes AS blocked_writes,
    blocked.total_elapsed_time AS blocked_elapsed_time,
    blocked_inner.text AS blocked_query,
    blocker.session_id AS blocking_session_id,
    blocker.status AS blocking_status,
    blocker.cpu_time AS blocking_cpu_time,
    blocker.text AS blocking_query
FROM sys.dm_exec_requests blocked
CROSS APPLY sys.dm_exec_sql_text(blocked.sql_handle) blocked_inner
JOIN sys.dm_exec_requests blocker ON blocked.blocking_session_id = blocker.session_id
OUTER APPLY sys.dm_exec_sql_text(blocker.sql_handle) blocker;
```

### Deadlock Detection and Resolution

Deadlocks occur when two or more transactions hold locks that the other needs, creating a circular dependency. Modern databases detect deadlocks automatically and resolve them by rolling back one transaction, but this rollback can cause application errors:

```sql
-- PostgreSQL: Check for recent deadlocks
SELECT 
    pid,
    now() - pg_postmaster_start_time() AS uptime,
    query,
    state,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE datname = current_database()
AND state = 'idle in transaction'
AND query NOT ILIKE '%pg_stat_activity%'
AND now() - query_start > interval '10 seconds';
```

Prevent deadlocks through careful application design:

```python
import asyncio
from contextlib import asynccontextmanager

class DeadlockPreventer:
    """
    Prevent deadlocks by enforcing consistent lock ordering
    across transactions.
    """
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.lock_order = ["users", "accounts", "transactions", "orders"]
    
    @asynccontextmanager
    async def transaction(self):
        """Transaction with automatic lock ordering"""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # All locks acquired in consistent order
                # prevents deadlocks in multi-table operations
                yield conn

# Application code that could deadlock without proper ordering
async def transfer_funds(db_pool, from_account: int, to_account: int, amount: Decimal):
    """Transfer funds between accounts with deadlock prevention"""
    async with DeadlockPreventer(db_pool).transaction() as conn:
        # Always lock accounts in numeric order
        first_account = min(from_account, to_account)
        second_account = max(from_account, to_account)
        
        # Lock in consistent order
        await conn.execute(
            "SELECT * FROM accounts WHERE id = $1 FOR UPDATE", 
            first_account
        )
        await conn.execute(
            "SELECT * FROM accounts WHERE id = $1 FOR UPDATE",
            second_account
        )
        
        # Perform transfer
        await conn.execute(
            "UPDATE accounts SET balance = balance - $1 WHERE id = $2",
            amount, from_account
        )
        await conn.execute(
            "UPDATE accounts SET balance = balance + $1 WHERE id = $2", 
            amount, to_account
        )
```

## Memory and CPU Troubleshooting

### Database Memory Configuration

Memory is typically the most critical resource for database performance. Incorrect configuration leads to either memory exhaustion or poor cache utilization. PostgreSQL's shared buffer configuration is the most important setting:

```sql
-- Check current memory configuration
SHOW shared_buffers;
SHOW effective_cache_size;
SHOW work_mem;
SHOW maintenance_work_mem;

-- Calculate recommended shared_buffers
-- Rule of thumb: 25% of available RAM for dedicated database server
-- For example, with 16GB RAM: shared_buffers = 4GB
```

For a PostgreSQL server with 16GB RAM dedicated to the database:

```ini
# postgresql.conf
shared_buffers = 4GB              # 25% of RAM
effective_cache_size = 12GB        # 75% of RAM (combined with OS cache)
work_mem = 64MB                   # Per sort operation, adjust based on queries
maintenance_work_mem = 1GB         # For VACUUM, CREATE INDEX, etc.
```

Monitor memory usage in production:

```sql
-- PostgreSQL: Check buffer cache hit ratio
SELECT 
    sum(heap_blks_read) AS heap_read,
    sum(heap_blks_hit) AS heap_hit,
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) AS ratio
FROM pg_statio_user_tables;

-- Check index cache hit ratio  
SELECT 
    sum(idx_blks_read) AS idx_read,
    sum(idx_blks_hit) AS idx_hit,
    sum(idx_blks_hit) / (sum(idx_blks_hit) + sum(idx_blks_read)) AS ratio
FROM pg_statio_user_indexes;
```

A cache hit ratio below 0.90 indicates insufficient memory for the working set. This often means either increasing `shared_buffers` or optimizing queries to work with data that fits in memory.

### CPU Performance Issues

High CPU usage in databases typically stems from CPU-intensive queries, excessive connections, or background maintenance. Identify CPU-intensive queries:

```sql
-- PostgreSQL: Find queries with highest CPU usage
SELECT 
    substring(query, 1, 80) AS query,
    calls,
    total_exec_time,
    mean_exec_time,
    stddev_exec_time,
    rows,
    shared_blks_hit,
    shared_blks_read,
    local_blks_hit,
    local_blks_read,
    temp_blks_read,
    temp_blks_written
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

For MySQL, enable the performance schema:

```sql
-- MySQL: Find CPU-intensive queries
SELECT 
    digest_text,
    count_star AS executions,
    sum_timer_wait / 1000000000 AS total_seconds,
    avg_timer_wait / 1000000 AS avg_ms,
    sum_cpu_time / 1000000000 AS cpu_seconds
FROM performance_schema.events_statements_summary_by_digest
ORDER BY cpu_seconds DESC
LIMIT 20;
```

## Network and Connection Issues

### Diagnosing Connection Problems

Connection issues manifest as timeouts, connection refusals, or erratic behavior. Systematic diagnosis requires checking multiple layers:

```python
import socket
import asyncio
from typing import Optional

async def diagnose_connection(
    host: str, 
    port: int, 
    timeout: float = 5.0
) -> dict:
    """Diagnose database connection issues"""
    results = {
        "host": host,
        "port": port,
        "dns_resolved": False,
        "port_open": False,
        "ssl_established": False,
        "errors": []
    }
    
    # Check DNS resolution
    try:
        ip = socket.gethostbyname(host)
        results["dns_resolved"] = True
        results["resolved_ip"] = ip
    except socket.gaierror as e:
        results["errors"].append(f"DNS resolution failed: {e}")
        return results
    
    # Check if port is open
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )
        results["port_open"] = True
        results["ssl_established"] = True if writer.get_extra_info('ssl_context') else False
        writer.close()
        await writer.wait_closed()
    except asyncio.TimeoutError:
        results["errors"].append(f"Connection timed out after {timeout}s")
    except ConnectionRefusedError:
        results["errors"].append("Connection refused - is the database running?")
    except Exception as e:
        results["errors"].append(f"Connection failed: {e}")
    
    return results

# Usage
async def test_db_connectivity():
    issues = await diagnose_connection("db.example.com", 5432)
    if issues["errors"]:
        print("Connection issues detected:")
        for error in issues["errors"]:
            print(f"  - {error}")
```

### Connection Pool Exhaustion

Connection pool exhaustion is a common production problem that causes "too many connections" errors. Monitor pool metrics and implement protection:

```python
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Deque

@dataclass
class ConnectionEvent:
    timestamp: datetime
    action: str  # 'acquired', 'released', 'failed'
    duration_ms: float = 0

class ConnectionPoolMonitor:
    def __init__(self, max_pool_size: int, warning_threshold: float = 0.8):
        self.max_pool_size = max_pool_size
        self.warning_threshold = warning_threshold
        self.events: Deque[ConnectionEvent] = Deque(maxlen=1000)
        self.active_connections = 0
        self.peak_connections = 0
    
    def record_acquisition(self, duration_ms: float = 0):
        self.active_connections += 1
        self.peak_connections = max(self.peak_connections, self.active_connections)
        self.events.append(ConnectionEvent(
            datetime.utcnow(), "acquired", duration_ms
        ))
        
        if self.active_connections >= self.max_pool_size * self.warning_threshold:
            self._alert_high_usage()
    
    def record_release(self, duration_ms: float = 0):
        self.active_connections = max(0, self.active_connections - 1)
        self.events.append(ConnectionEvent(
            datetime.utcnow(), "released", duration_ms
        ))
    
    def record_failure(self):
        self.events.append(ConnectionEvent(
            datetime.utcnow(), "failed"
        ))
    
    def _alert_high_usage(self):
        # Integration with alerting system
        print(f"ALERT: Connection pool at {self.active_connections}/{self.max_pool_size}")
    
    def get_stats(self) -> dict:
        recent_failures = sum(
            1 for e in self.events 
            if e.action == "failed" 
            and e.timestamp > datetime.utcnow() - timedelta(minutes=5)
        )
        
        return {
            "current_active": self.active_connections,
            "peak_connections": self.peak_connections,
            "max_pool_size": self.max_pool_size,
            "utilization_pct": self.active_connections / self.max_pool_size,
            "recent_failures_5min": recent_failures
        }
```

## Data Corruption Recovery

### Detecting Data Corruption

Data corruption can occur due to hardware failures, software bugs, or improper shutdowns. Early detection prevents further damage:

```sql
-- PostgreSQL: Check for corruption
SELECT 
    schemaname,
    relname,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_autovacuum,
    last_analyze
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;

-- Check for table bloat (indicates maintenance needed)
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    CASE 
        WHEN pg_total_relation_size(schemaname||'.'||tablename) > 0
        THEN round(
            100.0 * pg_relation_size(schemaname||'.'||tablename) / 
            pg_total_relation_size(schemaname||'.'||tablename), 2
        )
        ELSE 100
    END AS table_pct
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

For MySQL, check tables for errors:

```sql
-- MySQL: Check tables for corruption
CHECK TABLE table_name EXTENDED;

-- Repair if needed
REPAIR TABLE table_name;
```

### Recovery Strategies

When corruption occurs, have a recovery plan:

```python
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

class DatabaseRecoveryManager:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.backup_dir = Path(db_config.get("backup_dir", "/backups"))
    
    async def create_emergency_backup(self, table_name: str) -> Path:
        """Create emergency backup before recovery attempt"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{table_name}_{timestamp}.sql"
        
        # Export current table data
        process = await asyncio.create_subprocess_exec(
            "pg_dump",
            "--data-only",
            "--table=" + table_name,
            "-f", str(backup_file),
            self.db_config["database"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Backup failed: {process.stderr}")
        
        return backup_file
    
    async def vacuum_full_recovery(self, table_name: str):
        """
        VACUUM FULL reclaims space and rebuilds the table.
        This requires exclusive lock and can be slow for large tables.
        """
        import psycopg2
        
        conn = psycopg2.connect(self.db_config)
        try:
            # Create backup first
            await self.create_emergency_backup(table_name)
            
            # Perform VACUUM FULL
            with conn.cursor() as cur:
                cur.execute(f"VACUUM FULL {table_name}")
                conn.commit()
                
            return {"status": "success", "table": table_name}
        finally:
            conn.close()
    
    async def reindex_recovery(self, table_name: str):
        """
        Rebuild indexes to fix index corruption
        """
        import psycopg2
        
        conn = psycopg2.connect(self.db_config)
        try:
            with conn.cursor() as cur:
                # Get all indexes on the table
                cur.execute("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE tablename = %s
                """, (table_name,))
                
                indexes = [row[0] for row in cur.fetchall()]
                
                # Rebuild each index
                for idx in indexes:
                    cur.execute(f"REINDEX INDEX {idx}")
                
                conn.commit()
                
            return {"status": "success", "indexes_rebuilt": len(indexes)}
        finally:
            conn.close()
```

## Systematic Troubleshooting Methodology

### The DRIP Framework

When troubleshooting database issues, follow a systematic approach:

**D - Define**: Clearly state the problem. "The application is slow" is not actionable. "Orders page takes 8 seconds to load" provides a measurable target.

**R - Reproduce**: Can you reliably reproduce the issue? If it's intermittent, gather more data. If it's reproducible, you can test fixes.

**I - Isolate**: Use divide and conquer. Disable features, simplify queries, reduce data volume. Each test should isolate a variable.

**P - Plan**: Based on evidence gathered, create a hypothesis. "The query is doing a sequential scan because there's no index on the status column."

### Monitoring and Observability

Implement comprehensive monitoring to catch issues before they become emergencies:

```python
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class DatabaseMetrics:
    timestamp: datetime
    connections_active: int
    connections_idle: int
    connections_waiting: int
    cache_hit_ratio: float
    transactions_per_second: float
    queries_per_second: float
    avg_query_duration_ms: float
    slow_queries_count: int

class DatabaseMonitor:
    def __init__(self, db_pool, alert_threshold_ms=1000):
        self.db_pool = db_pool
        self.alert_threshold_ms = alert_threshold_ms
        self.samples = []
    
    async def collect_metrics(self) -> DatabaseMetrics:
        async with self.db_pool.acquire() as conn:
            # Connection stats
            conn_stats = await conn.fetchrow("""
                SELECT 
                    count(*) FILTER (WHERE state = 'active') AS active,
                    count(*) FILTER (WHERE state = 'idle') AS idle,
                    count(*) FILTER (WHERE state = 'idle in transaction') AS waiting
                FROM pg_stat_activity
                WHERE datname = current_database()
            """)
            
            # Cache hit ratio
            cache_stats = await conn.fetchrow("""
                SELECT 
                    sum(heap_blks_hit) / NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0) AS ratio
                FROM pg_statio_user_tables
            """)
            
            # Query statistics
            query_stats = await conn.fetchrow("""
                SELECT 
                    sum(xact_commit + xact_rollback) AS transactions,
                    sum(calls) AS queries
                FROM pg_stat_statements
            """)
            
            # Slow query count
            slow_queries = await conn.fetchval("""
                SELECT count(*) FROM pg_stat_statements
                WHERE mean_exec_time > $1
            """, self.alert_threshold_ms)
            
            return DatabaseMetrics(
                timestamp=datetime.utcnow(),
                connections_active=conn_stats["active"],
                connections_idle=conn_stats["idle"],
                connections_waiting=conn_stats["waiting"],
                cache_hit_ratio=cache_stats["ratio"] or 0,
                transactions_per_second=0,  # Calculate from delta
                queries_per_second=0,
                avg_query_duration_ms=0,
                slow_queries_count=slow_queries
            )
    
    async def monitor_loop(self, interval_seconds=30):
        """Continuous monitoring loop"""
        while True:
            metrics = await self.collect_metrics()
            self.samples.append(metrics)
            
            # Keep last hour of samples
            cutoff = datetime.utcnow().timestamp() - 3600
            self.samples = [m for m in self.samples if m.timestamp.timestamp() > cutoff]
            
            # Check for alert conditions
            if metrics.slow_queries_count > 10:
                print(f"ALERT: {metrics.slow_queries_count} slow queries detected")
            
            if metrics.cache_hit_ratio < 0.8:
                print(f"WARNING: Low cache hit ratio: {metrics.cache_hit_ratio:.2%}")
            
            await asyncio.sleep(interval_seconds)
```

## Common Pitfalls to Avoid

When troubleshooting database issues, avoid these common mistakes:

1. **Jumping to conclusions**: Always verify with data. The obvious cause is rarely the actual cause.

2. **Ignoring the obvious**: Check basic things first—connection strings, network connectivity, disk space.

3. **Not looking at the full picture**: A slow query might be slow because of locks, not because of the query itself.

4. **Assuming the problem is new**: Check if performance degraded gradually or suddenly. Recent changes are more likely to be the cause.

5. **Restarting without investigation**: Restarting masks symptoms and can make debugging harder. Understand the problem first.

6. **Ignoring logs**: Database logs contain valuable information about errors, checkpoints, and maintenance operations.

7. **Forgetting about maintenance**: Tables need VACUUM and ANALYZE. Indexes need rebuilding. Missing maintenance causes performance degradation over time.

8. **Testing in production**: Always reproduce issues in a non-production environment first.

## Conclusion

Database troubleshooting requires a combination of knowledge, tools, and systematic methodology. The techniques in this guide provide a foundation for diagnosing and resolving the most common database problems. Remember to:

- Use monitoring and logging to establish baselines and detect anomalies early
- Understand your database's internal workings to interpret diagnostic data correctly
- Follow systematic troubleshooting methodologies rather than guessing
- Implement protective measures like connection pooling, rate limiting, and alerting
- Maintain and test recovery procedures before emergencies occur

With these skills, you'll be equipped to handle the database challenges that arise in production systems and keep your applications running smoothly.

## See Also

- [Database API Design](../02_intermediate/04_database_api_design.md) - Designing robust database integrations
- [Database Architecture Patterns](../03_system_design/database_architecture_patterns.md) - CQRS and Saga patterns
- [Database Selection Framework](../01_foundations/database_selection_framework.md) - Choosing the right database
- [SQLite Deep Dive](../02_core_concepts/sqlite_deep_dive.md) - Embedded database optimization
