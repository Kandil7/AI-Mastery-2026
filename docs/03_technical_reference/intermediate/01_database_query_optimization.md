# Database Query Optimization Guide

## Overview

Database query optimization is a critical skill for building high-performance applications. This guide covers the fundamental techniques, strategies, and best practices for optimizing SQL queries across PostgreSQL, MySQL, and other relational databases. Understanding how the database executes your queries and knowing how to influence that execution through indexes, query restructuring, and configuration tuning can dramatically improve application performance—often by orders of magnitude.

The optimization process begins with understanding how the database query optimizer works, then systematically identifying bottlenecks through execution plan analysis, and finally applying the appropriate remediation techniques. This guide provides a comprehensive framework for each step of this process, along with practical examples and code samples that you can apply directly to your projects.

Whether you are dealing with a small application with thousands of records or a large-scale system handling billions of rows, the principles and techniques outlined here will help you build more efficient, scalable database operations. The guide also covers advanced topics like cost-based optimization, query caching strategies, and performance benchmarking methodologies that are essential for production systems.

---

## 1. Query Execution Plans and Optimization

### Understanding EXPLAIN and EXPLAIN ANALYZE

The first step in query optimization is understanding how the database executes your queries. Every major relational database provides tools to inspect query execution plans, which show the steps the optimizer will take to retrieve your data. In PostgreSQL, the `EXPLAIN` command reveals the planned execution strategy, while `EXPLAIN ANALYZE` actually executes the query and measures real performance metrics including execution time, row counts at each stage, and I/O operations.

```sql
-- Basic execution plan analysis in PostgreSQL
EXPLAIN
SELECT customer_id, COUNT(*) AS order_count, SUM(total_amount) AS total_spent
FROM orders
WHERE order_date >= '2024-01-01' AND status = 'completed'
GROUP BY customer_id
ORDER BY total_spent DESC
LIMIT 10;
```

The output shows several important elements that reveal how the query will be executed. The `Seq Scan` operation indicates a full table scan, which can be expensive for large tables. The presence of `Index Scan` or `Index Only Scan` suggests that indexes are being used effectively. For joins, you might see `Hash Join`, `Merge Join`, or `Nested Loop`, each with different performance characteristics depending on the data volume and available indexes.

```sql
-- Detailed analysis with timing and buffer information
EXPLAIN (ANALYZE, BUFFERS, TIMING, FORMAT JSON)
SELECT o.*, c.name, c.email
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.created_at > NOW() - INTERVAL '30 days'
  AND c.status = 'active';
```

The `ANALYZE` option actually runs the query and provides real timing information, which is crucial for identifying actual bottlenecks rather than theoretical ones. The `BUFFERS` option shows how many pages were read from memory versus disk, helping you understand cache effectiveness. A high ratio of shared buffer hits indicates good cache utilization, while excessive disk reads suggest you need to either increase available memory or optimize your queries and indexes.

### Common Execution Plan Operations

Execution plans consist of various node types that represent different operations. Understanding these operations is essential for interpreting plans correctly and identifying optimization opportunities. Each operation has specific performance characteristics and indicates different things about how your query is being processed.

| Operation | Description | Performance Implication |
|-----------|-------------|------------------------|
| **Seq Scan** | Full table scan reading all rows | Slow for large tables, often indicates missing index |
| **Index Scan** | Uses index to find and retrieve rows | Efficient for selective queries |
| **Index Only Scan** | Uses index without table access | Fastest for covered queries |
| **Bitmap Heap Scan** | Bitmap-based row retrieval | Efficient for combining multiple index scans |
| **Nested Loop** | Iterates rows from one table for each row of another | Good for small tables or highly selective queries |
| **Hash Join** | Builds hash table for join | Efficient for large tables, requires memory |
| **Merge Join** | Sorts and merges sorted inputs | Efficient when inputs are already sorted |
| **Sort/Order By** | Sorts result set | Can be expensive for large datasets |
| **Aggregate** | Groups and aggregates data | Performance depends on grouping efficiency |

The choice between different join strategies depends on table sizes, available indexes, and statistical information about the data. The query optimizer makes these decisions automatically based on cost estimates, but understanding the trade-offs helps you know what to look for and how to influence the optimizer when it makes poor choices.

### Identifying Performance Issues

PostgreSQL's `pg_stat_statements` extension provides a powerful way to identify the most expensive queries in your system. This extension tracks execution statistics for all queries, allowing you to focus your optimization efforts on the queries that have the biggest impact on performance.

```sql
-- Enable pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find the top 10 most time-consuming queries
SELECT
    substring(query, 1, 100) AS query_preview,
    calls,
    total_exec_time / 1000 AS total_seconds,
    mean_exec_time AS avg_ms,
    rows / calls AS avg_rows,
    (100 * total_exec_time / sum(total_exec_time) OVER())::numeric(5,2) AS pct_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

This query reveals which queries consume the most database resources. The `calls` column shows how often each query runs, while `total_exec_time` shows cumulative time. A query with high `mean_exec_time` but low `calls` might benefit from optimization, but a query with moderate execution time that runs thousands of times per minute might be more impactful to optimize.

```sql
-- Identify queries with poor cache performance
SELECT
    query,
    calls,
    shared_blks_hit,
    shared_blks_read,
    CASE
        WHEN shared_blks_hit + shared_blks_read > 0
        THEN round(100.0 * shared_blks_hit / (shared_blks_hit + shared_blks_read), 2)
        ELSE 0
    END AS cache_hit_ratio
FROM pg_stat_statements
WHERE shared_blks_read > 0
ORDER BY shared_blks_read DESC
LIMIT 20;
```

Low cache hit ratios often indicate queries that are scanning large portions of the database without using indexes effectively. These queries may benefit from additional indexes, query restructuring, or increased buffer pool size. A cache hit ratio below 90% for frequently executed queries is worth investigating.

---

## 2. Index Optimization and Maintenance

### Index Types and Use Cases

Choosing the right index type for your specific use case is fundamental to achieving optimal query performance. Different index types are optimized for different query patterns, and using the wrong type can actually degrade performance due to increased write overhead and storage space.

**B-Tree indexes** are the default and most commonly used index type in PostgreSQL and MySQL. They excel at equality comparisons and range queries, making them ideal for most operational queries. B-Tree indexes maintain data in sorted order, which supports both exact matches and efficient range scans.

```sql
-- Standard B-Tree indexes for common query patterns
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status_date ON orders(status, order_date DESC);
CREATE INDEX idx_products_category_price ON products(category_id, price DESC);

-- Partial indexes for specific query patterns (PostgreSQL)
CREATE INDEX idx_active_orders ON orders(customer_id, order_date)
WHERE status = 'active';

CREATE INDEX idx_pending_shipments ON orders(customer_id, shipped_at)
WHERE status IN ('pending', 'processing', 'shipped');
```

Partial indexes are particularly powerful because they are smaller, faster to maintain, and more targeted to your actual query patterns. By indexing only the rows that are frequently queried, you reduce both storage overhead and the maintenance cost of keeping the index current.

**Hash indexes** provide O(1) lookup performance for equality comparisons but cannot support range queries. They are smaller than B-Tree indexes and can be faster for point lookups, but the lack of range query support limits their applicability.

```sql
-- Hash index for session token lookups
CREATE INDEX idx_sessions_token USING hash (session_token);

-- For UUID columns, consider hash indexes for exact matches
CREATE INDEX idx_users_uuid_hash USING hash (user_uuid);
```

**GIN (Generalized Inverted Index)** indexes are designed for composite data types like arrays, JSONB, and full-text search. They excel at queries that need to check whether a value exists within a column or search for specific terms within text.

```sql
-- GIN index for array columns
CREATE INDEX idx_products_tags ON products USING GIN (tags);

-- GIN index for JSONB columns
CREATE INDEX idx_orders_metadata ON orders USING GIN (metadata);

-- GIN index for full-text search
CREATE INDEX idx_articles_content ON articles USING GIN (to_tsvector('english', content));
```

**BRIN (Block Range Index)** indexes are designed for very large, naturally ordered data such as time-series data. They divide the table into ranges of pages and maintain summary information for each range, making them extremely compact and fast to update.

```sql
-- BRIN index for time-series data
CREATE INDEX idx_sensor_readings_time ON sensor_readings USING BRIN (recorded_at);

-- BRIN index with page range specific size
CREATE INDEX idx_events_created ON events USING BRIN (created_at)
WITH (pages_per_range = 128);
```

### Index Design Best Practices

Effective index design requires understanding your query patterns and balancing read performance against write overhead. Each index you create speeds up reads but slows down INSERT, UPDATE, and DELETE operations because the database must maintain the index. Therefore, you should create indexes strategically based on actual query patterns rather than trying to index every column.

The **covering index** pattern stores additional columns in the index itself, allowing the database to satisfy queries entirely from the index without accessing the table. This technique, called an "index-only scan" in PostgreSQL or "covering index" in MySQL, can dramatically improve performance for queries that otherwise would require table lookups.

```sql
-- Covering index in PostgreSQL using INCLUDE clause
CREATE INDEX idx_orders_covering ON orders (customer_id, order_date)
INCLUDE (total_amount, status, shipping_address);

-- Now this query can be served entirely from the index
SELECT order_date, total_amount, status
FROM orders
WHERE customer_id = 12345
  AND order_date > '2024-01-01';

-- Covering index in MySQL
CREATE INDEX idx_orders_covering
ON orders (customer_id, order_date, total_amount, status);
```

Index column ordering matters significantly. Columns used in equality conditions should come first, followed by columns used in range conditions or sorting. The database can use index columns most efficiently when they appear in the order that matches your query's WHERE clause.

```sql
-- Good: equality on first column, range on second
CREATE INDEX idx_sales_region_date ON sales(region, sale_date);

-- Bad: range on first column prevents using second column effectively
CREATE INDEX idx_sales_bad ON sales(sale_date, region);

-- For ORDER BY, include the sorted columns
CREATE INDEX idx_products_category_price ON products(category_id, price DESC);
```

### Index Maintenance and Optimization

Indexes require ongoing maintenance to remain effective. Over time, indexes can become fragmented, bloated with dead tuples, and outdated statistics can cause the optimizer to make poor choices. Regular maintenance ensures consistent performance.

```sql
-- Check index usage and find unused indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

Indexes that are never scanned are wasting storage and slowing down writes. Consider dropping unused indexes after confirming they are not needed for application functionality. However, be cautious with indexes that might be used infrequently but are critical when needed.

```sql
-- Rebuild fragmented indexes to improve performance
REINDEX INDEX CONCURRENTLY idx_orders_customer_id;
REINDEX TABLE CONCURRENTLY orders;

-- For PostgreSQL 16+, use the more efficient REINDEX (VERBOSE)
REINDEX (VERBOSE, PARALLEL) TABLE orders;
```

Index bloat occurs when deleted or updated rows leave gaps in the index. This bloat increases index size, reduces cache efficiency, and slows scans. Regular reindexing eliminates bloat and can significantly improve performance, especially for indexes that experience heavy updates.

```sql
-- Monitor index bloat
SELECT
    schemaname || '.' || tablename AS table_name,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    pg_size_pretty(pg_relation_size(indexrelid) -
        pg_column_size(indexrelid)) AS bloat_size,
    round(100.0 * (pg_relation_size(indexrelid) -
        pg_column_size(indexrelid)) /
        pg_relation_size(indexrelid), 1) AS bloat_pct
FROM pg_stat_user_indexes
WHERE pg_relation_size(indexrelid) > 1024 * 1024
ORDER BY bloat_size DESC;
```

---

## 3. Query Rewriting Techniques

### Subquery Optimization

Subqueries can often be rewritten as joins for better performance. While modern optimizers are increasingly capable of transforming subqueries automatically, understanding these transformations helps you write more efficient queries and diagnose performance issues.

```sql
-- Original subquery
SELECT * FROM products
WHERE category_id IN (
    SELECT id FROM categories WHERE parent_id = 1
);

-- Rewritten as JOIN (often more efficient)
SELECT p.* FROM products p
INNER JOIN categories c ON p.category_id = c.id
WHERE c.parent_id = 1;

-- Alternative: Using IN with derived table
SELECT p.* FROM products p
WHERE p.category_id IN (
    SELECT id FROM categories WHERE parent_id = 1
);

-- For EXISTS, ensure correlation is efficient
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.id
    AND o.created_at > NOW() - INTERVAL '30 days'
);

-- Rewrite as JOIN with DISTINCT if appropriate
SELECT DISTINCT c.* FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
WHERE o.created_at > NOW() - INTERVAL '30 days';
```

The choice between subquery and JOIN depends on whether you need distinct values, the size of the tables involved, and available indexes. Generally, IN subqueries with small result sets perform well, while EXISTS is efficient when checking for existence without needing the actual joined data.

### Predicate Pushdown

Predicate pushdown moves filtering operations closer to the data source, reducing the amount of data that needs to be processed and transferred. This optimization is particularly valuable in complex queries involving views, CTEs, and federated tables.

```sql
-- Instead of filtering after join, push predicates to the joined tables
-- Bad: filters applied after retrieving all data
SELECT c.name, o.total
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE c.status = 'active' AND o.status = 'completed';

-- Good: filters pushed to WHERE clause for proper join optimization
SELECT c.name, o.total
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
WHERE c.status = 'active'
  AND o.status = 'completed';

-- For views, use WITH CHECK OPTION to ensure predicates are pushed
CREATE VIEW active_customers AS
SELECT * FROM customers WHERE status = 'active'
WITH CHECK OPTION;
```

### Using CTEs Effectively

Common Table Expressions (CTEs) improve query readability and can enable certain optimizations, but they are not always the most efficient approach. Understanding when to use CTEs versus subqueries or temporary tables helps you write optimal queries.

```sql
-- CTE for complex logic - readable but may not optimize as well
WITH recent_orders AS (
    SELECT customer_id, SUM(total_amount) AS total_spent
    FROM orders
    WHERE created_at > NOW() - INTERVAL '1 year'
    GROUP BY customer_id
),
top_customers AS (
    SELECT customer_id FROM recent_orders
    ORDER BY total_spent DESC LIMIT 1000
)
SELECT c.*, r.total_spent
FROM customers c
INNER JOIN recent_orders r ON c.id = r.customer_id
WHERE c.status = 'active'
ORDER BY r.total_spent DESC;

-- Consider materialization for expensive CTEs
WITH recent_orders AS (
    SELECT customer_id, SUM(total_amount) AS total_spent
    FROM orders
    WHERE created_at > NOW() - INTERVAL '1 year'
    GROUP BY customer_id
), MATERIALIZED top_customers AS (
    SELECT customer_id FROM recent_orders
    ORDER BY total_spent DESC LIMIT 1000
)
SELECT c.*, r.total_spent
FROM customers c
INNER JOIN recent_orders r ON c.id = r.customer_id
WHERE c.status = 'active'
ORDER BY r.total_spent DESC;
```

The `MATERIALIZED` hint in PostgreSQL prevents the CTE from being inlined into the parent query, which can be beneficial when the CTE is referenced multiple times or when inlining would cause poor performance.

---

## 4. Cost-Based vs Rule-Based Optimization

### Understanding the Query Optimizer

Modern relational databases use cost-based optimizers that evaluate multiple execution strategies and select the one with the lowest estimated cost. The optimizer estimates costs based on statistical information about table sizes, index selectivity, data distribution, and available resources. This approach generally produces better plans than rule-based systems, but it relies heavily on accurate statistics.

```sql
-- Update statistics for accurate optimization
ANALYZE orders;
ANALYZE orders (status, customer_id);  -- PostgreSQL: specify columns

-- Check table statistics
SELECT
    schemaname,
    relname,
    n_live_tup AS row_count,
    n_dead_tup AS dead_rows,
    last_vacuum,
    last_autovacuum,
    last_analyze
FROM pg_stat_user_tables
WHERE relname = 'orders';
```

Accurate statistics are essential for the optimizer to make good decisions. When statistics are stale, the optimizer may choose inefficient execution plans. Regular `ANALYZE` operations, typically run as part of auto-vacuum in PostgreSQL, keep statistics current.

### Influencing the Optimizer

While the cost-based optimizer generally makes good decisions, there are cases where you may need to provide hints or restructure queries to achieve better performance. Understanding how to influence the optimizer gives you control when automatic optimization falls short.

```sql
-- PostgreSQL: Use optimizer hints via extension
-- First, enable the pg_hint_plan extension
CREATE EXTENSION pg_hint_plan;

-- Force specific join order
/*+
    NestLoop(orders customers)
    SeqScan(orders)
*/
SELECT * FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE customers.region = 'US';

-- PostgreSQL: Set optimizer parameters
SET enable_seqscan = off;        -- Prefer index scans
SET enable_nestloop = off;      -- Disable nested loop joins
SET random_page_cost = 1.1;     -- Assume fast disk (SSD)
SET effective_cache_size = '8GB';  -- Tell optimizer available cache
```

While hints can solve specific performance issues, they should be used judiciously. Hints can become outdated as data distributions change, and they may prevent the optimizer from adapting to new query patterns. Whenever possible, prefer improving statistics, indexes, or query structure over using hints.

### Statistics and Plan Stability

Maintaining stable query performance requires consistent statistics and understanding how plan changes affect your application. Major changes in data volume or distribution can cause the optimizer to choose different plans, sometimes with worse performance.

```sql
-- Create statistics for better optimization on correlated columns
CREATE STATISTICS order_stats (dependencies, ndistinct)
ON customer_id, status FROM orders;

-- Monitor for significant plan changes
SELECT
    query,
    plan,
    calls,
    mean_exec_time,
    stddev_exec_time
FROM pg_stat_statements
WHERE stddev_exec_time > mean_exec_time * 2
ORDER BY stddev_exec_time DESC;
```

Multi-column statistics help the optimizer understand relationships between columns, which is crucial for accurate cardinality estimates in queries with multiple predicates. Without such statistics, the optimizer may underestimate or overestimate result set sizes, leading to poor join order or join method selection.

---

## 5. Performance Benchmarking Methodologies

### Micro-Benchmarking Queries

Systematic benchmarking is essential for understanding query performance characteristics and validating optimization efforts. Micro-benchmarks focus on individual query performance, allowing precise measurement of optimization effects.

```sql
-- Timing queries in PostgreSQL
\timing on

-- Using pgbench for standardized benchmarking
-- Initialize test database
pgbench -i -s 100 mydb

-- Simple benchmark: 10 clients, 2 threads, 1000 transactions
pgbench -c 10 -j 2 -t 1000 mydb

-- Benchmark with specific SQL
pgbench -c 10 -j 2 -t 1000 -f /path/to/test_queries.sql mydb

-- Custom benchmark script for specific query testing
\benchmark 'SELECT * FROM orders WHERE customer_id = 1234' 1000
```

pgbench provides a standardized way to measure database throughput under concurrent load. By running consistent benchmarks before and after optimizations, you can quantify the impact of your changes. The tool supports custom scripts for testing specific query patterns relevant to your application.

```python
# Python-based query benchmarking
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import psycopg2

def benchmark_query(query, iterations=100, connections=10):
    """Benchmark a query with multiple connections"""
    times = []

    def run_query():
        conn = psycopg2.connect("dbname=mydb user=postgres")
        cur = conn.cursor()
        start = time.perf_counter()
        cur.execute(query)
        cur.fetchall()
        elapsed = time.perf_counter() - start
        cur.close()
        conn.close()
        return elapsed

    with ThreadPoolExecutor(max_workers=connections) as executor:
        results = list(executor.map(lambda _: run_query(), range(iterations)))

    return {
        'mean': statistics.mean(results),
        'median': statistics.median(results),
        'stdev': statistics.stdev(results) if len(results) > 1 else 0,
        'min': min(results),
        'max': max(results),
        'p95': sorted(results)[int(len(results) * 0.95)],
    }

# Example usage
result = benchmark_query(
    "SELECT * FROM orders WHERE status = 'pending'",
    iterations=1000,
    connections=20
)
print(f"Mean: {result['mean']*1000:.2f}ms, P95: {result['p95']*1000:.2f}ms")
```

For more realistic benchmarking, consider using application-level testing that simulates actual user behavior. This includes connection establishment, transaction management, and result processing. Pure query benchmarking can miss important overhead from application logic and ORM frameworks.

### Load Testing with Realistic Workloads

Production-like load testing validates that optimizations work under realistic conditions. This testing should simulate concurrent users, realistic query mixes, and appropriate data volumes.

```yaml
# k6 load testing configuration
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    stages: [
        { duration: '2m', target: 100 },   // Ramp up
        { duration: '5m', target: 100 },   // Steady state
        { duration: '2m', target: 200 },   // Stress
        { duration: '5m', target: 200 },   // Steady state
        { duration: '2m', target: 0 },     // Ramp down
    ],
    thresholds: {
        http_req_duration: ['p(95)<500'],  // 95% requests < 500ms
        http_req_failed: ['rate<0.01'],     // Error rate < 1%
    },
};

export default function () {
    const customerId = Math.floor(Math.random() * 10000) + 1;

    // Test query performance
    const response = http.get(
        `http://localhost:8080/api/customers/${customerId}/orders`,
        {
            tags: { name: 'customer_orders' },
            timeout: '10s'
        }
    );

    check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 500ms': (r) => r.timings.duration < 500
    });

    sleep(1);
}
```

Load testing should simulate realistic user behavior patterns, including think times, varying query complexity, and mixed read/write workloads. This helps identify performance bottlenecks that might not appear in micro-benchmarks but emerge under real-world conditions.

---

## Related Resources

- For database fundamentals, see [Database Fundamentals](../01_foundations/01_database_fundamentals.md)
- For indexing strategies, see [Indexing Strategies](../01_foundations/04_indexing_strategies.md)
- For performance tuning, see [Performance Tuning](../01_foundations/03_database_performance_tuning.md)
- For query processing basics, see [Query Processing](../01_foundations/01_database_fundamentals.md)