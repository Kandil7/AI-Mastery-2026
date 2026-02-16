# Query Execution Plans

Understanding how queries are executed helps senior AI/ML engineers write more efficient SQL and diagnose performance issues in production database systems.

## Overview

Query execution plans show the database's strategy for executing a query. They reveal which indexes are used, join methods selected, and estimated costs—critical information for optimizing AI/ML data pipelines and real-time inference systems.

## Reading Execution Plans

The `EXPLAIN` command shows how the database will execute your query, including which indexes it uses, join methods, and estimated costs.

### Basic EXPLAIN
```sql
EXPLAIN SELECT customer_id, COUNT(*)
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY customer_id;
```

### Detailed EXPLAIN with Analysis
```sql
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT customer_id, COUNT(*)
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY customer_id;
```

### Example Output Analysis
```
HashAggregate  (cost=1234.56..1234.60 rows=10 width=24) (actual time=15.234..15.238 rows=50 loops=1)
  ->  Index Scan using idx_orders_date on orders  (cost=0.45..1230.00 rows=1000 width=8) (actual time=0.123..10.456 rows=5000 loops=1)
        Index Cond: (order_date >= '2024-01-01'::date)
Planning Time: 0.456 ms
Execution Time: 15.678 ms
```

## Key Plan Components

### Cost Estimates
- **Cost**: Estimated cost units (arbitrary scale)
- **Rows**: Estimated number of rows processed
- **Width**: Average row size in bytes
- **Actual time**: Real execution time in milliseconds

### Operation Types
| Operation | Description | When Used |
|-----------|-------------|-----------|
| Seq Scan | Full table scan | No suitable index, small tables |
| Index Scan | Index-based retrieval | Selective queries |
| Index Only Scan | Index covers all needed columns | Covered queries |
| Nested Loop | Join by iterating rows | Small tables, indexed joins |
| Hash Join | Build hash table for join | Large table joins |
| Merge Join | Sorted join | Already sorted inputs |

## Common Query Patterns and Their Plans

### Point Queries (Single Row)
```sql
-- Good: Uses index
EXPLAIN SELECT * FROM users WHERE user_id = 123;

-- Bad: Sequential scan (no index)
EXPLAIN SELECT * FROM users WHERE email = 'user@example.com';
```

### Range Queries
```sql
-- Good: Index scan on date range
EXPLAIN SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31';

-- Bad: Sequential scan on non-indexed column
EXPLAIN SELECT * FROM logs WHERE message LIKE '%error%';
```

### Aggregation Queries
```sql
-- Good: Hash aggregate with indexed filter
EXPLAIN SELECT customer_id, COUNT(*) 
FROM orders 
WHERE status = 'completed' 
GROUP BY customer_id;

-- Bad: Sequential scan with sort
EXPLAIN SELECT * FROM large_table ORDER BY created_at DESC LIMIT 10;
```

### Join Queries
```sql
-- Good: Hash join with indexed foreign keys
EXPLAIN SELECT u.name, o.total_amount
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE u.created_at > '2024-01-01';

-- Bad: Nested loop with poor selectivity
EXPLAIN SELECT * FROM large_table1 t1
JOIN large_table2 t2 ON t1.id = t2.id;
```

## Query Optimization Techniques

### Index Usage Strategies

#### Covering Indexes
Create indexes that include all columns needed for the query to avoid table lookups:
```sql
-- Covering index for common query
CREATE INDEX idx_orders_user_status_total
ON orders(user_id, status) INCLUDE (total_amount);
```

#### Partial Indexes
Index only relevant subsets of data:
```sql
-- Index only active records
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';
```

### Join Optimization

#### Join Order
The database optimizer chooses join order, but you can influence it:
```sql
-- Force join order (use cautiously)
SELECT /*+ Leading(t1 t2 t3) */ *
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.t1_id
JOIN table3 t3 ON t2.id = t3.t2_id;
```

#### Materialized Views
Pre-compute expensive joins and aggregations:
```sql
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

## AI/ML Specific Query Patterns

### Feature Engineering Queries
```sql
-- Complex feature calculation
EXPLAIN SELECT 
    user_id,
    COUNT(*) as order_count,
    AVG(total_amount) as avg_order_value,
    MAX(order_date) as last_order_date,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_amount) as median_order_value
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY user_id;
```

### Model Training Data Preparation
```sql
-- Join multiple sources for training data
EXPLAIN SELECT 
    u.user_id,
    u.age,
    u.gender,
    COUNT(o.order_id) as purchase_count,
    SUM(o.total_amount) as total_spent,
    ARRAY_AGG(DISTINCT p.category) as purchased_categories
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
WHERE u.created_at < CURRENT_DATE - INTERVAL '30 days'
GROUP BY u.user_id, u.age, u.gender;
```

### Real-time Inference Queries
```sql
-- Low-latency queries for inference
EXPLAIN SELECT 
    model_id,
    version_number,
    artifact_path,
    metrics->>'accuracy' as accuracy
FROM model_versions
WHERE model_id = 'recommendation-model-1'
ORDER BY created_at DESC
LIMIT 1;
```

## Performance Diagnostic Checklist

### When Queries Are Slow
1. **Check for sequential scans** on large tables
2. **Verify index usage** matches query patterns
3. **Look for nested loops** with large datasets
4. **Check memory usage** (work_mem settings)
5. **Analyze statistics** (are they up-to-date?)

### Common Issues and Solutions
| Issue | Symptoms | Solution |
|-------|----------|----------|
| Missing indexes | Seq Scan on large tables | Create appropriate indexes |
| Poor join order | Nested loops with large tables | Add indexes, rewrite queries |
| Sort operations | "Sort" operations with high cost | Add covering indexes |
| Hash table overflow | "Hash" operations with high memory | Increase work_mem |
| Vacuum issues | High dead tuples, slow updates | Run VACUUM ANALYZE |

## Related Resources

- [Indexing Strategies] - How to create effective indexes
- [Performance Optimization] - Advanced query tuning techniques
- [Database Tuning] - Comprehensive performance optimization guide
- [AI/ML Query Patterns] - Specialized query patterns for machine learning applications