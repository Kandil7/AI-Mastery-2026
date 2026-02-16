# Query Rewrite Patterns

Query rewriting involves transforming SQL queries to improve performance while maintaining the same logical result. This is a powerful technique for optimizing database performance in AI/ML applications.

## Overview

Query rewriting goes beyond indexing to fundamentally change how queries are structured for better execution plans. For senior AI/ML engineers, understanding these patterns is essential for optimizing complex data processing pipelines and real-time inference systems.

## Common Rewrite Patterns

### EXISTS vs COUNT for Existence Checks
Replace expensive COUNT operations with EXISTS for existence checks.

```sql
-- Bad: Count all matching rows
IF (SELECT COUNT(*) FROM users WHERE email = 'test@example.com') > 0 THEN ...

-- Good: Stop after finding first match
IF EXISTS (SELECT 1 FROM users WHERE email = 'test@example.com') THEN ...
```

### JOIN vs Subquery Optimization
Rewrite correlated subqueries as JOINs for better performance.

```sql
-- Bad: Correlated subquery
SELECT u.name, u.email
FROM users u
WHERE u.id IN (
    SELECT o.user_id 
    FROM orders o 
    WHERE o.total_amount > 1000
);

-- Good: JOIN (often faster)
SELECT DISTINCT u.name, u.email
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.total_amount > 1000;
```

### Window Functions vs Self-Joins
Use window functions instead of self-joins for ranking and aggregation.

```sql
-- Bad: Self-join for ranking
SELECT u1.name, COUNT(*) as rank
FROM users u1
LEFT JOIN users u2 ON u1.score < u2.score
GROUP BY u1.id, u1.name;

-- Good: Window function
SELECT name, 
       RANK() OVER (ORDER BY score DESC) as rank
FROM users;
```

### Materialized Views vs Complex Queries
Pre-compute expensive aggregations using materialized views.

```sql
-- Bad: Complex query run repeatedly
SELECT 
    DATE(order_date) as day,
    COUNT(*) as orders,
    SUM(total_amount) as revenue,
    AVG(total_amount) as avg_order
FROM orders
GROUP BY DATE(order_date)
ORDER BY day DESC
LIMIT 30;

-- Good: Materialized view
CREATE MATERIALIZED VIEW daily_sales_summary AS
SELECT 
    DATE(order_date) as day,
    COUNT(*) as orders,
    SUM(total_amount) as revenue,
    AVG(total_amount) as avg_order
FROM orders
GROUP BY DATE(order_date);

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_sales_summary;
```

## AI/ML Specific Rewrite Patterns

### Feature Engineering Queries
Optimize complex feature calculation queries.

```sql
-- Bad: Multiple sequential queries
SELECT * FROM users WHERE created_at < '2024-01-01';
SELECT * FROM orders WHERE user_id IN (...);
SELECT * FROM features WHERE user_id IN (...);

-- Good: Single optimized query with joins
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
WHERE u.created_at < '2024-01-01'
GROUP BY u.user_id, u.age, u.gender;
```

### Model Training Data Preparation
Optimize queries that generate training datasets.

```sql
-- Bad: N+1 query problem
FOR user IN users LOOP
    SELECT COUNT(*) FROM orders WHERE user_id = user.id;
    SELECT AVG(total_amount) FROM orders WHERE user_id = user.id;
END LOOP;

-- Good: Single query with conditional aggregation
SELECT 
    u.user_id,
    COUNT(CASE WHEN o.order_id IS NOT NULL THEN 1 END) as order_count,
    AVG(CASE WHEN o.order_id IS NOT NULL THEN o.total_amount END) as avg_order_value,
    MAX(o.order_date) as last_order_date
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id;
```

### Real-time Inference Queries
Optimize low-latency queries for production inference.

```sql
-- Bad: Multiple round trips
SELECT model_id FROM models WHERE name = 'recommendation-model';
SELECT artifact_path FROM model_versions WHERE model_id = ? AND version = 'latest';

-- Good: Single query with JOIN
SELECT mv.artifact_path, mv.metrics
FROM models m
INNER JOIN model_versions mv ON m.model_id = mv.model_id
WHERE m.name = 'recommendation-model' 
  AND mv.created_at = (
      SELECT MAX(created_at) 
      FROM model_versions 
      WHERE model_id = m.model_id
  );
```

## Advanced Rewrite Techniques

### Query Decomposition
Break complex queries into simpler components.

```sql
-- Original complex query
SELECT u.name, COUNT(o.order_id), SUM(o.total_amount)
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
WHERE u.created_at > '2023-01-01'
  AND o.status = 'completed'
  AND oi.quantity > 0
GROUP BY u.user_id, u.name
HAVING COUNT(o.order_id) > 5
ORDER BY SUM(o.total_amount) DESC
LIMIT 10;

-- Decomposed approach
WITH active_users AS (
    SELECT user_id, name
    FROM users 
    WHERE created_at > '2023-01-01'
),
completed_orders AS (
    SELECT user_id, order_id, total_amount
    FROM orders 
    WHERE status = 'completed'
),
valid_items AS (
    SELECT order_id
    FROM order_items 
    WHERE quantity > 0
)
SELECT 
    au.name,
    COUNT(co.order_id) as order_count,
    SUM(co.total_amount) as total_spent
FROM active_users au
LEFT JOIN completed_orders co ON au.user_id = co.user_id
INNER JOIN valid_items vi ON co.order_id = vi.order_id
GROUP BY au.user_id, au.name
HAVING COUNT(co.order_id) > 5
ORDER BY SUM(co.total_amount) DESC
LIMIT 10;
```

### Predicate Pushdown
Move filtering conditions as early as possible in the query plan.

```sql
-- Bad: Filter after join
SELECT u.name, o.total_amount
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE u.created_at > '2024-01-01' AND o.total_amount > 100;

-- Good: Filter before join (predicate pushdown)
SELECT u.name, o.total_amount
FROM (
    SELECT * FROM users WHERE created_at > '2024-01-01'
) u
JOIN (
    SELECT * FROM orders WHERE total_amount > 100
) o ON u.user_id = o.user_id;
```

## Performance Testing Methodology

### Before/After Benchmarking
Always measure performance before and after rewrites:

```sql
-- Baseline measurement
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT * FROM complex_query;

-- After rewrite
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT * FROM rewritten_query;
```

### Load Testing
Test under realistic load conditions:

```sql
-- Simulate concurrent queries
DO $$
DECLARE
    i INTEGER := 0;
BEGIN
    WHILE i < 100 LOOP
        PERFORM * FROM optimized_query WHERE parameter = i % 100;
        i := i + 1;
    END LOOP;
END $$;
```

## Best Practices

1. **Start with EXPLAIN**: Understand current execution plan first
2. **Measure impact**: Always benchmark before and after
3. **Consider maintenance**: Complex rewrites may be harder to maintain
4. **Test edge cases**: Ensure rewritten queries handle all scenarios
5. **Document changes**: Explain why the rewrite was necessary
6. **Review regularly**: Query patterns change with data growth

## Related Resources

- [Execution Plans] - Understanding query execution strategies
- [Index Optimization] - How indexes affect query performance
- [Database Performance Tuning] - Comprehensive optimization guide
- [AI/ML Query Patterns] - Specialized patterns for machine learning applications