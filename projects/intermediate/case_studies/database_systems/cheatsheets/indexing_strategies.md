# Indexing Strategies

## Index Types Overview

| Index Type | Use Case | Example |
|------------|----------|---------|
| **B-tree** | Equality, range queries | `WHERE status = 'active'` |
| **Hash** | Simple equality | `WHERE id = ?` |
| **GIN** | Full-text, arrays, JSONB | `WHERE data->'tags' ? 'featured'` |
| **GiST** | Geospatial, full-text | `WHERE location && box` |
| **BRIN** | Time-series, sequential | `WHERE timestamp > '2024-01-01'` |
| **Partial** | Subset of rows | `WHERE status = 'active'` |
| **Composite** | Multi-column | `WHERE (a, b) = (?, ?)` |

---

## When to Create Indexes

### Create an index when:

1. **Column in WHERE clause**
   ```sql
   SELECT * FROM orders WHERE customer_id = 123;
   -- Index on customer_id helps
   ```

2. **Column in JOIN**
   ```sql
   SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id;
   -- Index on orders.customer_id AND customers.id helps
   ```

3. **Column in ORDER BY**
   ```sql
   SELECT * FROM products ORDER BY created_at DESC;
   -- Index on created_at helps avoid sort
   ```

4. **Column in GROUP BY**
   ```sql
   SELECT category, COUNT(*) FROM products GROUP BY category;
   -- Index on category helps
   ```

### Don't create indexes when:

1. **Small tables** (< 1000 rows)
2. **Frequently updated columns** (write-heavy)
3. **Low-cardinality columns** (boolean, gender)
4. **Columns rarely queried**

---

## Index Design Patterns

### Pattern 1: Composite Index Column Order

**Rule:** Equality conditions first, then range conditions

```sql
-- Query: WHERE status = 'active' AND created_at > '2024-01-01'
-- Best index: (status, created_at)
CREATE INDEX idx_orders_status_date ON orders(status, created_at);

-- WRONG: (created_at, status) - range first prevents index usage
```

### Pattern 2: Covering Index

Include all columns needed by query:

```sql
-- Query: SELECT id, name, email FROM users WHERE email = '?'
-- Best index: Include all selected columns
CREATE INDEX idx_users_email_covering ON users(email) INCLUDE (id, name);
```

### Pattern 3: Partial Index

Index only active rows:

```sql
-- Query: SELECT * FROM orders WHERE status = 'active'
-- Partial index: Only index active orders
CREATE INDEX idx_orders_active ON orders(created_at) 
WHERE status = 'active';
```

### Pattern 4: BRIN for Time-Series

Block Range Index for sequential data:

```sql
-- For time-series data, small pages
CREATE INDEX idx_metrics_time_brin ON metrics USING BRIN(timestamp)
WITH (pages_per_range = 128);
```

---

## Common Index Mistakes

### Mistake 1: Indexing Every Column

```sql
-- BAD: Index on every column
CREATE INDEX idx1 ON table(col1);
CREATE INDEX idx2 ON table(col2);
CREATE INDEX idx3 ON table(col3);

-- GOOD: Composite for common queries
CREATE INDEX idx_composite ON table(col1, col2);
```

### Mistake 2: Using Functions on Indexed Columns

```sql
-- BAD: Function prevents index usage
SELECT * FROM users WHERE LOWER(email) = 'john@example.com';

-- GOOD: Case-insensitive index
CREATE INDEX idx_users_email_ci ON users(LOWER(email));
-- Query:
SELECT * FROM users WHERE LOWER(email) = LOWER('John@Example.com');
```

### Mistake 3: Ignoring Index Selectivity

```sql
-- BAD: Index on low-selectivity column (boolean)
CREATE INDEX idx_users_active ON users(is_active);
-- Only 5% of rows are active - table scan might be faster

-- GOOD: Use with other columns
CREATE INDEX idx_users_active_country ON users(is_active, country);
```

---

## Diagnosing Missing Indexes

### PostgreSQL: pg_stat_statements

```sql
-- Enable extension
CREATE EXTENSION pg_stat_statements;

-- Find slow queries
SELECT query, calls, total_exec_time, mean_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

### PostgreSQL: EXPLAIN ANALYZE

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders 
WHERE customer_id = 123 
AND created_at > '2024-01-01';

-- Look for:
-- - "Seq Scan" (bad for large tables)
-- - "Index Scan using idx_..." (good)
-- - "Bitmap" (acceptable)
```

### MongoDB: explain()

```javascript
db.orders.find({ customer_id: 123, created_at: { $gt: ISODate("2024-01-01") }})
.explain("executionStats")

// Look for:
// - "IXSCAN" (good)
// - "COLLSCAN" (bad)
```

---

## Index Maintenance

### Check Index Usage

```sql
-- PostgreSQL: Unused indexes
SELECT indexrelname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexrelname NOT LIKE '%pkey%';
```

### Rebuild Fragmented Indexes

```sql
-- PostgreSQL
REINDEX INDEX CONCURRENTLY idx_orders_customer;

-- MongoDB
db.orders.reIndex()
```

### Monitor Index Size

```sql
-- PostgreSQL
SELECT pg_size_pretty(pg_relation_size('idx_orders_customer'));
```

---

## Quick Reference

| Scenario | Index Type | Example |
|----------|------------|---------|
| User lookup by email | B-tree | `CREATE INDEX ON users(email)` |
| Product search by name | GIN (full-text) | `CREATE INDEX ON products USING GIN(to_tsvector('english', name))` |
| Recent orders by date | BRIN | `CREATE INDEX ON orders USING BRIN(created_at)` |
| Location queries | GiST | `CREATE INDEX ON stores USING GIST(location)` |
| Array contains | GIN | `CREATE INDEX ON products USING GIN(tags)` |
| JSONB field | GIN | `CREATE INDEX ON orders USING GIN(data)` |
