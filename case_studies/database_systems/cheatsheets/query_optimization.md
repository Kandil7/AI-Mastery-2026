# Query Optimization Patterns

## Slow Query Patterns

### Pattern 1: N+1 Queries

**Problem:**
```python
# BAD: N+1 queries
users = db.query("SELECT * FROM users")
for user in users:
    orders = db.query("SELECT * FROM orders WHERE user_id = ?", user.id)
    # Process orders
```

**Solution:**
```python
# GOOD: Single query with JOIN
results = db.query("""
    SELECT u.*, o.id as order_id, o.total
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
""")
```

**MongoDB:**
```javascript
// BAD: N+1
db.users.find().forEach(user => {
    const orders = db.orders.find({userId: user._id})
})

// GOOD: $lookup
db.users.aggregate([
    {
        $lookup: {
            from: "orders",
            localField: "_id",
            foreignField: "userId",
            as: "orders"
        }
    }
])
```

---

### Pattern 2: SELECT *

**Problem:**
```sql
-- BAD: Fetching all columns
SELECT * FROM orders WHERE id = 123;

-- GOOD: Fetch only needed columns
SELECT id, status, total FROM orders WHERE id = 123;
```

---

### Pattern 3: Pagination Without Index

**Problem:**
```sql
-- BAD: Offset on large table
SELECT * FROM orders ORDER BY created_at DESC LIMIT 10 OFFSET 100000;

-- GOOD: Keyset pagination
SELECT * FROM orders 
WHERE created_at < :last_seen_timestamp
AND id < :last_seen_id
ORDER BY created_at DESC, id DESC
LIMIT 10;
```

**Cursor-based approach:**
```python
def get_orders_after(cursor=None, limit=10):
    if cursor:
        last_timestamp, last_id = cursor
        query = """
            SELECT * FROM orders 
            WHERE (created_at, id) < (%s, %s)
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        """
        return db.execute(query, (last_timestamp, last_id, limit))
```

---

### Pattern 4: Non-Sargable Queries

**Problem:**
```sql
-- BAD: Function on column
SELECT * FROM orders WHERE YEAR(created_at) = 2024;
SELECT * FROM users WHERE LOWER(email) = 'john@example.com';

-- GOOD: Use indexed column directly
SELECT * FROM orders WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
SELECT * FROM users WHERE email = 'john@example.com';
```

**Range queries:**
```sql
-- BAD
SELECT * FROM products WHERE price * 1.1 > 100;

-- GOOD
SELECT * FROM products WHERE price > 100 / 1.1;
```

---

### Pattern 5: Missing Index on JOIN

**Problem:**
```sql
-- No index on orders.customer_id
SELECT o.*, c.name 
FROM orders o 
JOIN customers c ON o.customer_id = c.id
WHERE c.region = 'US';

-- GOOD: Add index
CREATE INDEX idx_orders_customer ON orders(customer_id);
```

---

## Optimization Techniques

### Use EXPLAIN

```sql
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT ...
```

Look for:
- **Seq Scan** on large tables (bad)
- **Index Scan** (good)
- **Nested Loop** for small tables (ok)
- **Hash Join** for large tables (good)
- **Sort** using index (good)

---

### Batch Operations

**Bad:**
```python
# Individual inserts
for item in items:
    db.execute("INSERT INTO orders ...", item)
```

**Good:**
```python
# Batch insert
db.executemany("INSERT INTO orders ...", items)
```

**PostgreSQL:**
```sql
INSERT INTO orders (customer_id, total) VALUES
(1, 100), (2, 200), (3, 300);
```

---

### Use Appropriate Data Types

```sql
-- BAD: Using VARCHAR for IDs
CREATE TABLE orders (id VARCHAR(20), ...);

-- GOOD: Using appropriate types
CREATE TABLE orders (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    status VARCHAR(20) CHECK (status IN ('pending', 'complete')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### Leverage Caching

```python
def get_user(user_id):
    # Check cache first
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Query database
    user = db.query("SELECT * FROM users WHERE id = ?", user_id)
    
    # Cache result
    redis.setex(f"user:{user_id}", 300, json.dumps(user))
    return user
```

---

## Query Analysis Checklist

- [ ] Is the query using an index? (Check EXPLAIN)
- [ ] Are only needed columns selected?
- [ ] Is there an N+1 pattern?
- [ ] Is pagination efficient?
- [ ] Are functions on indexed columns avoided?
- [ ] Are batch operations used for bulk data?
- [ ] Is appropriate data type used?
- [ ] Can results be cached?

---

## Common Fixes

| Problem | Solution |
|---------|----------|
| Slow JOIN | Add indexes on JOIN columns |
| Slow GROUP BY | Add index on GROUP BY columns |
| Slow DISTINCT | Consider using GROUP BY |
| Slow OR | Use UNION instead |
| Slow IN | Use EXISTS or JOIN |
| Slow LIKE | Use full-text search or trigram index |
| Slow NOT NULL | Use composite index with IS NULL |

---

## MongoDB Optimization

```javascript
// Use projection
db.orders.find({status: 'complete'}, {total: 1, customer_id: 1})

// Use covered queries
db.orders.createIndex({status: 1, total: 1})  // All queried fields indexed

// Use hint
db.orders.find({status: 'complete'}).hint({status: 1})
```

---

## Redis Optimization

```python
# Use pipelining for batch operations
pipe = redis.pipeline()
for key in keys:
    pipe.get(key)
results = pipe.execute()  # Round trip instead of N round trips

# Use MGET for multiple gets
values = redis.mget(*keys)  # Single round trip
```

---

*See also: [Indexing Strategies](indexing_strategies.md)*
