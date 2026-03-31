# Comprehensive Database Interview Preparation Guide

## Overview

Database knowledge is fundamental to software engineering interviews, particularly for backend, full-stack, and infrastructure roles. Whether you're interviewing for a junior position or a senior architect role, understanding database concepts demonstrates your ability to build scalable, performant systems.

This guide provides a comprehensive overview of database interview topics, organized from fundamental concepts to advanced distributed systems knowledge. It includes conceptual questions, practical scenarios, and system design challenges that reflect real interview experiences at major technology companies.

The interview topics covered here align with common interview formats including phone screens, technical interviews, and onsite loops. Each section includes the knowledge you need, example questions, and approaches to answering effectively.

This guide assumes basic computer science knowledge including data structures, algorithms, and programming fundamentals. Specific database experience varies by role—senior positions expect deeper knowledge while entry roles focus on fundamentals.

## Foundational Concepts

### SQL and Querying

#### Basic SQL Operations

Interviewers frequently test SQL proficiency through practical queries. Master these fundamental operations:

**SELECT with Filtering and Aggregation**:

```sql
-- Basic filtering
SELECT * FROM employees WHERE department = 'Engineering' AND salary > 100000;

-- Aggregation with GROUP BY
SELECT 
    department,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary,
    MAX(salary) as max_salary,
    MIN(salary) as min_salary
FROM employees
GROUP BY department
HAVING COUNT(*) > 10
ORDER BY avg_salary DESC;

-- Subqueries
SELECT * FROM employees 
WHERE salary > (SELECT AVG(salary) FROM employees);

-- JOINs
SELECT 
    o.order_id,
    c.customer_name,
    p.product_name,
    oi.quantity,
    oi.unit_price * oi.quantity as total
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= '2024-01-01';
```

**Common Interview Query Patterns**:

```sql
-- Finding duplicates
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Running total / cumulative sum
SELECT 
    order_date,
    amount,
    SUM(amount) OVER (ORDER BY order_date) as cumulative_amount
FROM orders;

-- Rank within groups
SELECT 
    employee_name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank,
    DENSE_RANK() OVER (ORDER BY salary DESC) as overall_rank
FROM employees;

-- Top N per group
SELECT * FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rn
    FROM employees
) ranked
WHERE rn <= 3;

-- Percentage of total
SELECT 
    category,
    sales,
    100.0 * sales / SUM(sales) OVER () as pct_of_total
FROM category_sales;
```

#### Common SQL Interview Questions

**Question: Given a table of transactions with columns (transaction_id, user_id, amount, timestamp), write a query to find the user's second transaction.**

```sql
-- Solution using ROW_NUMBER
SELECT * FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) as rn
    FROM transactions
) numbered
WHERE rn = 2;

-- Alternative using correlated subquery
SELECT t1.*
FROM transactions t1
WHERE (
    SELECT COUNT(*)
    FROM transactions t2
    WHERE t2.user_id = t1.user_id
    AND t2.timestamp < t1.timestamp
) = 1;
```

**Question: How would you find the maximum consecutive days a user logged in?**

```sql
-- Using LAG to identify gaps
WITH login_with_gaps AS (
    SELECT 
        user_id,
        login_date,
        LAG(login_date) OVER (PARTITION BY user_id ORDER BY login_date) as prev_login,
        login_date - LAG(login_date) OVER (PARTITION BY user_id ORDER BY login_date) as gap
    FROM user_logins
),
grouped AS (
    SELECT 
        user_id,
        login_date,
        SUM(CASE WHEN gap > 1 THEN 1 ELSE 0 END) 
            OVER (PARTITION BY user_id ORDER BY login_date) as grp
    FROM login_with_gaps
)
SELECT 
    user_id,
    COUNT(*) as consecutive_days
FROM grouped
GROUP BY user_id, grp
ORDER BY consecutive_days DESC
LIMIT 1;
```

### Database Normalization

Understanding normalization is essential for database design questions:

#### Normal Forms

| Normal Form | Description | Example |
|-------------|-------------|---------|
| 1NF | Atomic values, no repeating groups | Split "phone_numbers" into separate rows |
| 2NF | 1NF + no partial dependencies | Remove transitive dependencies |
| 3NF | 2NF + no transitive dependencies | Store city_id instead of city name in employee table |
| BCNF | 3NF + every determinant is a candidate key | Handle overlapping candidate keys |

**Interview Question: Explain the difference between normalization and denormalization. When would you use each?**

Answer: Normalization organizes data to reduce redundancy, improving data integrity but potentially impacting read performance due to joins. Denormalization intentionally adds redundancy to optimize read performance. Use normalization for transactional systems requiring data integrity; use denormalization for read-heavy analytical systems where performance outweighs update complexity.

### Indexes and Query Optimization

#### Index Types

**B-Tree Indexes**: The most common type, suitable for equality and range queries.

```sql
-- Create indexes
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date DESC);
CREATE INDEX idx_products_category_price ON products(category, price);
```

**Hash Indexes**: Optimized for equality comparisons, not range queries.

```sql
CREATE INDEX idx_users_email ON users USING HASH(email);
```

**Composite Indexes**: Index on multiple columns. The order matters:

```sql
-- This index supports:
-- WHERE category = 'X'
-- WHERE category = 'X' AND price < 100
-- Does NOT support WHERE price < 100
CREATE INDEX idx_products_cat_price ON products(category, price);
```

**Interview Question: What index would you create for this query?**

```sql
SELECT * FROM orders 
WHERE customer_id = 123 
AND status = 'pending' 
ORDER BY created_at DESC;
```

Answer: A composite index on (customer_id, status, created_at) would be optimal. The columns should be ordered by selectivity—customer_id first (most selective), then status, then created_at for sorting.

### ACID Properties

Understanding transactions and ACID properties is fundamental:

**Atomicity**: All operations in a transaction complete or none do. If any operation fails, the entire transaction is rolled back.

**Consistency**: A transaction brings the database from one valid state to another, maintaining all database constraints.

**Isolation**: Concurrent transactions appear to execute serially. The isolation level determines how they interact.

**Durability**: Once committed, transactions persist even if the system fails.

**Interview Question: Explain the difference between READ COMMITTED and REPEATABLE READ isolation levels.**

Answer: READ COMMITTED sees only committed data at the time of each statement. REPEATABLE READ sees a consistent snapshot from the start of the transaction, preventing non-repeatable reads where the same query returns different results within a transaction.

## Intermediate Concepts

### Database Locking and Concurrency

Understanding locks is crucial for performance and correctness:

#### Lock Types

**Shared Locks (S)**: Allow reading, block writes.

**Exclusive Locks (X)**: Allow reading and writing, block everything else.

**Deadlock Example**:

```
Transaction A: LOCK account_1 (X), then tries LOCK account_2 (X)
Transaction B: LOCK account_2 (X), then tries LOCK account_1 (X)
Result: Deadlock - each waiting for the other
```

**Interview Question: How would you prevent deadlocks in a financial transfer system?**

Answer: Prevent deadlocks by enforcing consistent lock ordering. Always acquire locks in a predetermined order (e.g., always lock accounts with lower IDs first). Also, keep transactions short, use appropriate isolation levels, and implement deadlock detection with automatic rollback.

### Query Execution and Optimization

#### Reading Execution Plans

```sql
EXPLAIN ANALYZE
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.region = 'West';
```

Key metrics to examine:

- **Seq Scan vs Index Scan**: Sequential scans indicate missing indexes or unselective queries
- **Nested Loop**: Can be expensive for large datasets unless inner table has indexes
- **Hash Join**: Efficient for large tables without indexes
- **Sort/Merge**: Used for large ordered joins

### Replication and High Availability

#### Replication Types

**Synchronous Replication**: Waits for all replicas to acknowledge before committing. Guarantees zero data loss but increases latency.

**Asynchronous Replication**: Commits to primary immediately, replicates in background. Potential data loss on primary failure, lower latency.

**Semi-synchronous**: Waits for at least one replica before committing. Balanced approach.

**Interview Question: What happens when a replica falls behind in asynchronous replication?**

Answer: The replica lags behind the primary, serving stale data. If the primary fails, recent transactions may be lost (or the replica may need to be rebuilt). Solutions include monitoring lag, alerts for thresholds, automatic promotion when primary fails, and application-level handling of stale reads.

### Sharding and Partitioning

#### Horizontal vs Vertical Partitioning

**Vertical Partitioning**: Splitting columns across tables (e.g., storing blob data separately).

**Horizontal Sharding**: Splitting rows across databases based on a shard key.

```sql
-- Range-based partitioning (PostgreSQL)
CREATE TABLE orders (
    order_id BIGINT,
    ...
    created_at DATE
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
```

**Interview Question: How would you choose a shard key?**

Answer: Choose a key that evenly distributes data, minimizes cross-shard queries, and matches access patterns. Good choices often align with common query filters. Avoid keys with hot spots (e.g., timestamps with recent data getting all writes).

## Advanced Topics

### Distributed Transactions

Distributed transactions spanning multiple databases or services present significant challenges:

#### Two-Phase Commit (2PC)

A protocol for ensuring atomicity across multiple participants:

```
Phase 1 (Prepare):
- Coordinator asks all participants to prepare
- Participants lock resources and prepare to commit
- Participants respond with prepared/not prepared

Phase 2 (Commit):
- If all prepared, coordinator tells all to commit
- If any failed, coordinator tells all to rollback
```

**Limitations**: Coordinator is a single point of failure; blocking if coordinator fails after prepare.

#### Saga Pattern

For distributed systems, sagas provide an alternative to ACID transactions:

```
Order Saga:
1. Create Order (local transaction)
2. Reserve Inventory (local transaction)
3. Process Payment (local transaction)
4. Schedule Shipping (local transaction)

If step 3 fails:
- Cancel Shipping
- Refund Payment
- Release Inventory
- Cancel Order
```

**Interview Question: How does the Saga pattern differ from 2PC?**

Answer: 2PC provides atomicity with blocking; sagas provide eventual consistency with compensation. Sagas don't require distributed locking, scale better, but require designing compensating actions for each step.

### CAP Theorem and Distributed Trade-offs

Understanding CAP theorem helps reason about distributed databases:

- **Consistency**: All nodes see the same data simultaneously
- **Availability**: Every request receives a non-error response
- **Partition Tolerance**: System continues operating despite network failures

You can only guarantee two of three. In practice, partitions are inevitable, so you're choosing between CP (consistent during partitions) and AP (available during partitions).

**Interview Question: Is CAP theorem still relevant?**

Answer: Modern understanding recognizes CAP describes only a narrow window during network partitions. More nuanced frameworks consider latency, consistency levels (strong, eventual, causal), and application requirements. The conversation has evolved from "choose two" to "optimize for your requirements."

**Interview Question: What is the PACELC model?**

Answer: PACELC extends CAP by acknowledging that even without partitions, systems must choose between Latency and Consistency (ELC). This provides a more complete picture: during a partition (P), choose between Availability (A) or Consistency (C); else (E), choose between Latency (L) or Consistency (C). For example, DynamoDB prioritizes low latency and accepts eventual consistency during normal operation.

### Consistency Models

Beyond ACID, modern databases offer various consistency guarantees:

| Model | Description | Use Case |
|-------|-------------|----------|
| Strong Consistency | Reads see most recent write | Financial transactions |
| Eventual Consistency | Writes propagate asynchronously | Social media, caching |
| Causal Consistency | Respects causality | Collaborative apps |
| Read Your Writes | See your own writes immediately | User sessions |
| Session Consistency | Within a session, reads are consistent | Shopping carts |

**Interview Question: Why do many distributed databases default to eventual consistency?**

Answer: Eventual consistency provides lower latency and higher availability. Strong consistency requires coordination (voting, locking) that adds latency and can cause unavailability during partitions. Many applications can tolerate staleness—showing an order as "processing" for a few seconds before updating to "confirmed" doesn't break functionality.

## System Design Questions

### Design a URL Shortener

**Core Requirements**:

- Shorten long URLs
- Redirect to original URL
- Track click analytics

**Database Schema**:

```sql
CREATE TABLE urls (
    id BIGINT PRIMARY KEY,
    short_code VARCHAR(10) UNIQUE NOT NULL,
    original_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    click_count INT DEFAULT 0,
    expires_at TIMESTAMP
);

CREATE INDEX idx_short_code ON urls(short_code);
CREATE INDEX idx_created_at ON urls(created_at);
```

**Scale Considerations**:

- 100M URLs: Single database can handle with proper indexing
- 1B+ URLs: Consider sharding by short_code hash
- Read-heavy: Add read replicas, cache frequently accessed URLs

**Interview Follow-up: How would you handle the redirect endpoint under high load?**

Answer: Use Redis/memcached for the hot URLs. Precompute short codes for new URLs and cache immediately. Consider CDN for static redirect responses. Use consistent hashing for sharding to maximize cache hits.

### Design a Rate Limiting System

**Architecture Components**:

- Counter storage (Redis or database)
- Sliding window algorithm
- Distributed locking for accurate counts

```python
# Redis-based rate limiter
def is_allowed(user_id: str, limit: int, window_seconds: int) -> bool:
    key = f"rate_limit:{user_id}"
    
    pipe = redis.pipeline()
    pipe.incr(key)
    pipe.expire(key, window_seconds)
    results = pipe.execute()
    
    current_count = results[0]
    return current_count <= limit
```

### Design a Search System

**Core Components**:

- Full-text search engine (Elasticsearch, Algolia)
- Database for persistence
- Caching layer

**Indexing Strategy**:

```json
{
  "mappings": {
    "properties": {
      "title": {"type": "text", "analyzer": "english"},
      "content": {"type": "text", "analyzer": "english"},
      "author": {"type": "keyword"},
      "created_at": {"type": "date"},
      "tags": {"type": "keyword"}
    }
  }
}
```

**Query Patterns**:

```json
{
  "query": {
    "bool": {
      "must": [
        {"match": {"title": "search query"}}
      ],
      "filter": [
        {"term": {"author": "john"}},
        {"range": {"created_at": {"gte": "2024-01-01"}}}
      ]
    }
  }
}
```

### Design an Analytics Pipeline

**Architecture**:

1. Ingest layer (Kafka, Kinesis)
2. Stream processing (Flink, Spark Streaming)
3. Storage layer (ClickHouse, BigQuery)
4. Visualization (Grafana, Looker)

**Schema Design**:

```sql
-- Fact table (events)
CREATE TABLE events (
    event_id UUID,
    event_type VARCHAR(50),
    user_id UUID,
    timestamp DATETIME,
    properties JSON
) PARTITION BY RANGE (timestamp);

-- Aggregation tables
CREATE MATERIALIZED VIEW hourly_stats AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    event_type,
    COUNT(*) as count,
    COUNT(DISTINCT user_id) as unique_users
FROM events
GROUP BY 1, 2;
```

## Practical Coding Challenges

### SQL Coding Problems

**Problem: Top 3 Salaries by Department**

```sql
SELECT 
    d.name as department,
    e.name as employee,
    e.salary
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE (
    SELECT COUNT(DISTINCT e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
    AND e2.salary > e.salary
) < 3
ORDER BY d.name, e.salary DESC;
```

**Problem: Consecutive Available Seats**

```sql
WITH numbered AS (
    SELECT 
        seat_id,
        is_available,
        seat_id - ROW_NUMBER() OVER (ORDER BY seat_id) as grp
    FROM seats
    WHERE is_available = true
)
SELECT 
    MIN(seat_id) as start_seat,
    MAX(seat_id) as end_seat,
    COUNT(*) as consecutive_count
FROM numbered
GROUP BY grp
HAVING COUNT(*) >= 3;
```

### Database Design Problems

**Problem: Design a Parking Lot System**

Key entities:

- ParkingLot (id, name, total_spots)
- Floor (id, lot_id, floor_number)
- Spot (id, floor_id, spot_type, status)
- Vehicle (id, license_plate, spot_id, entry_time)

Key queries:

- Find available spots
- Check-in vehicle
- Check-out vehicle and calculate fee
- Floor utilization report

### Transaction Problems

**Problem: Implement Bank Transfer with Race Condition Protection**

```python
# Without protection (vulnerable)
def transfer(from_account, to_account, amount):
    from_balance = get_balance(from_account)
    if from_balance >= amount:
        set_balance(from_account, from_balance - amount)
        set_balance(to_account, get_balance(to_account) + amount)

# With database transaction
def transfer(from_account, to_account, amount):
    with db.transaction():
        # Lock rows to prevent concurrent modification
        from_account = db.execute(
            "SELECT * FROM accounts WHERE id = ? FOR UPDATE",
            from_account
        ).fetchone()
        
        if from_account.balance >= amount:
            db.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ?",
                amount, from_account
            )
            db.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                amount, to_account
            )
```

## Behavioral and Experience Questions

### Common Questions

**"Tell me about a database performance problem you solved."**

Structure your answer with the STAR method:

- **Situation**: Describe the context—legacy system with slow queries
- **Task**: Your responsibility—improve query performance
- **Action**: What you did—analyzed execution plans, added composite index, denormalized reporting table
- **Result**: Measurable improvement—query time from 30 seconds to 200 milliseconds

**"How do you handle database schema changes in production?"**

Key points to cover:

- Backward-compatible changes first
- Use feature flags for schema changes
- Run migrations with proper backups
- Have rollback plan
- Test in staging environment
- Consider blue-green deployments

**"What's your experience with database security?"**

Topics to discuss:

- Principle of least privilege
- SQL injection prevention (parameterized queries)
- Encryption at rest and in transit
- Audit logging
- Regular security updates
- Penetration testing

## Interview Preparation Checklist

### Before the Interview

- [ ] Review SQL syntax and be able to write queries without autocomplete
- [ ] Understand ACID properties and isolation levels
- [ ] Know when to use indexes and how to choose columns
- [ ] Understand replication and sharding concepts
- [ ] Be familiar with CAP theorem and distributed trade-offs
- [ ] Practice system design questions
- [ ] Review your past projects involving databases

### During the Interview

- [ ] Ask clarifying questions before writing code
- [ ] Think out loud—explain your reasoning
- [ ] Consider edge cases and error handling
- [ ] Discuss trade-offs when asked
- [ ] If stuck, ask for hints
- [ ] Test your solution with sample inputs

### Questions to Ask the Interviewer

- [ ] What's the primary database used?
- [ ] How do you handle database scaling?
- [ ] What's the biggest database challenge the team faces?
- [ ] How are schema changes handled?
- [ ] What's the team's approach to data modeling?

## Resources for Further Study

### Books

- "Database Internals" by Alex Petrov - Deep dive into storage engines
- "The Art of Database Design" - Conceptual database design
- "High Performance MySQL" - MySQL optimization (concepts apply broadly)

### Practice Platforms

- LeetCode Database section
- HackerRank SQL challenges
- Mode Analytics SQL Tutorial
- PostgreSQL Exercises

### Documentation

- PostgreSQL documentation (excellent for learning SQL)
- MySQL documentation
- Distributed systems papers (Google's Spanner, Amazon's Dynamo)

This guide provides comprehensive preparation for database interviews across experience levels. Focus on understanding concepts deeply rather than memorizing—interviewers value genuine understanding over rote memorization.

## See Also

- [Database Architecture Patterns](../03_system_design/database_architecture_patterns.md) - CQRS, Event Sourcing, Saga patterns
- [Database Troubleshooting](../04_tutorials/troubleshooting/02_database_troubleshooting_debugging.md) - Practical debugging skills
- [Real-time Streaming Patterns](../02_intermediate/05_realtime_streaming_database_patterns.md) - CDC and streaming
- [Database Selection Framework](../01_foundations/database_selection_framework.md) - Decision-making framework
