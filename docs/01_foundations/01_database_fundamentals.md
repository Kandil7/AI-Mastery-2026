# Database Fundamentals

This document provides a comprehensive introduction to database fundamentals, covering ACID properties, transaction management, database types, indexing strategies, and query processing basics. This knowledge is essential for building robust, scalable AI applications that rely on persistent data storage.

---

## Table of Contents

1. [ACID Properties and Transactions](#1-acid-properties-and-transactions)
2. [Database Types](#2-database-types)
3. [Database Engines and Storage Architectures](#3-database-engines-and-storage-architectures)
4. [Indexing Strategies](#4-indexing-strategies)
5. [Query Processing Basics](#5-query-processing-basics)
6. [Concurrency Control](#6-concurrency-control)
7. [First Steps: Guided Exercises](#7-first-steps-guided-exercises)
8. [Beginner-Friendly Glossary](#8-beginner-friendly-glossary)

---

## 1. ACID Properties and Transactions

The ACID properties form the foundation of reliable database transaction processing. These four properties ensure that database operations are processed reliably, maintaining data integrity even in the face of system failures, power outages, or concurrent access.

### Atomicity

Atomicity guarantees that a transaction is treated as a single, indivisible unit of work. Either all operations within a transaction complete successfully, or none of them do. There is no partial completion—if any operation fails, the entire transaction is rolled back, leaving the database in its original state.

**Why It Matters**: Without atomicity, a system crash mid-operation could leave data in an inconsistent state. For example, in a financial transfer, money could be deducted from one account but never credited to another.

**Example - Financial Transfer**:

```sql
START TRANSACTION;

-- Debit from source account
UPDATE accounts
SET balance = balance - 1000
WHERE account_id = 'ACC001';

-- Credit to destination account
UPDATE accounts
SET balance = balance + 1000
WHERE account_id = 'ACC002';

-- If both operations succeed, commit the transaction
COMMIT;
```

In this example, if the credit operation fails after the debit has succeeded, atomicity ensures that the debit is also rolled back, preventing money from disappearing from the system.

### Consistency

Consistency ensures that a transaction transforms the database from one valid state to another valid state. All database constraints, triggers, and rules must be satisfied before and after the transaction completes.

**Why It Matters**: Consistency ensures data integrity by enforcing rules. If a transaction would violate any constraint (foreign key, unique, check, etc.), it is rolled back rather than leaving the database in an invalid state.

**Example - Enforcing Constraints**:

```sql
-- Create orders table with foreign key constraint
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    order_date DATE DEFAULT CURRENT_DATE,
    total_amount DECIMAL(10, 2) NOT NULL,

    -- This constraint ensures consistency
    CONSTRAINT fk_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);

-- This constraint ensures positive amounts
ALTER TABLE orders
ADD CONSTRAINT chk_positive_amount
CHECK (total_amount > 0);
```

### Isolation

Isolation ensures that concurrent transactions execute as if they were sequential, preventing interference between transactions. However, strict isolation can impact performance, leading to different isolation levels that trade off consistency for performance.

**Isolation Levels**:

| Isolation Level | Dirty Reads | Non-Repeatable Reads | Phantom Reads |
|-----------------|--------------|----------------------|---------------|
| READ UNCOMMITTED | Possible | Possible | Possible |
| READ COMMITTED | Prevented | Possible | Possible |
| REPEATABLE READ | Prevented | Prevented | Possible |
| SERIALIZABLE | Prevented | Prevented | Prevented |

**Understanding the Phenomena**:

- **Dirty Reads**: Reading uncommitted data from another transaction
- **Non-Repeatable Reads**: Getting different results when reading the same row twice within a transaction
- **Phantom Reads**: Seeing new rows that were inserted by other transactions

**Example - Setting Isolation Level**:

```sql
-- Set isolation level for a transaction
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- Or set it for the entire session
SET SESSION DEFAULT TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

### Durability

Durability guarantees that once a transaction is committed, its effects persist even if the system crashes or loses power. This is typically achieved through write-ahead logging (WAL).

**Why It Matters**: Without durability, committed transactions could be lost in a system failure, leading to data loss.

**PostgreSQL Example**:

```sql
-- Ensure synchronous commit for durability
ALTER SYSTEM SET synchronous_commit = on;

-- Check current setting
SHOW synchronous_commit;
```

---

## 2. Database Types

Understanding the different types of databases and their use cases is crucial for selecting the right technology for your AI application.

### Relational Databases (SQL)

Relational databases organize data into tables with rows and columns, using SQL as the query language. They enforce schemas and support complex queries with joins, aggregations, and subqueries.

**Best For**:
- Structured data with complex relationships
- Financial transactions requiring ACID guarantees
- Reporting and analytics
- Applications requiring strict data integrity

**Common Examples**: PostgreSQL, MySQL, MariaDB, Oracle, SQL Server

**Example Query - Customer Analytics**:

```sql
SELECT
    c.customer_id,
    c.name,
    c.email,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS lifetime_value,
    AVG(o.total_amount) AS avg_order_value
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.status = 'active'
GROUP BY c.customer_id, c.name, c.email
ORDER BY lifetime_value DESC
LIMIT 10;
```

### NoSQL Databases

NoSQL databases sacrifice some ACID properties for flexibility and scalability. They are designed to handle unstructured or semi-structured data, scale horizontally, and provide high performance for specific workloads.

#### Document Databases

Document databases store data as flexible JSON-like documents, allowing variable fields within documents. Each document can have a different structure, making them ideal for evolving schemas.

**Best For**: Content management, catalogs, user profiles, IoT data

**Common Examples**: MongoDB, CouchDB, Amazon DocumentDB

**Example - MongoDB Document**:

```javascript
{
  "_id": "user123",
  "name": "John Doe",
  "email": "john@example.com",
  "preferences": {
    "theme": "dark",
    "notifications": true,
    "language": "en"
  },
  "orders": [
    {"order_id": "ORD001", "total": 150.00},
    {"order_id": "ORD002", "total": 89.99}
  ],
  "created_at": ISODate("2024-01-15T10:30:00Z")
}
```

#### Key-Value Stores

Key-value stores provide the simplest data model—each key maps to a value. They offer extremely fast lookups and are often used for caching, session storage, and rate limiting.

**Best For**: Caching, session storage, real-time analytics, rate limiting

**Common Examples**: Redis, Amazon DynamoDB, etcd

**Example - Redis Operations**:

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, db=0)

# Store a user profile
user_profile = {
    'name': 'John',
    'age': 30,
    'preferences': {'theme': 'dark'}
}
r.set('user:123:profile', json.dumps(user_profile))

# Retrieve the profile
profile_data = r.get('user:123:profile')
profile = json.loads(profile_data)

# Store session with expiration
r.setex('session:abc123', 3600, json.dumps({'user_id': 123}))
```

#### Column-Family Databases

Column-family databases store data in columns rather than rows, optimized for read/write of large datasets. They are designed for high write throughput and can handle massive scale.

**Best For**: Time-series data, IoT, analytics, write-heavy workloads

**Common Examples**: Apache Cassandra, Apache HBase, Amazon Keyspaces

**Example - Cassandra CQL**:

```sql
CREATE TABLE analytics.events (
    event_id timeuuid,
    user_id text,
    event_type text,
    properties map<text, text>,
    timestamp timestamp,
    PRIMARY KEY (user_id, event_type, timestamp)
) WITH CLUSTERING ORDER BY (event_type ASC, timestamp DESC);

-- Insert an event
INSERT INTO analytics.events (event_id, user_id, event_type, properties, timestamp)
VALUES (now(), 'user123', 'purchase', {'amount': '99.99', 'currency': 'USD'}, toTimestamp(now()));
```

#### Graph Databases

Graph databases are optimized for data with complex relationships, using nodes, edges, and properties. They excel at traversing relationships efficiently.

**Best For**: Social networks, recommendation engines, fraud detection, network analysis

**Common Examples**: Neo4j, Amazon Neptune, Apache AGE

**Example - Neo4j Cypher**:

```cypher
-- Create nodes and relationships
CREATE (alice:Person {name: 'Alice', age: 30})
CREATE (bob:Person {name: 'Bob', age: 25})
CREATE (alice)-[:KNOWS {since: 2020}]->(bob)

-- Find friends of friends (recommendation)
MATCH (person:Person {name: 'Alice'})-[:KNOWS]->(friend)-[:KNOWS]->(friendOfFriend)
WHERE NOT (person)-[:KNOWS]->(friendOfFriend)
RETURN DISTINCT friendOfFriend.name AS suggested_friend,
       COUNT(*) AS mutual_friends
ORDER BY mutual_friends DESC
```

### NewSQL Databases

NewSQL systems aim to provide ACID guarantees with NoSQL scalability. They distribute data across multiple nodes while maintaining strong consistency.

**Best For**: Distributed applications requiring strong consistency, geo-distributed systems, high-scale OLTP

**Common Examples**: CockroachDB, Google Spanner, TiDB, YugabyteDB

**Example - CockroachDB with Regional Distribution**:

```sql
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
) WITH (regional_by_row = true);

-- Query with locality hints
SELECT * FROM orders@regional WHERE customer_id = 'abc123';
```

### Time-Series Databases

Time-series databases are optimized for storing and querying time-stamped data points. They excel at handling high write throughput and efficient time-range queries.

**Best For**: IoT sensors, metrics monitoring, financial tickers, application logs

**Common Examples**: InfluxDB, TimescaleDB, Prometheus, QuestDB

**Example - TimescaleDB Hypertable**:

```sql
-- Create a hypertable for time-series data
CREATE TABLE measurements (
    time TIMESTAMPTZ NOT NULL,
    device_id INT NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    PRIMARY KEY (device_id, time)
);

-- Convert to hypertable for automatic partitioning
SELECT create_hypertable('measurements', 'time');

-- Create continuous aggregate for downsampling
CREATE MATERIALIZED VIEW hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    device_id,
    AVG(temperature) AS avg_temp,
    MAX(temperature) AS max_temp,
    MIN(temperature) AS min_temp
FROM measurements
GROUP BY bucket, device_id;
```

### Vector Databases

Vector databases are designed for storing and searching high-dimensional vector embeddings, essential for AI applications like semantic search, similarity matching, and retrieval-augmented generation (RAG).

**Best For**: AI/ML embeddings, semantic search, similarity matching, RAG applications

**Common Examples**: Pinecone, Weaviate, Milvus, pgvector, Chroma

**Example - pgvector (PostgreSQL Extension)**:

```sql
-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    embedding vector(1536)  -- OpenAI embeddings are 1536 dimensions
);

-- Create index for fast similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Query for similar documents
SELECT id, title,
       1 - (embedding <=> $1) AS similarity
FROM documents
ORDER BY embedding <=> $1
LIMIT 5;
```

**Example - Pinecone Python Client**:

```python
from pinecone import Pinecone

# Initialize client
pc = Pinecone(api_key="your-api-key")
index = pc.Index("documents")

# Upsert vectors with metadata
vectors = [
    {
        "id": "doc1",
        "values": [0.1, 0.3, 0.5, ...],  # 1536-dimensional embedding
        "metadata": {"title": "Introduction to Databases", "category": "technology"}
    },
    {
        "id": "doc2",
        "values": [0.2, 0.4, 0.6, ...],
        "metadata": {"title": "Machine Learning Basics", "category": "ai"}
    }
]
index.upsert(vectors=vectors)

# Query for similar documents
results = index.query(
    vector=[0.1, 0.3, 0.5, ...],
    top_k=3,
    include_metadata=True,
    filter={"category": {"$eq": "technology"}}
)
```

---

## 3. Database Engines and Storage Architectures

Understanding how databases store data internally helps in making better design decisions and optimizing performance.

### Row-Oriented Storage

Row-oriented storage stores complete rows together on disk. This format is optimal for transactional workloads (OLTP) where you typically read or write entire records.

**Advantages**:
- Fast writes (single row at a time)
- Efficient for point queries and full row retrieval
- Good for workloads with many columns

**Common Engines**: InnoDB (MySQL), PostgreSQL default, SQL Server

### Column-Oriented Storage

Column-oriented storage stores data column by column on disk. This format is optimized for analytical workloads (OLAP) where you often aggregate values across many rows.

**Advantages**:
- Efficient for aggregations (SUM, AVG, COUNT)
- Better compression ratios
- Only读取需要的列

**Common Engines**: ClickHouse, Amazon Redshift, Google BigQuery, Apache Druid

### Buffer Pool and Caching

Modern databases use buffer pools to keep frequently accessed data in memory, dramatically improving read performance.

**PostgreSQL Example**:

```sql
-- Configure shared buffer size
ALTER SYSTEM SET shared_buffers = '4GB';

-- Check buffer hit ratio
SELECT
    blks_hit::float / NULLIF(blks_hit + blks_read, 0) * 100 AS hit_ratio
FROM pg_stat_database
WHERE datname = current_database();
```

### Write-Ahead Logging (WAL)

WAL ensures durability by writing changes to a transaction log before applying them to data files. This allows for fast recovery after crashes.

**PostgreSQL Example**:

```sql
-- Set WAL level
ALTER SYSTEM SET wal_level = 'replica';

-- Check WAL status
SELECT * FROM pg_wal;
```

---

## 4. Indexing Strategies

Indexes are data structures that speed up data retrieval. Understanding different index types helps you choose the right one for your query patterns.

### B-Tree Indexes

B-Tree (Balanced Tree) indexes are the most common index type, supporting efficient range queries, equality comparisons, and sorted results.

**Best For**: Most use cases, especially when querying ranges or sorting

```sql
-- Simple B-Tree index
CREATE INDEX idx_users_email ON users(email);

-- Composite index (covering multiple columns)
CREATE INDEX idx_orders_customer_date
ON orders(customer_id, order_date DESC);

-- Partial index (only for active records)
CREATE INDEX idx_active_orders
ON orders(customer_id)
WHERE status = 'active';
```

### Hash Indexes

Hash indexes provide O(1) lookup time for equality comparisons but cannot handle range queries.

**Best For**: Exact match lookups only

```sql
CREATE INDEX idx_sessions_token
USING hash (session_token);
```

### GIN Indexes

GIN (Generalized Inverted Index) indexes are ideal for full-text search, arrays, and JSONB columns.

**Best For**: Full-text search, array columns, JSONB data

```sql
-- Index for array columns
CREATE INDEX idx_products_tags
ON products USING GIN (tags);

-- Index for JSONB
CREATE INDEX idx_orders_metadata
ON orders USING GIN (metadata);

-- Full-text search index
CREATE INDEX idx_articles_content
ON articles
USING GIN (to_tsvector('english', content));
```

### BRIN Indexes

BRIN (Block Range Index) indexes are extremely compact and efficient for naturally ordered data like time-series.

**Best For**: Time-series data, append-only tables, very large datasets

```sql
CREATE INDEX idx_measurements_time
ON measurements USING BRIN (recorded_at);
```

### Index Maintenance

Indexes need maintenance as data changes:

```sql
-- Check unused indexes
SELECT indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0;

-- Rebuild fragmented index
REINDEX INDEX CONCURRENTLY idx_orders_customer_id;

-- Analyze table to update statistics
ANALYZE orders;
```

---

## 5. Query Processing Basics

Understanding how queries are executed helps you write more efficient SQL and diagnose performance issues.

### Reading Execution Plans

The `EXPLAIN` command shows how the database will execute your query, including which indexes it uses, join methods, and estimated costs.

```sql
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT customer_id, COUNT(*)
FROM orders
WHERE order_date >= '2024-01-01'
GROUP BY customer_id;
```

**Example Output**:

```
HashAggregate  (cost=1234.56..1234.60 rows=10 width=24) (actual time=15.234..15.238 rows=50 loops=1)
  ->  Index Scan using idx_orders_date on orders  (cost=0.45..1230.00 rows=1000 width=8) (actual time=0.123..10.456 rows=5000 loops=1)
        Index Cond: (order_date >= '2024-01-01'::date)
Planning Time: 0.456 ms
Execution Time: 15.678 ms
```

### Common Plan Operations

| Operation | Description | When Used |
|-----------|-------------|-----------|
| Seq Scan | Full table scan | No suitable index, small tables |
| Index Scan | Index-based retrieval | Selective queries |
| Index Only Scan | Index covers all needed columns | Covered queries |
| Nested Loop | Join by iterating rows | Small tables, indexed joins |
| Hash Join | Build hash table for join | Large table joins |
| Merge Join | Sorted join | Already sorted inputs |

### Query Optimization Tips

1. **Select Only Needed Columns**: Avoid `SELECT *`

```sql
-- Bad
SELECT * FROM orders WHERE customer_id = 123;

-- Good
SELECT order_id, order_date, total_amount
FROM orders
WHERE customer_id = 123;
```

2. **Use Parameterized Queries**: Prevents SQL injection and allows query plan caching

```python
# Bad - SQL injection vulnerable
query = f"SELECT * FROM users WHERE username = '{user_input}'"

# Good - Parameterized query
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, [user_input])
```

3. **Understand Index Usage**

```sql
-- Index is used (leading column in composite)
SELECT * FROM orders WHERE customer_id = 123;

-- Index NOT used (not leading column)
SELECT * FROM orders WHERE order_date = '2024-01-01';
```

---

## 6. Concurrency Control

Databases must handle multiple concurrent transactions while maintaining data integrity.

### Lock Types

- **Shared (S) Lock**: Allows reading, blocks writing
- **Exclusive (X) Lock**: Blocks both reading and writing

```sql
-- Set lock timeout to prevent indefinite waiting
SET lock_timeout = '5s';

-- Explicit lock
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- ... perform operations ...
COMMIT;
```

### Optimistic Concurrency Control (OCC)

OCC uses version numbers to detect conflicts. Transactions read data without locking, then verify the version hasn't changed before committing.

```sql
-- Update with version check
UPDATE products
SET price = 29.99, version = version + 1
WHERE id = 1 AND version = 5;

-- If version changed, 0 rows affected - retry needed
```

### MVCC (Multi-Version Concurrency Control)

MVCC allows readers to see consistent snapshots without blocking writers. Each transaction sees a snapshot of the database at a point in time.

```sql
-- PostgreSQL: See data as of a specific time
SELECT * FROM orders AS OF SYSTEM TIME '2024-01-01 10:00:00';
```

---

## 7. First Steps: Guided Exercises

Complete these hands-on exercises to solidify your understanding of database fundamentals. These exercises build on the concepts covered in the previous sections.

### Exercise 1: ACID Properties Lab

**Objective**: Understand how ACID properties protect data integrity.

1. Create a simple bank account table:
```sql
CREATE TABLE accounts (
    account_id VARCHAR(10) PRIMARY KEY,
    balance DECIMAL(10,2) NOT NULL CHECK (balance >= 0)
);
```

2. Insert two accounts:
```sql
INSERT INTO accounts VALUES ('ACC001', 1000.00), ('ACC002', 500.00);
```

3. Simulate a transfer with explicit transaction:
```sql
BEGIN;
-- Try to transfer $200 from ACC001 to ACC002
UPDATE accounts SET balance = balance - 200 WHERE account_id = 'ACC001';
-- Intentionally cause an error (e.g., divide by zero)
SELECT 1/0;
ROLLBACK;
```

4. Check balances - they should remain unchanged due to atomicity.

5. Now try without transaction:
```sql
UPDATE accounts SET balance = balance - 200 WHERE account_id = 'ACC001';
-- Crash the system or kill the connection
-- Check balances afterward - notice inconsistency!
```

### Exercise 2: Constraint Enforcement

**Objective**: Practice creating and testing database constraints.

1. Create a products table with various constraints:
```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL CHECK (price > 0),
    category VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (name, category)  -- Prevent duplicate product names in same category
);
```

2. Test constraint violations:
   - Try inserting a product with negative price
   - Try inserting duplicate product name in same category
   - Try inserting NULL name

3. Add a foreign key constraint to orders table:
```sql
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

4. Test referential integrity:
   - Try inserting order with non-existent product_id
   - Try deleting product that has orders

### Exercise 3: Index Performance Comparison

**Objective**: Measure the impact of indexing on query performance.

1. Create a large table (100,000 rows):
```sql
CREATE TABLE large_table (
    id SERIAL PRIMARY KEY,
    category VARCHAR(20) NOT NULL,
    value INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO large_table (category, value)
SELECT 
    CASE WHEN i % 3 = 0 THEN 'A'
         WHEN i % 3 = 1 THEN 'B'
         ELSE 'C' END,
    i % 1000
FROM generate_series(1, 100000) AS i;
```

2. Time a query without index:
```sql
EXPLAIN ANALYZE SELECT COUNT(*) FROM large_table WHERE category = 'A';
```

3. Create an index and time again:
```sql
CREATE INDEX idx_large_table_category ON large_table(category);
EXPLAIN ANALYZE SELECT COUNT(*) FROM large_table WHERE category = 'A';
```

4. Compare execution times and plan operations.

### Exercise 4: Concurrency Simulation

**Objective**: Observe concurrency control in action.

1. Create a simple counter table:
```sql
CREATE TABLE counters (
    name VARCHAR(50) PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
INSERT INTO counters VALUES ('hits', 0);
```

2. In two separate database sessions:
   - Session 1: `BEGIN; SELECT value FROM counters WHERE name = 'hits' FOR UPDATE;`
   - Session 2: `SELECT value FROM counters WHERE name = 'hits';` (should wait)
   - Session 1: `UPDATE counters SET value = value + 1 WHERE name = 'hits'; COMMIT;`
   - Session 2: Should now complete with correct value

3. Try optimistic concurrency:
```sql
-- Add version column
ALTER TABLE counters ADD COLUMN version INTEGER DEFAULT 0;

-- Update with version check
UPDATE counters 
SET value = value + 1, version = version + 1 
WHERE name = 'hits' AND version = 0;
```

### Exercise 5: Query Optimization Challenge

**Objective**: Apply optimization techniques to improve query performance.

Given this slow query:
```sql
SELECT 
    c.customer_id,
    c.name,
    COUNT(o.order_id) as order_count,
    SUM(o.total_amount) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE c.created_at > '2023-01-01'
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 10;
```

1. Identify potential bottlenecks
2. Create appropriate indexes
3. Rewrite the query for better performance
4. Measure before/after performance

---

## 8. Beginner-Friendly Glossary

Here's a simple glossary of essential database terms for beginners:

### Core Concepts
- **Database**: A structured collection of data stored electronically
- **Table**: A collection of related data organized in rows and columns
- **Row/Record**: A single entry in a table (one instance of data)
- **Column/Field**: A specific attribute or characteristic of the data
- **Primary Key**: A unique identifier for each row in a table
- **Foreign Key**: A field that links to the primary key of another table
- **Schema**: The structure of a database (tables, columns, relationships)

### Operations
- **CRUD**: Create, Read, Update, Delete - the four basic data operations
- **Query**: A request for data from the database
- **Transaction**: A sequence of operations treated as a single unit
- **Commit**: Make transaction changes permanent
- **Rollback**: Undo transaction changes

### Data Integrity
- **ACID**: Atomicity, Consistency, Isolation, Durability - properties ensuring reliable transactions
- **Constraint**: A rule that enforces data validity (NOT NULL, UNIQUE, CHECK, FOREIGN KEY)
- **Normalization**: Process of organizing data to reduce redundancy
- **Index**: A data structure that speeds up data retrieval

### Performance Terms
- **Index**: Like a book's index - helps find data quickly
- **Join**: Combining data from multiple tables
- **Query Plan**: The strategy the database uses to execute a query
- **Buffer Pool**: Memory area where frequently accessed data is cached
- **WAL (Write-Ahead Logging)**: Technique to ensure data durability

### Database Types
- **SQL/Relational**: Tables with defined relationships (PostgreSQL, MySQL)
- **NoSQL**: Flexible data models (MongoDB, Redis, Cassandra)
- **Document Database**: Stores data as JSON-like documents
- **Key-Value Store**: Simple pair of key and value (like a dictionary)
- **Graph Database**: Optimized for relationship-based data

### Common Abbreviations
- **OLTP**: Online Transaction Processing (transactional workloads)
- **OLAP**: Online Analytical Processing (analytical workloads)
- **ETL**: Extract, Transform, Load (data integration process)
- **RDBMS**: Relational Database Management System
- **DDL**: Data Definition Language (CREATE, ALTER, DROP)
- **DML**: Data Manipulation Language (SELECT, INSERT, UPDATE, DELETE)

> 💡 **Pro Tip**: Keep this glossary handy as you work through the documentation. Refer back to it whenever you encounter unfamiliar terms.

---

## Related Resources

- For practical database tutorials, see [04. Tutorials](../04_tutorials/README.md)
- For advanced query optimization, see [Query Optimization Deep Dive](./query_optimization_deep_dive.md)
- For database design patterns, see [Database Design](./database_design.md)
- For time-series databases, see [Time-Series Fundamentals](./time_series_fundamentals.md)
- For vector databases, see [Vector Search Basics](./vector_search_basics.md)