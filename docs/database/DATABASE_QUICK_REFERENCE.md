# Database Quick Reference Guide

This guide provides quick command syntax, common patterns, decision trees, and configuration templates for database operations. Use this as a handy reference for daily tasks.

---

## Table of Contents

1. [Database Selection](#database-selection)
2. [SQL Quick Reference](#sql-quick-reference)
3. [Common Patterns](#common-patterns)
4. [Decision Trees](#decision-trees)
5. [Configuration Templates](#configuration-templates)
6. [Connection Strings](#connection-strings)
7. [Monitoring Commands](#monitoring-commands)
8. [Troubleshooting Quick Fixes](#troubleshooting-quick-fixes)

---

## Database Selection

### Quick Selection Guide

| Need | Recommended Database | Alternative |
|------|---------------------|-------------|
| ACID transactions | PostgreSQL | MySQL |
| Document storage | MongoDB | CouchDB |
| Key-value cache | Redis | Memcached |
| Time-series data | TimescaleDB | InfluxDB |
| Graph relationships | Neo4j | Amazon Neptune |
| Vector similarity search | Qdrant | Pinecone |
| Distributed SQL | CockroachDB | Google Spanner |
| Full-text search | Elasticsearch | OpenSearch |
| Analytics warehouse | Snowflake | BigQuery |

### Cloud Service Quick Reference

| Provider | Relational | NoSQL | Cache | Vector |
|----------|-----------|-------|-------|--------|
| **AWS** | RDS, Aurora | DynamoDB | ElastiCache | OpenSearch |
| **GCP** | Cloud SQL | Firestore | Memorystore | Vertex AI |
| **Azure** | SQL Database | Cosmos DB | Azure Cache | Azure AI |

---

## SQL Quick Reference

### Basic CRUD Operations

```sql
-- Create table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert
INSERT INTO users (email) VALUES ('user@example.com');
INSERT INTO users (email) VALUES ('a@b.com'), ('c@d.com');

-- Select
SELECT * FROM users;
SELECT id, email FROM users WHERE id > 10;
SELECT COUNT(*) FROM users;

-- Update
UPDATE users SET email = 'new@example.com' WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 1;
```

### Joins

```sql
-- Inner Join
SELECT * FROM orders o
INNER JOIN users u ON o.user_id = u.id;

-- Left Join
SELECT * FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Multiple Joins
SELECT u.name, o.total, p.product_name
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id;
```

### Aggregation

```sql
-- Basic aggregation
SELECT COUNT(*), SUM(total), AVG(total), MIN(total), MAX(total)
FROM orders;

-- Group by
SELECT user_id, COUNT(*) as order_count, SUM(total) as total_spent
FROM orders
GROUP BY user_id
HAVING COUNT(*) > 5;

-- Window functions
SELECT 
    name,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) as dept_avg,
    RANK() OVER (ORDER BY salary DESC) as salary_rank
FROM employees;
```

### Indexing

```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Composite index
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at DESC);

-- Unique index
CREATE UNIQUE INDEX idx_products_sku ON products(sku);

-- Partial index
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- GIN index for JSON
CREATE INDEX idx_data ON documents USING gin(data jsonb_path_ops);
```

### Common Patterns

```sql
-- Upsert (PostgreSQL)
INSERT INTO users (email, name) VALUES ('a@b.com', 'Alice')
ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name;

-- Upsert (MySQL)
INSERT INTO users (email, name) VALUES ('a@b.com', 'Alice')
ON DUPLICATE KEY UPDATE name = VALUES(name);

-- Recursive CTE
WITH RECURSIVE org_tree AS (
    SELECT id, name, manager_id, 1 as level
    FROM employees WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, ot.level + 1
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT * FROM org_tree;

-- Pagination
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 20;
-- Or with keyset pagination (faster)
SELECT * FROM users WHERE id > 20 ORDER BY id LIMIT 10;
```

---

## Common Patterns

### Read Replicas Pattern

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Primary DB │────▶│  Replica 1  │────▶│  Replica 2  │
│  (Write)    │     │  (Read)     │     │  (Read)     │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Use when**: Read-heavy workload (90%+ reads), can tolerate slight replication lag

### CQRS Pattern

```
┌─────────────┐     ┌─────────────┐
│    Write    │     │    Read    │
│   Commands  │────▶│   Models   │
└─────────────┘     └─────────────┘
        │                   ▲
        ▼                   │
┌─────────────┐     ┌─────────────┐
│   Write DB  │────▶│  Read DB   │
│  (Normalized)│     │(Denormalized)│
└─────────────┘     └─────────────┘
```

**Use when**: Different read/write patterns, complex queries, event sourcing

### Sharding Pattern

```
┌─────────────┐
│  Router/    │
│  Proxy      │
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐ ┌─────┐
│Shard│ │Shard│
│  0  │ │  1  │
└─────┘ └─────┘
```

**Use when**: Data exceeds single node capacity, need horizontal scaling

### Event Sourcing Pattern

```
┌─────────────┐     ┌─────────────┐
│   Events    │────▶│   Event     │
│  (Append)   │     │   Store     │
└─────────────┘     └──────┬──────┘
                          │
                          ▼
                 ┌─────────────┐
                 │ Projections │
                 │ (Materialized│
                 │    Views)    │
                 └─────────────┘
```

**Use when**: Need audit trail, complex state transitions, replay capability

### Change Data Capture (CDC)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Source DB  │────▶│    CDC      │────▶│  Downstream │
│             │     │  Connector  │     │   Systems   │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Use when**: Data replication, microservices, data pipelines

---

## Decision Trees

### Which Index Type?

```
Start
  │
  ▼
Is the column a primary key or unique?
  │
  ├─ Yes → Use PRIMARY KEY or UNIQUE index
  │
  ▼ (No)
Does the query use range operators (>, <, BETWEEN)?
  │
  ├─ Yes → Use B-Tree index
  │
  ▼ (No)
Is the column JSON or full-text?
  │
  ├─ JSON → Use GIN index
  │
  ├─ Full-text → Use GIN or GiST index with tsvector
  │
  ▼ (No)
Is this for exact match lookups?
  │
  ├─ Yes → Use B-Tree index
  │
  └─ Consider composite index with leading column
```

### Database Selection Decision

```
Start
  │
  ▼
Do you need ACID transactions?
  │
  ├─ Yes → Continue
  │
  └─ No → Consider NoSQL options
        │
        ▼
Is your data highly relational?
  │
  ├─ Yes → Relational (PostgreSQL/MySQL)
  │
  └─ No → Continue
        │
        ▼
What is your primary access pattern?
  │
  ├─ Documents → MongoDB/CouchDB
  │
  ├─ Key-Value → Redis/Memcached
  │
  ├─ Time-Series → TimescaleDB/InfluxDB
  │
  ├─ Graph → Neo4j
  │
  ├─ Vector → Qdrant/Pinecone
  │
  └─ Search → Elasticsearch
```

### Scaling Decision

```
Start
  │
  ▼
Current load?
  │
  ├─ < 1000 ops/sec → Single node + read replicas
  │
  ├─ 1000-10,000 → Vertical scaling + caching
  │
  ├─ 10,000-100,000 → Sharding needed
  │
  └─ > 100,000 → Distributed database
        │
        ▼
  Can you partition data?
  │
  ├─ Yes → Application-level sharding
  │
  └─ No → NewSQL (CockroachDB/Spanner)
```

---

## Configuration Templates

### PostgreSQL Connection Pool (PgBouncer)

```ini
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 5
max_db_connections = 100
```

### Redis Configuration

```redis
# Memory
maxmemory 4gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Replication
replica-read-only yes

# Cluster
cluster-enabled yes
cluster-config-file nodes.conf

# Performance
tcp-backlog 511
timeout 0
tcp-keepalive 300
```

### MongoDB Replica Set

```javascript
// rs.initiate()
{
  _id: "rs0",
  members: [
    { _id: 0, host: "mongo-primary:27017", priority: 2 },
    { _id: 1, host: "mongo-replica1:27017", priority: 1 },
    { _id: 2, host: "mongo-replica2:27017", priority: 1 }
  ]
}
```

### TimescaleDB Hypertable

```sql
-- Create hypertable
CREATE TABLE readings (
    time        TIMESTAMPTZ       NOT NULL,
    device_id   TEXT              NOT NULL,
    temperature DOUBLE PRECISION  NULL,
    humidity    DOUBLE PRECISION  NULL
);

SELECT create_hypertable('readings', 'time');

-- Add compression
ALTER TABLE readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id'
);

-- Add compression policy
SELECT add_compression_policy('readings', INTERVAL '7 days');
```

### Vector Database (Qdrant)

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

# Python client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)
client.create_collection(
    "vectors",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

---

## Connection Strings

### Common Formats

```bash
# PostgreSQL
postgresql://user:password@host:5432/database
postgresql+psycopg2://user:password@host:5432/database

# MySQL
mysql://user:password@host:3306/database
mysql+pymysql://user:password@host:3306/database

# MongoDB
mongodb://user:password@host:27017/database
mongodb+srv://cluster.mongodb.net/database

# Redis
redis://localhost:6379/0
redis://:password@host:6379/0

# Elasticsearch
https://elasticsearch:9200

# Qdrant (gRPC)
grpc://localhost:6334
```

### Environment Variables Template

```bash
# .env.database
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydb
DB_USER=user
DB_PASSWORD=password

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379

# MongoDB
MONGODB_URI=mongodb://localhost:27017/mydb

# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## Monitoring Commands

### PostgreSQL

```sql
-- Current connections
SELECT count(*) FROM pg_stat_activity;

-- Slow queries
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Table sizes
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;

-- Index usage
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Replication lag
SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;
```

### Redis

```bash
# Info
redis-cli INFO

# Memory
redis-cli INFO memory

# Clients
redis-cli CLIENT LIST

# Slow log
redis-cli SLOWLOG GET 10

# Monitor
redis-cli MONITOR
```

### MongoDB

```javascript
// Current operations
db.currentOp()

// Collection stats
db.collection.stats()

// Index stats
db.collection.getIndexes()

// Replica set status
rs.status()
```

---

## Troubleshooting Quick Fixes

### High CPU Usage

| Possible Cause | Fix |
|---------------|-----|
| Missing index | `CREATE INDEX` |
| Full table scan | `EXPLAIN` query, add index |
| Too many connections | Connection pooling |
| Long-running query | Kill query or add timeout |

```sql
-- Find slow queries
SELECT query, state, wait_event_type, wait_event
FROM pg_stat_activity
WHERE state != 'idle'
AND query_start < now() - interval '5 minutes';
```

### High Memory Usage

| Possible Cause | Fix |
|---------------|-----|
| Large result sets | Add LIMIT, pagination |
| Missing indexes | Create indexes |
| Connection leak | Check pooling config |
| Large cache | Increase memory or optimize |

### Connection Issues

| Error | Fix |
|-------|-----|
| Connection refused | Check port, firewall |
| Too many connections | Reduce pool size |
| Authentication failed | Check credentials |
| Timeout | Check network, increase timeout |

### Replication Lag

| Cause | Fix |
|-------|-----|
| Network issues | Check network |
| Heavy write load | Scale replicas |
| Long transactions | Break into smaller units |

---

## Additional Resources

- [Database Documentation Index](./DATABASE_DOCUMENTATION_INDEX.md) - Comprehensive documentation
- [Database Learning Path](../01_learning_roadmap/database_learning_path.md) - Structured learning
- [PostgreSQL Tutorial](../04_tutorials/tutorial_postgresql_basics.md) - Hands-on guide
- [Performance Tuning](./01_foundations/03_database_performance_tuning.md) - Optimization guide

---

*Last Updated: February 2026*
*Version: 1.0*
