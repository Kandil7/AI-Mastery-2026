# Database Types

Understanding the different types of databases and their use cases is crucial for selecting the right technology for your AI application. This document covers the major database categories and their characteristics.

## Overview

Database systems can be broadly categorized based on their data model, consistency guarantees, scalability characteristics, and intended use cases. Senior AI/ML engineers need to understand these differences to make informed architectural decisions.

## Relational Databases (SQL)

Relational databases organize data into tables with rows and columns, using SQL as the query language. They enforce schemas and support complex queries with joins, aggregations, and subqueries.

### Key Characteristics
- **ACID compliance**: Strong consistency guarantees
- **Schema enforcement**: Strict data typing and constraints
- **Complex queries**: Support for joins, aggregations, subqueries
- **Mature ecosystem**: Extensive tooling and community support

### Best For
- Structured data with complex relationships
- Financial transactions requiring ACID guarantees
- Reporting and analytics
- Applications requiring strict data integrity

### Common Examples
- PostgreSQL (open-source, feature-rich)
- MySQL/MariaDB (widely adopted, good performance)
- Oracle (enterprise-grade, high availability)
- SQL Server (Microsoft ecosystem integration)

### Example Query - Customer Analytics

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

## NoSQL Databases

NoSQL databases sacrifice some ACID properties for flexibility and scalability. They are designed to handle unstructured or semi-structured data, scale horizontally, and provide high performance for specific workloads.

### Document Databases

Store data as flexible JSON-like documents, allowing variable fields within documents.

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

### Key-Value Stores

Provide the simplest data model—each key maps to a value. Offer extremely fast lookups and are often used for caching, session storage, and rate limiting.

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

### Column-Family Databases

Store data in columns rather than rows, optimized for read/write of large datasets. Designed for high write throughput and massive scale.

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
```

### Graph Databases

Optimized for data with complex relationships, using nodes, edges, and properties. Excel at traversing relationships efficiently.

**Best For**: Social networks, recommendation engines, fraud detection, network analysis

**Common Examples**: Neo4j, Amazon Neptune, Apache AGE

**Example - Neo4j Cypher**:
```cypher
-- Create nodes and relationships
CREATE (alice:Person {name: 'Alice', age: 30})
CREATE (bob:Person {name: 'Bob', age: 25})
CREATE (alice)-[:KNOWS {since: 2020}]->(bob)

-- Find friends of friends
MATCH (person:Person {name: 'Alice'})-[:KNOWS]->(friend)-[:KNOWS]->(friendOfFriend)
WHERE NOT (person)-[:KNOWS]->(friendOfFriend)
RETURN DISTINCT friendOfFriend.name AS suggested_friend,
       COUNT(*) AS mutual_friends
ORDER BY mutual_friends DESC
```

## NewSQL Databases

NewSQL systems aim to provide ACID guarantees with NoSQL scalability. They distribute data across multiple nodes while maintaining strong consistency.

### Key Characteristics
- **Distributed architecture**: Horizontal scaling
- **Strong consistency**: ACID compliance across nodes
- **High performance**: Optimized for OLTP workloads
- **Geo-distribution**: Multi-region deployment capabilities

### Best For
- Distributed applications requiring strong consistency
- Geo-distributed systems
- High-scale OLTP workloads

### Common Examples
- CockroachDB (PostgreSQL-compatible, distributed)
- Google Spanner (globally distributed, strong consistency)
- TiDB (MySQL-compatible, distributed)
- YugabyteDB (PostgreSQL-compatible, distributed)

### Example - CockroachDB with Regional Distribution
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

## Time-Series Databases

Optimized for storing and querying time-stamped data points. Excel at handling high write throughput and efficient time-range queries.

### Key Characteristics
- **Time-based partitioning**: Automatic data organization by time
- **High write throughput**: Optimized for append-only workloads
- **Efficient range queries**: Fast time-window operations
- **Downsampling capabilities**: Built-in aggregation for historical data

### Best For
- IoT sensors
- Metrics monitoring
- Financial tickers
- Application logs

### Common Examples
- InfluxDB (purpose-built time-series)
- TimescaleDB (PostgreSQL extension)
- Prometheus (monitoring-focused)
- QuestDB (high-performance, SQL-based)

### Example - TimescaleDB Hypertable
```sql
-- Create a hypertable
CREATE TABLE measurements (
    time TIMESTAMPTZ NOT NULL,
    device_id INT NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    PRIMARY KEY (device_id, time)
);

-- Convert to hypertable for automatic partitioning
SELECT create_hypertable('measurements', 'time');

-- Create continuous aggregate
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

## Vector Databases

Designed for storing and searching high-dimensional vector embeddings, essential for AI applications like semantic search, similarity matching, and retrieval-augmented generation (RAG).

### Key Characteristics
- **Vector indexing**: Specialized indexes for similarity search
- **High-dimensional support**: Optimized for 100s-1000s dimensions
- **Similarity metrics**: Cosine, Euclidean, Manhattan distance
- **Scalable search**: Approximate nearest neighbor algorithms

### Best For
- AI/ML embeddings storage
- Semantic search
- Similarity matching
- RAG applications

### Common Examples
- Pinecone (managed service)
- Weaviate (open-source, hybrid search)
- Milvus (open-source, scalable)
- pgvector (PostgreSQL extension)
- Chroma (lightweight, Python-native)

### Example - pgvector (PostgreSQL Extension)
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

## Choosing the Right Database for AI/ML Applications

### Decision Framework

1. **Data Structure**
   - Structured, relational data → Relational databases
   - Flexible, evolving schema → Document databases
   - Time-series data → Time-series databases
   - Vector embeddings → Vector databases
   - Complex relationships → Graph databases

2. **Consistency Requirements**
   - Strong consistency required → Relational/NewSQL
   - Eventual consistency acceptable → NoSQL

3. **Scale Requirements**
   - Vertical scaling sufficient → Traditional RDBMS
   - Horizontal scaling needed → NewSQL/NoSQL

4. **Query Patterns**
   - Complex joins/aggregations → Relational
   - Simple key-value lookups → Key-value stores
   - Similarity search → Vector databases
   - Relationship traversal → Graph databases

## Related Resources

- [ACID Properties] - Understanding transaction fundamentals
- [Storage Architectures] - How different databases store data internally
- [Indexing Strategies] - Optimizing query performance across database types
- [AI/ML Integration Patterns] - Database patterns specifically for AI applications