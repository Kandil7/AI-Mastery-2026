# Database Selection Guide

## Quick Decision Matrix

| Requirement | First Choice | Alternative |
|-------------|--------------|-------------|
| **ACID transactions required** | PostgreSQL | CockroachDB |
| **Flexible schema** | MongoDB | PostgreSQL JSONB |
| **Massive write throughput** | Cassandra | ClickHouse |
| **Sub-millisecond cache** | Redis | Memcached |
| **Full-text search** | Elasticsearch | PostgreSQL FTS |
| **Complex relationships** | Neo4j | PostgreSQL |
| **Graph traversal** | Neo4j | ArangoDB |
| **Time-series data** | TimescaleDB | InfluxDB |
| **Analytics/Business Intelligence** | ClickHouse | BigQuery |
| **Vector similarity search** | Pinecone | Qdrant |
| **Geo-spatial queries** | PostgreSQL PostGIS | MongoDB Geospatial |
| **Simple key-value** | DynamoDB | Redis |
| **Multi-region global distribution** | CockroachDB | Spanner |
| **HTAP (mixed workloads)** | TiDB | SingleStore |

---

## Database Family Selection

### Relational (SQL)

**Choose when:**
- ACID compliance is mandatory
- Complex queries with joins
- Structured data with fixed schema
- Reporting and analytics on transactional data

**Popular options:**
- PostgreSQL (open-source, most feature-rich)
- MySQL (widely used, web applications)
- CockroachDB (distributed, global)
- Snowflake (cloud analytics)

**Avoid when:**
- Schema is highly variable
- Write throughput > 100K/second
- Simple key-value access pattern

---

### Document Store

**Choose when:**
- Data structure varies between entities
- Rapid prototyping needed
- Hierarchical data (JSON trees)
- High write throughput

**Popular options:**
- MongoDB (most popular, rich ecosystem)
- CouchDB (offline-first, sync)
- DynamoDB (AWS managed, auto-scale)

**Avoid when:**
- Complex transactions across documents
- Heavy analytical queries
- Strict schema requirements

---

### Key-Value Store

**Choose when:**
- Ultra-low latency required
- Simple lookup by known key
- Caching session data
- Rate limiting

**Popular options:**
- Redis (rich data structures)
- DynamoDB (AWS managed)
- etcd (configuration, service discovery)

**Avoid when:**
- Need complex queries
- Need to search by value
- Data relationships matter

---

### Wide-Column Store

**Choose when:**
- Need massive scale (billions of rows)
- Write-heavy workloads
- Time-series or IoT data
- Query patterns are known upfront

**Popular options:**
- Apache Cassandra (most popular)
- Google Bigtable
- ScyllaDB (Cassandra-compatible, faster)

**Avoid when:**
- Need ACID transactions
- Complex ad-hoc queries
- Small to medium datasets

---

### Graph Database

**Choose when:**
- Relationships are core to the problem
- Social networks, recommendations
- Fraud detection patterns
- Knowledge graphs

**Popular options:**
- Neo4j (most mature)
- Amazon Neptune (AWS managed)
- ArangoDB (multi-model)

**Avoid when:**
- Relationships are simple
- Mostly read-heavy workloads
- Need strong ACID

---

### Columnar/OLAP

**Choose when:**
- Analytical queries on large datasets
- Data warehouse workloads
- Business intelligence
- Aggregations on billions of rows

**Popular options:**
- ClickHouse (open-source, fast)
- Snowflake (cloud-native)
- BigQuery (serverless)
- Redshift (AWS)

**Avoid when:**
- Transactional workloads
- Need sub-second queries
- Small datasets

---

### Time-Series Database

**Choose when:**
- Storing timestamped measurements
- Monitoring and metrics
- IoT sensor data
- Financial tick data

**Popular options:**
- TimescaleDB (PostgreSQL-based)
- InfluxDB (purpose-built)
- QuestDB (high performance)

**Avoid when:**
- Primary access is by key, not time
- Need complex transactions
- General-purpose storage

---

### Vector Database

**Choose when:**
- Semantic similarity search
- RAG applications
- Image/video similarity
- Recommendation systems

**Popular options:**
- Pinecone (managed, easy)
- Qdrant (open-source, fast)
- Weaviate (hybrid search)
- Milvus (scale)

**Avoid when:**
- Keyword search suffices
- Exact match queries
- General data storage

---

## Workload-Based Selection

### OLTP (Transactional)

**Best:** PostgreSQL, MySQL, CockroachDB

**Characteristics:**
- Many small transactions
- Complex updates
- Row-by-row access

---

### OLAP (Analytical)

**Best:** ClickHouse, Snowflake, BigQuery

**Characteristics:**
- Few complex queries
- Full table scans
- Aggregations

---

### HTAP (Hybrid)

**Best:** TiDB, SingleStore, CockroachDB

**Characteristics:**
- Both OLTP and OLAP
- Real-time analytics
- Unified pipeline

---

## Scaling Considerations

### Vertical Scaling (Add CPU/RAM)

- PostgreSQL
- MySQL
- MongoDB (standalone)

### Horizontal Scaling (Add Nodes)

- Cassandra
- CockroachDB
- TiDB
- ClickHouse
- Redis Cluster

### Managed Services (Auto-scale)

- DynamoDB
- Aurora
- Snowflake
- BigQuery
- Pinecone

---

## Cost Optimization Tips

| Database | Cost Optimization Strategy |
|----------|---------------------------|
| PostgreSQL | Use read replicas, RDS reserved instances |
| Redis | Right-size instance, use Redis Cluster |
| MongoDB | Atlas serverless, Atlas Data Lake |
| Cassandra | Use S3 for archival, right-size nodes |
| ClickHouse | Tiered storage, compression |
| Snowflake | Virtual warehouses, caching |

---

## Security Considerations

| Database | Key Security Features |
|----------|----------------------|
| PostgreSQL | RLS, row-level security, encryption |
| MongoDB | Field-level encryption, LDAP |
| Cassandra | Internal authentication, encryption |
| Redis | ACLs, TLS, Redis Enterprise |
| Neo4j | Role-based access, encryption |

---

*Use this guide alongside the comprehensive [Database Systems Mastery](../database_systems_mastery.md) for detailed explanations.*
