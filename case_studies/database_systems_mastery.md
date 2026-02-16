# Database Systems Mastery: A Production-Oriented Guide

## Executive Summary

This comprehensive guide provides a structured pathway for experienced Full-Stack and AI engineers to build production-oriented understanding of all major database types. Building on existing familiarity with PostgreSQL, MongoDB, Redis, and vector databases like Qdrant/pgvector, this curriculum delivers deep understanding of database families, their strengths and weaknesses, and real-world architectural patterns for modern polyglot persistence systems.

---

## 1. High-Level Map of Database Types

The database landscape in 2025-2026 presents a rich ecosystem of specialized systems, each designed for specific data models, query patterns, and scaling requirements. Understanding this landscape is essential for making informed architectural decisions in production systems.

### 1.1 Classification Framework

Modern databases can be broadly categorized into the following families:

| Database Family | Primary Data Model | Key Characteristic | Typical Workload |
|-----------------|-------------------|-------------------|------------------|
| **Relational (SQL)** | Tables with rows/columns | ACID compliance, joins, schemas | OLTP, transactional systems |
| **Document** | JSON-like documents | Flexible schemas, nested data | Content management, user profiles |
| **Key-Value** | Key-value pairs | Extreme simplicity, O(1) access | Caching, sessions, config |
| **Wide-Column** | Column families | Massive scale, write-optimized | IoT, time-series at scale |
| **Graph** | Nodes and edges | Relationship traversal | Social networks, fraud detection |
| **Columnar/OLAP** | Column-oriented tables | Analytical queries, compression | Business intelligence, analytics |
| **Time-Series** | Timestamped measurements | Time-range queries, retention | Monitoring, metrics, IoT |
| **In-Memory** | Memory-resident data | Sub-millisecond latency | Caching, real-time analytics |
| **NewSQL/HTAP** | Distributed SQL tables | Scale + ACID + analytics | Modern transactional + analytical |
| **Vector** | Embeddings + metadata | Similarity search | RAG, semantic search, AI apps |

### 1.2 Major Products by Family

#### Relational Databases (SQL)

**Open Source:**
- **PostgreSQL** — The most advanced open-source RDBMS, with extensive JSON support, full-text search, and growing vector capabilities via pgvector. Industry standard for complex transactional systems.
- **MySQL** — Widely adopted, particularly for web applications. Powers Facebook, Twitter, YouTube. InnoDB is the default storage engine.
- **MariaDB** — MySQL fork with enhanced features, faster performance, and open-source governance. Drop-in replacement for MySQL.

**Commercial:**
- **Microsoft SQL Server** — Enterprise-grade with tight Windows ecosystem integration. Strong BI capabilities via SSIS, SSAS.
- **Oracle Database** — The enterprise heavyweight. RAC (Real Application Clusters) for high availability, Exadata for extreme performance.
- **IBM Db2** — Enterprise solution with strong mainframe integration and LUW (Linux, Unix, Windows) variants.
- **Snowflake** — Cloud-native data warehouse (ELT/OLAP focus) that redefines analytical processing.

#### NoSQL Databases

**Document Stores:**
- **MongoDB** — Leading document database with flexible JSON schemas. Powers major platforms like eBay, Stripe. Aggregation pipeline is exceptionally powerful.
- **CouchDB** — Apache project with CouchDB Sync protocol for offline-first applications.
- **CouchBase** — Combines document flexibility with key-value speed, with N1QL (SQL-like query language).

**Key-Value Stores:**
- **Redis** — In-memory data structures server. Supports strings, lists, sets, sorted sets, hashes, streams, bitmaps. Essential for caching and real-time applications.
- **Amazon DynamoDB** — Fully managed key-value and document database with automatic scaling. Single-digit millisecond latency at any scale.
- **etcd** — Distributed key-value store for configuration and service discovery. Kubernetes uses it.
- **Consul** — Service networking solution with key-value storage for configuration.

**Wide-Column Stores:**
- **Apache Cassandra** — Massive scalability, tunable consistency. Powers Apple, Netflix, Instagram. CQL (Cassandra Query Language) resembles SQL.
- **Google Bigtable** — Internal Google product exposed as Cloud Bigtable. Underlies many Google services.
- **Apache HBase** — Hadoop ecosystem column store, runs on HDFS. Good for random read/write access.
- **ScyllaDB** — Cassandra-compatible written in C++ for higher performance.

**Graph Databases:**
- **Neo4j** — The graph database leader. Cypher query language. Powers Walmart, NASA, and many recommendation systems.
- **Amazon Neptune** — Fully managed graph database supporting Property Graph and RDF (SPARQL).
- **ArangoDB** — Multi-model (document, key-value, graph) — a single database for multiple models.
- **TigerGraph** — High-performance graph analytics platform for enterprise.

#### Columnar / OLAP Databases

**Open Source:**
- **ClickHouse** — Column-oriented DBMS for online analytical processing. Exceptional for real-time analytics. Powers Yandex.Metrica (world's largest web analytics platform).
- **Apache Druid** — Real-time analytics data store. Sub-second queries on large data sets. Good for event-driven analytics.
- **Apache Iceberg** — Open table format for huge analytic datasets. Works with S3, HDFS. Designed for analytical workloads.
- **DuckDB** — Embedded analytical database. SQLite for analytics. Runs in-process, no external dependencies.

**Cloud Data Warehouses:**
- **Snowflake** — Multi-cluster shared data architecture. Separate compute/storage. Near-unlimited concurrency.
- **Amazon Redshift** — AWS data warehouse. Redshift Spectrum for querying S3 directly. RA3 nodes with managed storage.
- **Google BigQuery** — Serverless, highly scalable data warehouse. Separation of storage/compute. Excellent for PB-scale analytics.
- **Azure Synapse Analytics** — Microsoft's unified analytics platform. SQL and Spark pools.

#### Time-Series Databases

- **InfluxDB** — Leading open-source time-series database. InfluxQL and Flux query languages. Telegraf for collection.
- **TimescaleDB** — PostgreSQL extension for time-series. Full SQL compatibility. Hypertable chunking for performance.
- **QuestDB** — High-performance time-series database. SQL support.列式存储 for compression.
- **QuestDB** — Open-source, stream processing and analytics. Time-series focused with SQL.
- **KDB+** — Commercial time-series database from Kx Systems. Used extensively in finance. Extremely fast.

#### In-Memory Databases

- **Redis** — Already listed under Key-Value, but essential for in-memory use cases.
- **Memcached** — Simple distributed memory object caching system. Pure caching, no persistence.
- **Hazelcast** — In-memory computing platform. Distributed data structures, processing.
- **SAP HANA** — In-memory columnar database. Extreme performance for enterprise ERP workloads.

#### NewSQL / HTAP Systems

- **TiDB** — Open-source distributed SQL. MySQL compatible. Separates storage/compute (TiKV). HTAP capabilities.
- **CockroachDB** — Distributed SQL surviving cloud outages. PostgreSQL wire compatible. Geospatial support.
- **SingleStore** — formerly MemSQL. Real-time analytics on transactional data. MySQL compatible.
- **YugabyteDB** — PostgreSQL compatible distributed SQL. Cosmos DB compatible API option.
- **Google Spanner** — Globally distributed relational database. Strong consistency, horizontal scaling.

#### Vector Databases

**Dedicated Vector Databases:**
- **Qdrant** — Open-source vector similarity search engine. Written in Rust. Fast, efficient. Self-hosted or cloud.
- **Pinecone** — Fully managed vector database. Excellent performance, easy scaling. Serverless option.
- **Weaviate** — Open-source vector search engine. GraphQL and REST API. Modules for ML models.
- **Milvus** — Open-source vector database. Billion-scale support. Cloud-native.
- **Chroma** — Open-source embedding store. Simple, developer-friendly. Built for AI applications.
- **Pinecone** — Managed service with serverless architecture. Real-time indexing.

**Vector Extensions:**
- **pgvector** — Open-source extension for PostgreSQL. Vector similarity search. Seamless integration.
- **Elasticsearch** — Full-text search with vector support. 8.0+ has dense vector fields.
- **Redis Stack** — Redis with vector search modules. Existing Redis users can add vector capabilities.

---

## 2. Deep Dive Per Database Family

### 2.1 Relational Databases (RDBMS)

#### Core Data Model

Relational databases organize data into **tables** consisting of **rows** (records) and **columns** (attributes). Each table has a predefined **schema** that defines column names, data types, and constraints. Tables relate to each other through **foreign keys**, enabling **joins** across multiple tables.

**Storage Architecture:**

- Data is stored in pages (typically 8KB-16KB)
- Tables are heap-organized or clustered by primary key
- Indexes are separate B-tree structures for fast lookups
- Write-ahead log (WAL) ensures durability

**Query Model:**

- SQL (Structured Query Language) as declarative query language
- Set-based operations (INSERT, UPDATE, DELETE operate on sets)
- Transaction support with ACID properties
- Complex joins, subqueries, window functions

#### Key Technical Strengths

1. **ACID Guarantees:** Atomicity, Consistency, Isolation, Durability ensure reliable transaction processing. Critical for financial systems, order processing, and any system where data integrity is non-negotiable.

2. **Mature Query Optimization:** Cost-based optimizers have decades of refinement. Query planners choose optimal execution paths based on statistics.

3. **Referential Integrity:** Foreign key constraints prevent orphaned records. Cascading deletes maintain data consistency.

4. **Rich Ecosystem:** ODBC, JDBC, ORMs, BI tools, backup solutions. Every conceivable integration exists.

5. **Strong Isolation Levels:** From Read Uncommitted to Serializable, precise control over concurrency behavior.

6. **Window Functions:** Advanced analytics (running totals, rankings, lag/lead) directly in SQL without application code.

#### Main Limitations and Trade-offs

1. **Horizontal Scaling Challenge:** Traditional RDBMS scale vertically. Sharding adds application complexity, loses cross-shard transactions.

2. **Schema Rigidity:** Schema changes require ALTER TABLE, which can be expensive on large tables (PostgreSQL online DDL helps).

3. **Object-Relational Impedance Mismatch:** Mapping OOP objects to tables requires ORM overhead, loses object semantics.

4. **Join Performance:** Complex multi-table joins on large datasets can be slow. Denormalization trades write performance for read speed.

5. **Cost:** Commercial licenses (Oracle, SQL Server) are expensive. Even open-source requires operational expertise.

#### When to Use (Great Choice)

- **Core transactional systems:** Banking, e-commerce orders, inventory, accounting
- **Systems requiring data integrity:** Financial ledgers, medical records, compliance-heavy industries
- **Complex relationships with integrity constraints:** ERP, CRM, supply chain
- **Analytical workloads on structured data:** Reporting, dashboards, business intelligence
- **When ACID is non-negotiable:** Payment processing, booking systems

#### When to Avoid (Bad Choice)

- **Unstructured or semi-structured data with varying schemas:** Document databases are better
- **Extreme write throughput:** Millions of writes/second — consider Cassandra or ScyllaDB
- **Hierarchical data:** XML documents, organizational charts — document or graph DBs
- **Simple key-value access patterns:** Overhead not justified — use Redis or DynamoDB
- **Massive data with simple queries:** Petabyte-scale analytics — use data warehouses

#### Real-World Use Cases

**Use Case 1: E-Commerce Order Management**

A large e-commerce platform processes 50,000 orders per hour during peak periods. Each order involves customer data, multiple line items, payment processing, inventory updates, shipping calculations, and notification triggers.

**Why PostgreSQL:**

- ACID transactions ensure inventory atomicity (decrement stock + create order = all-or-nothing)
- Foreign keys enforce referential integrity (order → customer, order → products)
- Row-level security enables multi-tenant isolation
- JSONB columns accommodate order metadata without schema changes

**Implementation Pattern:**

```sql
BEGIN TRANSACTION;

-- Lock inventory rows to prevent overselling
SELECT product_id, quantity FROM inventory 
WHERE product_id IN (...) FOR UPDATE;

-- Decrement inventory
UPDATE inventory SET quantity = quantity - 1 
WHERE product_id = ?;

-- Create order
INSERT INTO orders (customer_id, total, status) 
VALUES (?, ?, 'PENDING');

-- Add line items
INSERT INTO order_items (order_id, product_id, quantity, price)
SELECT ?, product_id, quantity, price FROM basket 
WHERE customer_id = ?;

-- Process payment (call payment service, record result)
INSERT INTO payments (order_id, amount, gateway, status)
VALUES (?, ?, 'stripe', 'AUTHORIZED');

COMMIT;
```

**Use Case 2: Financial Ledger**

A fintech startup needs double-entry bookkeeping with audit trails. Every transaction must balance (debits = credits), and historical records cannot be modified (immutability).

**Why PostgreSQL:**

- ENUM types for account types and transaction states
- CHECK constraints validate balance rules
- Row-level security for multi-tenant ledgers
- Partial indexes for efficient time-range queries on archived data

**Implementation Pattern:**

```sql
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    account_number VARCHAR(20) NOT NULL,
    account_type account_type NOT NULL,
    balance DECIMAL(19,4) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    posted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT NOT NULL,
    status transaction_status NOT NULL DEFAULT 'PENDING'
);

CREATE TABLE entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID NOT NULL REFERENCES transactions(id),
    account_id UUID NOT NULL REFERENCES accounts(id),
    debit_amount DECIMAL(19,4) NOT NULL CHECK (debit_amount >= 0),
    credit_amount DECIMAL(19,4) NOT NULL CHECK (credit_amount >= 0),
    CONSTRAINT balanced_entry CHECK (
        (debit_amount > 0 AND credit_amount = 0) OR
        (debit_amount = 0 AND credit_amount > 0)
    )
);

-- Trigger to ensure transaction balances
CREATE OR REPLACE FUNCTION validate_transaction_balance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'POSTED' THEN
        IF EXISTS (
            SELECT 1 FROM entries 
            WHERE transaction_id = NEW.id
            GROUP BY transaction_id
            HAVING SUM(debit_amount) <> SUM(credit_amount)
        ) THEN
            RAISE EXCEPTION 'Transaction must balance: debits must equal credits';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

**Use Case 3: Multi-Tenant SaaS Platform**

A B2B SaaS application serves 500+ companies, each requiring strict data isolation. Tenants have varying needs — some need audit logs, others require data residency in specific regions.

**Why PostgreSQL:**

- Row-Level Security (RLS) enforces tenant isolation at database level
- Separate schemas or databases per tenant for strictest isolation
- Column-level encryption for sensitive data
- Rich partitioning for data lifecycle management

**Implementation Pattern:**

```sql
-- Enable RLS on all tenant tables
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- Create policy enforcing tenant isolation
CREATE POLICY tenant_isolation_policy ON orders
    USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- Function to set tenant context
CREATE OR REPLACE FUNCTION set_tenant(tenant_id UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.tenant_id', tenant_id::TEXT, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

---

### 2.2 Document Databases

#### Core Data Model

Document databases store **self-describing data** in JSON, BSON, or similar formats. Each document is a **unit of data** (like a row in SQL) but can contain nested objects and arrays — no fixed schema required.

**Storage Architecture:**

- Documents stored in collections (analogous to tables)
- Each document has unique `_id` (or equivalent) primary key
- Documents are serialized (BSON for MongoDB, JSON for CouchDB)
- Indexes on document fields (including nested fields)
- WiredTiger storage engine (MongoDB) provides compression, concurrency

**Query Model:**

- Queries on document fields using dot notation for nesting
- Aggregation pipelines for complex transformations (MongoDB)
- Range queries, text search, geospatial queries
- No joins by default — denormalization expected (or $lookup for MongoDB)

#### Key Technical Strengths

1. **Flexible Schema:** Documents in the same collection can have different fields. Schema evolution is painless — add fields without migrations.

2. **Hierarchical Data Handling:** Nested arrays and objects map naturally to JSON. No need to normalize into separate tables.

3. **Developer Productivity:** JSON is ubiquitous. ORM-like mapping is natural. Document models match object-oriented code.

4. **Horizontal Scalability:** Sharding built-in (MongoDB, CouchDB). Automatic distribution of documents across shards.

5. **Rich Query Languages:** MongoDB aggregation pipeline rivals SQL expressiveness. Can do complex transformations in-database.

6. **Document Versioning:** Some databases (CouchDB) support conflict resolution for distributed offline-first apps.

#### Main Limitations and Trade-offs

1. **No ACID Across Documents:** Multi-document transactions exist (MongoDB 4.0+) but are more limited and slower than RDBMS.

2. **No Joins:** Must denormalize or perform application-level joins. $lookup has performance costs.

3. **Query Planning Limitations:** Less mature optimizers than SQL. Complex aggregations need careful indexing.

4. **Memory Footprint:** Denormalization leads to data duplication. Storage can exceed normalized RDBMS.

5. **Operational Complexity:** Sharding, replica sets require expertise. Not "set and forget."

#### When to Use (Great Choice)

- **Content management systems:** Articles, blog posts, media metadata with varying attributes
- **User profiles with custom attributes:** Preferences, settings, behavioral data varying by user type
- **Event logging and analytics:** Log entries with different event types having different fields
- **Catalogs with variable attributes:** Product catalogs where products have different specifications
- **Real-time analytics with high write throughput:** IoT data ingestion, clickstream analysis

#### When to Avoid (Bad Choice)

- **Systems requiring complex transactions:** Financial transactions across multiple entities
- **Highly normalized data with complex relationships:** Complex joins are painful
- **Strict schema requirements:** Regulated industries requiring schema enforcement
- **BI-style analytical queries:** Columnar/OLAP databases are far more efficient

#### Real-World Use Cases

**Use Case 1: Event Logging and Analytics Platform**

A mobile gaming company captures player events — login, level completion, purchase, chat messages — each with different attributes. They need to query by player, time range, and event type, with 100,000 events/second ingestion.

**Why MongoDB:**

- Each event type is a document with different fields (purchase has amount, level has difficulty)
- High write throughput with WiredTiger
- Compound indexes on (player_id, timestamp, event_type) for common queries
- TTL indexes auto-expire old logs

**Implementation Pattern:**

```javascript
// Events collection with flexible schema per event type
db.events.createIndex(
    { "player_id": 1, "timestamp": -1, "event_type": 1 },
    { name: "player_time_type_idx" }
);

// Different document structures per event type
{
    "_id": ObjectId("..."),
    "event_type": "level_completed",
    "player_id": "player_12345",
    "timestamp": ISODate("2025-01-15T14:32:00Z"),
    "level": 42,
    "difficulty": "hard",
    "time_seconds": 180,
    "stars": 3,
    "coins_earned": 500
}

{
    "_id": ObjectId("..."),
    "event_type": "purchase",
    "player_id": "player_12345",
    "timestamp": ISODate("2025-01-15T14:33:00Z"),
    "product_id": "premium_pass",
    "amount_usd": 9.99,
    "currency": "USD",
    "payment_gateway": "stripe"
}

// Aggregation pipeline for analytics
db.events.aggregate([
    { $match: { 
        "event_type": "level_completed",
        "timestamp": { $gte: startDate, $lt: endDate }
    }},
    { $group: {
        _id: "$level",
        avg_time: { $avg: "$time_seconds" },
        completions: { $sum: 1 },
        avg_stars: { $avg: "$stars" }
    }},
    { $sort: { completions: -1 }}
]);
```

**Use Case 2: Flexible User Profiles and Preferences**

A B2B SaaS platform serves customers who each need custom profile fields — HR software where one company tracks certifications, another tracks languages spoken, another tracks security clearances.

**Why MongoDB:**

- Each tenant/company can have custom profile schemas
- Add new profile fields without ALTER TABLE
- Array fields for multi-valued attributes (certifications, languages)
- Embedded documents for related but denormalized data

**Implementation Pattern:**

```javascript
// Employee profiles with company-specific custom fields
{
    "_id": ObjectId("..."),
    "employee_id": "EMP001",
    "company_id": "COMPANY_A",
    "base_profile": {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@company-a.com",
        "department": "Engineering",
        "hire_date": ISODate("2020-03-15")
    },
    "custom_fields": {
        "certifications": [
            { "name": "AWS Solutions Architect", "expires": ISODate("2026-06-01") },
            { "name": "PMP", "expires": ISODate("2025-12-31") }
        ],
        "languages": ["English", "Spanish", "Mandarin"],
        "security_clearance": "Top Secret",
        "clearance_renewal_date": ISODate("2025-09-01")
    }
}

// Different company, different custom fields
{
    "_id": ObjectId("..."),
    "employee_id": "EMP002",
    "company_id": "COMPANY_B",
    "base_profile": { ... },
    "custom_fields": {
        "uniform_size": "L",
        "shift_preference": "night",
        "emergency_contact": { "name": "Jane", "phone": "555-0123" },
        "union_member": true
    }
}
```

**Use Case 3: Product Catalog with Variable Attributes**

An electronics retailer sells phones (RAM, screen size, carrier), laptops (CPU, storage, GPU), and appliances (dimensions, energy rating) — each with completely different attributes.

**Why MongoDB:**

- No product type requires all attributes of another type
- New product categories add without schema changes
- Product search filters by dynamic attributes
- EAV (Entity-Attribute-Value) pattern eliminated

**Implementation Pattern:**

```javascript
// Phone document
{
    "_id": ObjectId("..."),
    "sku": "IPHONE15-PRO-256",
    "product_type": "phone",
    "name": "iPhone 15 Pro",
    "brand": "Apple",
    "price": 999.99,
    "category": "smartphones",
    "specs": {
        "ram_gb": 8,
        "storage_gb": 256,
        "screen_size_inches": 6.1,
        "carrier": "unlocked",
        "color": "titanium",
        "battery_mah": 3274,
        "processor": "A17 Pro"
    },
    "inventory": { "warehouse_a": 150, "warehouse_b": 75 }
}

// Laptop document
{
    "_id": ObjectId("..."),
    "sku": "DELL-XPS-15-32",
    "product_type": "laptop",
    "name": "Dell XPS 15",
    "brand": "Dell",
    "price": 1899.99,
    "category": "laptops",
    "specs": {
        "ram_gb": 32,
        "storage_gb": 1024,
        "storage_type": "NVMe SSD",
        "screen_size_inches": 15.6,
        "resolution": "3456x2160",
        "cpu": "Intel Core i7-13700H",
        "gpu": "NVIDIA RTX 4060",
        "battery_whr": 86,
        "weight_lbs": 4.2,
        "os": "Windows 11"
    },
    "inventory": { "warehouse_a": 23, "warehouse_b": 12 }
}

// Query for gaming laptops under $2000
db.products.find({
    "product_type": "laptop",
    "price": { $lt: 2000 },
    "specs.ram_gb": { $gte: 16 },
    "specs.gpu": { $regex: "RTX|Radeon" }
});
```

---

### 2.3 Key-Value Stores

#### Core Data Model

The simplest data model: **key** maps to **value**. Keys are unique identifiers; values can be strings, blobs, serialized objects, or structured data. No query language — direct key access is O(1).

**Storage Architecture:**

- In-memory (Redis, Memcached) or disk-based (DynamoDB, etcd)
- Hash-based indexing for key lookups
- Append-only log for persistence (Redis AOF)
- Data structures beyond simple strings (Redis): lists, sets, sorted sets, hashes, streams, bitmaps, hyperloglogs

**Query Model:**

- GET/SET/MGET/MSET for simple values
- Data structure commands for complex operations
- No joins, no filtering, no aggregations
- TTL (time-to-live) for expiration

#### Key Technical Strengths

1. **Extreme Simplicity:** No schema, no query language. Store and retrieve by key. Minimal cognitive overhead.

2. **Sub-Millisecond Latency:** In-memory operations (Redis) deliver nanosecond to microsecond access. Cache layers for hot data.

3. **Rich Data Structures:** Redis isn't just strings — lists for queues, sets for unique collections, sorted sets for leaderboards, streams for event sourcing.

4. **Built-in Caching Features:** TTL, LRU/LFU eviction policies, cache invalidation pub/sub.

5. **Horizontal Scalability:** Redis Cluster shards by key. DynamoDB scales automatically.

6. **Operational Simplicity:** No complex query planning, no index management. Deploy and use.

#### Main Limitations and Trade-offs

1. **No Query Capabilities:** Must know the key. Can't iterate, filter, or aggregate without bringing data to application.

2. **Single-Key Transactions Only:** No multi-key ACID (Redis MULTI/EXEC is transactional but not ACID across all operations).

3. **Memory Constraints:** In-memory databases are expensive at scale. Must be strategic about what to cache.

4. **Data Modeling Complexity:** All query patterns must be anticipated. Application does heavy lifting.

5. **No Durability Guarantees (by default):** Redis with AOF every write is slower; RDB snapshots lose data between saves.

#### When to Use (Great Choice)

- **Session storage:** User sessions, shopping carts, authentication tokens
- **Caching layer:** Hot data in front of slower databases, API response caching
- **Rate limiting:** Token bucket, sliding window counters per user/IP
- **Feature flags:** Boolean or JSON config toggles, A/B test assignments
- **Leaderboards:** Real-time rankings with sorted sets (Redis)
- **Message queues:** Redis streams, pub/sub for real-time messaging
- **Distributed locks:** Redlock pattern for inter-process synchronization

#### When to Avoid (Bad Choice)

- **Complex queries required:** Need filtering, sorting, joins — use RDBMS or document store
- **Primary data store:** Data requiring ACID transactions and complex relationships
- **Analytical workloads:** Aggregations, reporting — use OLAP databases
- **Large blob storage:** Video, images — use object storage (S3)

#### Real-World Use Cases

**Use Case 1: API Response Caching**

A REST API serves product listings. Products change infrequently but are read 10,000 times per minute. Database can't handle that load, but caching can reduce it 99%.

**Why Redis:**

- Sub-millisecond GET/SET
- TTL for cache invalidation
- SET with NX (only if not exists) for cache-aside pattern
- GETEX for setting expiry on retrieval

**Implementation Pattern:**

```python
import redis
import json
import hashlib

r = redis.Redis(host='redis-master', decode_responses=True)

def get_products(category: str, page: int = 1) -> list:
    # Generate cache key from query
    cache_key = f"products:{category}:page:{page}"
    
    # Try cache first (cache-aside)
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Cache miss - fetch from database
    products = db.query("""
        SELECT * FROM products 
        WHERE category = %s 
        LIMIT 20 OFFSET %s
    """, category, (page - 1) * 20)
    
    # Store in cache with 60-second TTL
    r.setex(cache_key, 60, json.dumps(products))
    
    return products

def invalidate_product_cache(product_id: int):
    """Called when product is updated"""
    product = db.query("SELECT category FROM products WHERE id = %s", product_id)
    if product:
        # Invalidate all pages for this category
        r.delete(f"products:{product['category']}:page:*")
        # Also delete specific product cache
        r.delete(f"product:{product_id}")
```

**Use Case 2: Distributed Rate Limiting**

API rate limiting: 100 requests per minute per API key. Need sliding window, not fixed window (to prevent burst at boundaries).

**Why Redis:**

- INCR with EXPIRE atomic operations
- Sorted sets with timestamps for sliding window
- Sub-millisecond increment operations

**Implementation Pattern:**

```python
def check_rate_limit(api_key: str, limit: int = 100, window_seconds: int = 60) -> bool:
    """Sliding window rate limiting using Redis sorted set"""
    key = f"rate_limit:{api_key}"
    now = time.time()
    window_start = now - window_seconds
    
    pipe = r.pipeline()
    
    # Remove old entries outside window
    pipe.zremrangebyscore(key, 0, window_start)
    
    # Count requests in current window
    pipe.zcard(key)
    
    # Add current request
    pipe.zadd(key, {str(now): now})
    
    # Set expiry on the key
    pipe.expire(key, window_seconds)
    
    results = pipe.execute()
    request_count = results[1]
    
    if request_count > limit:
        # Over limit - remove the request we just added
        r.zrem(key, str(now))
        return False
    
    return True
```

**Use Case 3: Real-Time Leaderboard**

A gaming platform tracks player scores globally and per-region. Leaderboards must update in real-time as players complete levels, and support top-100 queries and player rank lookups.

**Why Redis:**

- Sorted sets (ZSET) maintain ordered data by score
- ZADD updates in O(log(N))
- ZREVRANGE gets top players in O(log(N)+M)
- ZRANK gives player position in O(log(N))

**Implementation Pattern:**

```python
def update_score(player_id: str, score_increase: int, region: str = "global"):
    """Increment player score and return new rank"""
    key = f"leaderboard:{region}"
    
    # ZINCRBY atomically increments score
    new_score = r.zincrby(key, score_increase, player_id)
    new_rank = r.zrevrank(key, player_id) + 1  # 0-indexed
    
    return {"score": new_score, "rank": new_rank}

def get_top_players(region: str = "global", top_n: int = 100):
    """Get top N players with scores"""
    key = f"leaderboard:{region}"
    
    # ZREVRANGE returns highest scores first (with scores)
    results = r.zrevrange(key, 0, top_n - 1, withscores=True)
    
    return [
        {"rank": idx + 1, "player_id": player_id, "score": score}
        for idx, (player_id, score) in enumerate(results)
    ]

def get_player_around_me(player_id: str, region: str = "global", count: int = 5):
    """Get players around a specific player"""
    key = f"leaderboard:{region}"
    
    rank = r.zrevrank(key, player_id)
    if rank is None:
        return {"error": "Player not found"}
    
    start = max(0, rank - count)
    end = rank + count
    
    results = r.zrevrange(key, start, end, withscores=True)
    
    return [
        {"rank": idx + start + 1, "player_id": pid, "score": score}
        for idx, (pid, score) in enumerate(results)
    ]
```

---

### 2.4 Wide-Column Databases

#### Core Data Model

Wide-column stores organize data into **column families** (similar to tables) but with **dynamic columns** — rows can have different columns, and columns can be added freely. Data is stored by column, not by row, enabling efficient analytical queries.

**Storage Architecture:**

- Column-oriented: All values for a column stored together
- Column families group related columns
- Row keys (primary keys) for record identification
- Partitioning by row key for horizontal scaling
- Compression is highly effective (similar values stored together)

**Query Model:**

- HBase, Cassandra use CQL (similar to SQL but limited)
- Primary key lookups are fast (O(1))
- Range scans on row keys possible
- Secondary indexes available but slower
- No joins; denormalization expected

#### Key Technical Strengths

1. **Massive Write Throughput:** Optimized for writes. Append-only log structure. Cassandra handles millions of writes/second across clusters.

2. **Petabyte Scale:** Proven at massive scale (Apple, Netflix, Instagram). Linear horizontal scaling.

3. **Column-Oriented Storage:** Analytical queries on specific columns read less data. Excellent compression ratios.

4. **Tunable Consistency:** Cassandra: eventual to strong consistency configurable per query. CAP theorem: choose availability or consistency.

5. **No Single Point of Failure:** Cassandra, HBase have no master (or handle master failover). Multi-datacenter replication built-in.

6. **Time-Series Friendly:** Append-heavy workloads, data that grows over time. TTL support for automatic expiration.

#### Main Limitations and Trade-offs

1. **Limited Query Patterns:** Primary key access or range scans. No secondary indexes are efficient. Query patterns must be designed around.

2. **No ACID:** Cassandra provides BASE (Basically Available, Soft state, Eventual consistency). Not suitable for transactional systems.

3. **Data Modeling Complexity:** Query-first design. Must know all access patterns upfront. Denormalization is mandatory.

4. **No Joins:** Application must handle joins or denormalize into single column families.

5. **Operational Complexity:** Managing distributed clusters, repairs, tombstones, compaction. Expertise required.

#### When to Use (Great Choice)

- **IoT and sensor data:** High-velocity writes, simple key-based retrieval by device/time
- **Event logging:** Clickstream, audit logs, system events
- **Time-series data:** Metrics, monitoring data with TTL requirements
- **Large-scale analytics:** Batch analytical queries on massive datasets
- **Cassandra-only: Globally distributed apps:** Multi-region, always-available requirements (e.g., Instagram, Netflix use cases)

#### When to Avoid (Bad Choice)

- **Transactional workloads requiring ACID:** Financial transactions, inventory management
- **Complex queries and joins:** BI-style queries on related data
- **Small-scale applications:** Operational overhead not worth it for < 10K writes/second
- **Data with complex relationships:** Graph database better for connected data

#### Real-World Use Cases

**Use Case 1: IoT Sensor Data Ingestion**

A smart building system collects temperature, humidity, occupancy, and energy usage from 50,000 sensors every 30 seconds. That's 6,000 writes/second peak, with 5-year retention and query by building/floor/sensor/time-range.

**Why Apache Cassandra:**

- Linear scaling handles write throughput
- TTL for automatic data expiration
- Row key design: (building_id, floor_id, sensor_id, timestamp)
- Efficient time-range queries within partition
- Multi-datacenter replication for disaster recovery

**Implementation Pattern:**

```sql
-- Table design: partition by sensor to keep related data together
CREATE TABLE sensor_data (
    sensor_id uuid,
    timestamp timestamp,
    temperature decimal,
    humidity decimal,
    occupancy int,
    energy_watts decimal,
    PRIMARY KEY (sensor_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC)
  AND compaction = {'class': 'TimeWindowCompactionStrategy'}
  AND default_time_to_live = 157680000;  -- 5 years

-- Query: all data for one sensor in a time range
SELECT * FROM sensor_data 
WHERE sensor_id = ? 
AND timestamp >= '2025-01-01' 
AND timestamp < '2025-01-02';

-- Query: latest reading for all sensors in a building
CREATE TABLE latest_sensor_reading (
    building_id uuid,
    floor_id int,
    sensor_id uuid,
    timestamp timestamp,
    temperature decimal,
    humidity decimal,
    PRIMARY KEY ((building_id, floor_id), sensor_id)
) WITH CLUSTERING ORDER BY (sensor_id DESC);
```

**Use Case 2: User Event Tracking at Scale**

A fintech app tracks user events: login, transaction, page view, error. 50 million users, 1 billion events/day. Analysts need to query by user, event type, and time range for fraud analysis.

**Why Cassandra:**

- High write throughput for event ingestion
- TTL for data lifecycle management
- Efficient storage (columnar)
- Materialized views for different query patterns
- Analytics integration via Spark

**Implementation Pattern:**

```sql
-- Events table: query by user or by event type
CREATE TABLE user_events (
    user_id uuid,
    event_id timeuuid,
    event_type text,
    page text,
    amount decimal,
    status text,
    metadata map<text, text>,
    PRIMARY KEY (user_id, event_id)
) WITH CLUSTERING ORDER BY (event_id DESC);

-- Materialized view for querying by event type
CREATE MATERIALIZED VIEW events_by_type AS
SELECT * FROM user_events
WHERE event_type IS NOT NULL AND event_id IS NOT NULL
PRIMARY KEY (event_type, event_id, user_id);

-- Query: all login events for a user
SELECT * FROM user_events 
WHERE user_id = ? AND event_type = 'login';

-- Query: high-value transactions in time range (for fraud analysis)
SELECT * FROM user_events 
WHERE user_id = ?
AND event_type = 'transaction'
AND amount > 10000
AND event_id >= minTimeuuid('2025-01-01')
AND event_id < minTimeuuid('2025-01-02');
```

**Use Case 3: Messaging/Chat History**

A messaging platform stores chat messages. Users send millions of messages per minute. Query patterns: conversation history (user A + user B), search within conversation, recent messages.

**Why Cassandra:**

- Append-only writes for new messages
- TTL for message expiration (optional)
- Efficient partition for conversation lookup
- Sort by clustering key (timestamp)

**Implementation Pattern:**

```sql
-- Conversation messages: partition by conversation ID
CREATE TABLE messages (
    conversation_id uuid,
    message_id timeuuid,
    sender_id uuid,
    content text,
    attachments list<text>,
    read_at timestamp,
    deleted_at timestamp,
    PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);

-- Query: last 50 messages in conversation
SELECT * FROM messages 
WHERE conversation_id = ?
LIMIT 50;

-- Query: search messages containing keyword (requires SASI index or external search)
-- Note: Full-text search not native - consider Lucene/Solr integration

-- Query: unread count per conversation
CREATE TABLE conversation_unread (
    user_id uuid,
    conversation_id uuid,
    unread_count int,
    last_read_message_id timeuuid,
    PRIMARY KEY (user_id, conversation_id);
);
```

---

### 2.5 Graph Databases

#### Core Data Model

Graph databases represent data as **nodes** (entities) and **edges** (relationships). Both nodes and edges can have **properties** (key-value attributes). Edges are directed and typed, representing specific relationship types.

**Storage Architecture:**

- Adjacency lists: each node stores its incoming/outgoing edges
- Index-free adjacency: edge traversal is O(1) lookup, no index required
- Native graph processing for traversals
- Specialized graph engines for analytics (PageRank, community detection)

**Query Model:**

- Declarative query languages: Cypher (Neo4j), Gremlin (Apache TinkerPop), SPARQL (RDF)
- Pattern matching: find nodes connected by specific relationship paths
- Path finding: shortest path, all paths, weighted path
- Graph algorithms: centrality, community detection, similarity

#### Key Technical Strengths

1. **Relationship-First Design:** Relationships are first-class citizens. Traversing connected data is native and fast.

2. **Complex Relationship Queries:** Multi-hop queries (friend-of-friend-of-friend) are O(1) per hop, not O(n²) like SQL joins.

3. **Intuitive Modeling:** Graph models map naturally to real-world entities and their connections. Easier to understand and maintain.

4. **Native Graph Algorithms:** PageRank, community detection, shortest path built-in. No external libraries needed.

5. **Flexible Schema:** Add node types, relationship types, properties without migrations. Evolving models.

6. **Explainability:** Query results are paths, easy to trace and explain. Important for fraud detection explanations.

#### Main Limitations and Trade-offs

1. **Limited Scalability:** Graph databases historically struggled at massive scale (billions of nodes). Neo4j clustering, TigerGraph, and Neptune address this but with complexity.

2. **Query Performance on Large Graphs:** Deep traversals (6+ hops) become slow. Sharding graphs is extremely difficult.

3. **Less Mature Ecosystem:** Smaller tool ecosystem than RDBMS. Fewer ORMs, BI integrations.

4. **Specialized Skills Required:** Cypher/Gremlin expertise rare. Graph modeling different from relational modeling.

5. **Not General-Purpose:** When relationships aren't central, graph databases add unnecessary complexity.

#### When to Use (Great Choice)

- **Social networks:** Friends, followers, connections; multi-hop queries for "people you may know"
- **Fraud detection:** Complex patterns across entities; money laundering ring detection
- **Recommendation engines:** Co-purchase, co-view patterns; collaborative filtering via graphs
- **Knowledge graphs:** Entities and relationships; question answering, semantic search
- **Network/IT operations:** Infrastructure mapping, dependency graphs, impact analysis
- **Identity and access management:** User-role-permission hierarchies

#### When to Avoid (Bad Choice)

- **Simple CRUD with few relationships:** Overhead not worth it
- **Transactional systems:** ACID requirements better served by RDBMS
- **Massive-scale simple lookups:** Billions of simple entities without complex relationships
- **Reporting/BI:** Aggregations, dashboards — columnar/OLAP better

#### Real-World Use Cases

**Use Case 1: Social Network "People You May Know"**

A social platform with 50 million users needs to recommend connections. Rule: "friends of friends" not already connected, weighted by common friends count.

**Why Neo4j:**

- Efficient multi-hop traversal
- Cypher pattern matching is expressive and readable
- Algorithms library for common graph operations
- Proven at social scale

**Implementation Pattern:**

```cypher
// Find recommendations for user
MATCH (user:User {id: $user_id})-[:FRIEND]->(friend)-[:FRIEND]->(recommendation)
WHERE NOT (user)-[:FRIEND]->(recommendation)
AND recommendation.id <> user.id
WITH recommendation, count(friend) as common_friends
WHERE common_friends >= 2
RETURN recommendation.id, recommendation.name, common_friends
ORDER BY common_friends DESC
LIMIT 10
```

**Use Case 2: Fraud Detection — Money Laundering Pattern**

Detect "smurfing" pattern: multiple accounts splitting large transactions to avoid reporting thresholds, then aggregating funds.

**Why Graph:**

- Pattern detection across entity relationships
- Explainable detection (show the path)
- Real-time traversal as transactions occur

**Implementation Pattern:**

```cypher
// Detect accounts that receive from many sources, then send to few destinations
// (classic layering pattern)

MATCH (source:Account)<-[r1:TRANSFER]-(:Account)-[r2:TRANSFER]->(destination:Account)
WHERE r1.amount < 10000  // Below reporting threshold
AND r2.amount < 10000
WITH source, count(DISTINCT r1) as inbound_count, 
             count(DISTINCT r2) as outbound_count,
     collect(DISTINCT destination) as destinations
WHERE inbound_count >= 5 AND outbound_count >= 3
RETURN source.id, source.name, inbound_count, outbound_count,
       [d IN destinations | d.id] as destination_accounts
```

**Use Case 3: Knowledge Graph for Product Recommendations**

E-commerce knowledge graph: products, categories, brands, user purchases, ratings. Power recommendations by traversing: "users who bought X also bought Y", "products in category related to Z".

**Why Neo4j:**

- Rich relationship types (BOUGHT, RATED, IN_CATEGORY, SIMILAR_TO)
- Hybrid recommendations: content-based + collaborative
- Real-time graph traversal for online inference

**Implementation Pattern:**

```cypher
// Collaborative filtering: find products bought by similar users
MATCH (user:User {id: $current_user})-[:BOUGHT]->(product:Product)
MATCH (user)-[:BOUGHT]->(p2:Product)<-[:BOUGHT]-(similar:User)
WHERE NOT (user)-[:BOUGHT]->(p2)
WITH product, p2, count(similar) as similarity_score
ORDER BY similarity_score DESC
RETURN p2.name, p2.category, similarity_score
LIMIT 5

// Content-based: similar products by category/brand
MATCH (product:Product {id: $product_id})-[:IN_CATEGORY|RELATED_TO|SAME_BRAND]-()
WITH product, neighbors(product) as related
UNWIND related as rec
RETURN rec.name, rec.price, count(*) as relationship_strength
ORDER BY relationship_strength DESC
LIMIT 5
```

---

### 2.6 Columnar / OLAP Databases

#### Core Data Model

Columnar databases store data **by column** rather than by row. Values for each column are stored contiguously, enabling:

- **Analytical queries** that read few columns across many rows — read only needed columns
- **High compression** — similar values compress well
- **Vectorized processing** — SIMD operations on column batches

**Storage Architecture:**

- Column files (one per column per partition)
- Compression (dictionary, run-length, delta)
- Data types optimized for analytics (DATE, TIMESTAMP, DECIMAL)
- Partitioning by time or key range
- Tiered storage: hot (SSD), cold (HDD/S3)

**Query Model:**

- SQL is standard (most support PostgreSQL/MySQL dialect or extensions)
- Cost-based optimizers for complex queries
- Materialized views for pre-computed aggregations
- ETL/ELT integration for data loading

#### Key Technical Strengths

1. **Analytical Query Performance:** Orders of magnitude faster than row stores for aggregations, scans, GROUP BY on large datasets.

2. **Compression:** 5-10x compression common. Less I/O for queries. Lower storage costs.

3. **Vectorized Execution:** Process column batches with CPU SIMD. Modern CPUs love columnar.

4. **Massive Scale:** Petabyte-scale analytical workloads proven. Snowflake, BigQuery, Redshift handle this routinely.

5. **Separation of Compute/Storage:** Cloud data warehouses scale independently. Pay for queries, not idle clusters.

6. **Rich SQL Support:** Window functions, CTEs, complex joins. SQL for analytics is mature.

#### Main Limitations and Trade-offs

1. **Slow Single-Row Access:** Point queries (WHERE id = ?) require reading entire column files. Not for OLTP.

2. **Write Performance:** Columnar requires read-modify-rewrite for updates. Not for high-frequency OLTP.

3. **Latency:** Query latency ranges from milliseconds to minutes for complex queries. Not real-time.

4. **Cost:** Cloud data warehouses can be expensive at scale. Query optimization matters.

5. **Data Freshness:** ETL batches mean minutes to hours delay. Real-time requires additional streaming layers.

#### When to Use (Great Choice)

- **Business intelligence dashboards:** Complex aggregations on historical data
- **Ad-hoc analytical queries:** "Show me revenue by region, product, quarter for the last 3 years"
- **Data warehouse:** Central repository for reporting from multiple source systems
- **Log analytics:** Clickstream, server logs, application logs — simple fields, massive volume
- **Financial reporting:** Complex calculations, aggregations, time-series on financial data

#### When to Avoid (Bad Choice)

- **Transactional workloads:** Row stores are far faster for single-row inserts/updates
- **Real-time operational queries:** Sub-second queries on current data
- **Simple key-value access patterns:** Use Redis or document databases
- **Small data (< millions of rows):** Overhead not worth it

#### Real-World Use Cases

**Use Case 1: E-Commerce Business Intelligence**

A retail company with 500 stores, 100,000 products, 10 million transactions/month needs dashboards: revenue by store/product/region/time, profit margins, inventory turnover, customer lifetime value.

**Why ClickHouse:**

- Columnar storage for analytical queries
- Exceptional compression (10x+ on retail data)
- Real-time INSERT for near-real-time dashboards
- Cost-effective (open source, can self-host)
- Handles billions of rows

**Implementation Pattern:**

```sql
-- Sales table: partitioned by month
CREATE TABLE sales (
    sale_id UInt64,
    store_id UInt32,
    product_id UInt32,
    customer_id UInt64,
    sale_date Date,
    sale_datetime DateTime,
    quantity UInt16,
    unit_price Decimal(10,2),
    discount Decimal(10,2),
    region String,
    category String,
    subcategory String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(sale_date)
ORDER BY (store_id, sale_date, product_id);

-- Query: Revenue by category, month, region
SELECT 
    toYYYYMM(sale_date) as month,
    region,
    category,
    sum(quantity * unit_price * (1 - discount)) as revenue
FROM sales
WHERE sale_date >= '2024-01-01'
GROUP BY month, region, category
ORDER BY month DESC, revenue DESC;

-- Query: Rolling 30-day revenue per store
SELECT 
    store_id,
    sum(quantity * unit_price) as total_revenue,
    avg(quantity * unit_price) as avg_order_value,
    count(DISTINCT customer_id) as unique_customers
FROM sales
WHERE sale_date >= today() - 30
GROUP BY store_id
ORDER BY total_revenue DESC;

-- Materialized view for fast dashboard queries
CREATE MATERIALIZED VIEW sales_monthly
ENGINE = SummingMergeTree()
PARTITION BY year_month
ORDER BY (store_id, category)
AS SELECT
    toYYYYMM(sale_date) as year_month,
    store_id,
    category,
    sum(quantity * unit_price) as revenue,
    sum(quantity) as units_sold,
    count() as transaction_count
FROM sales
GROUP BY year_month, store_id, category;
```

**Use Case 2: User Behavior Analytics**

A SaaS platform needs to analyze user behavior: feature adoption, session length, funnel conversion, cohort retention. 100,000 daily active users, millions of events/day.

**Why BigQuery:**

- Serverless: no cluster management
- Separate storage/compute — pay for what you use
- Easy data sharing across organization
- Standard SQL, familiar to analysts
- Strong integrations (Looker, Tableau, Python)

**Implementation Pattern:**

```sql
-- Events table (partitioned by date)
CREATE TABLE analytics.events (
    event_id STRING,
    user_id STRING,
    event_type STRING,
    event_timestamp TIMESTAMP,
    platform STRING,
    session_id STRING,
    page_path STRING,
    properties STRING  -- JSON string for flexible properties
)
PARTITION BY DATE(event_timestamp)
OPTIONS(require_partition_filter=true);

-- Funnel analysis: signup -> onboarding -> first_action
WITH funnels AS (
    SELECT 
        user_id,
        MIN(IF(event_type = 'signup', event_timestamp, NULL)) as signup_time,
        MIN(IF(event_type = 'onboarding_complete', event_timestamp, NULL)) as onboarding_time,
        MIN(IF(event_type = 'first_action', event_timestamp, NULL)) as first_action_time
    FROM analytics.events
    WHERE DATE(event_timestamp) >= '2025-01-01'
    GROUP BY user_id
)
SELECT 
    COUNT(*) as total_users,
    COUNT(onboarding_time) as completed_onboarding,
    COUNT(first_action_time) as completed_first_action,
    ROUND(100.0 * COUNT(onboarding_time) / COUNT(*), 1) as onboarding_rate,
    ROUND(100.0 * COUNT(first_action_time) / COUNT(onboarding_time), 1) as first_action_rate
FROM funnels;

-- Cohort retention analysis
WITH cohorts AS (
    SELECT 
        user_id,
        DATE(MIN(event_timestamp)) as cohort_date
    FROM analytics.events
    WHERE event_type = 'signup'
    GROUP BY user_id
),
retention AS (
    SELECT 
        c.cohort_date,
        DATE(e.event_timestamp) as activity_date,
        DATEDIFF(DAY, c.cohort_date, DATE(e.event_timestamp)) as days_since_cohort,
        COUNT(DISTINCT e.user_id) as active_users
    FROM analytics.events e
    JOIN cohorts c ON e.user_id = c.user_id
    WHERE DATE(e.event_timestamp) >= c.cohort_date
    GROUP BY c.cohort_date, activity_date
)
SELECT 
    cohort_date,
    MAX(CASE WHEN days_since_cohort = 0 THEN active_users END) as day_0,
    MAX(CASE WHEN days_since_cohort = 1 THEN active_users END) as day_1,
    MAX(CASE WHEN days_since_cohort = 7 THEN active_users END) as day_7,
    MAX(CASE WHEN days_since_cohort = 30 THEN active_users END) as day_30
FROM retention
GROUP BY cohort_date
ORDER BY cohort_date DESC;
```

**Use Case 3: Clickstream Analytics**

A media company tracks article views, video plays, ad impressions. 500 million events/day. Need real-time dashboards for editorial team + historical analysis for content strategy.

**Why ClickHouse + Kafka:**

- Kafka for real-time ingestion
- ClickHouse for both real-time and historical
- Replicated tables for high availability
- TTL for data lifecycle

**Implementation Pattern:**

```sql
-- Real-time aggregated metrics
CREATE TABLE page_views_mv
ENGINE = SummingMergeTree()
ORDER BY (page_path, hour)
AS SELECT
    toStartOfHour(event_timestamp) as hour,
    page_path,
    count() as views,
    uniq(user_id) as unique_users,
    sum(duration_seconds) as total_time
FROM page_views
GROUP BY hour, page_path;

-- Query: Trending content now
SELECT 
    page_path,
    views,
    unique_users,
    avg_time_per_view
FROM page_views_mv
WHERE hour >= now() - INTERVAL 1 HOUR
ORDER BY views DESC
LIMIT 20;
```

---

### 2.7 Time-Series Databases

#### Core Data Model

Time-series databases (TSDB) are specialized for **timestamped data points** — sequences of values measured over time. Optimized for:

- **Time-range queries:** Fetch all data in a time range
- **Downsampling:** Aggregate high-resolution data into lower resolution
- **Retention policies:** Auto-delete old data
- **Partitioning:** By time (daily, monthly partitions)

**Storage Architecture:**

- Column-oriented storage (similar to OLAP)
- Compression by time ordering
- Tiered storage: recent data fast, old data compressed
- Automatic partition management
- Write-optimized: append-only, batch inserts

**Query Model:**

- SQL-like queries with time functions
- Aggregation over time windows
- Downsampling on-the-fly (1-minute → 1-hour)
- Interpolation for missing values
- Continuous queries for real-time alerting

#### Key Technical Strengths

1. **Efficient Time-Range Queries:** Optimized for fetching data in time ranges. Partitioning by time makes this fast.

2. **Automatic Retention:** Define retention policies. Old data automatically deleted or downsampled.

3. **Built-in Functions:** Time-weighted averages, rate of change, moving averages. Queries native to domain.

4. **Compression:** Time-series data compresses extremely well (similar values over time).

5. **Downsampling:** Automatic rollup of high-frequency data into lower resolution. Data reduction at ingest.

6. **High Write Throughput:** Optimized for append-heavy workloads. Batch writes for efficiency.

#### Main Limitations and Trade-offs

1. **Single-Dimensional Access:** Primary access pattern is time. Not good for random access by non-time keys.

2. **Limited Query Flexibility:** Optimized for time-range and aggregations. Ad-hoc queries on metadata slower.

3. **Not General-Purpose:** Can't replace RDBMS for transactional workloads.

4. **Learning Curve:** New query paradigms (Flux, InfluxQL) differ from SQL.

#### When to Use (Great Choice)

- **Infrastructure monitoring:** CPU, memory, disk metrics from servers, containers
- **Application performance monitoring:** Request latency, error rates, throughput
- **IoT sensor data:** Temperature, pressure, flow — device + timestamp + value
- **Financial tick data:** Stock prices, trade volumes — high-frequency time series
- **User analytics:** Daily active users, session counts over time

#### When to Avoid (Bad Choice)

- **Transactional workloads:** Need ACID, complex queries — use RDBMS
- **Document storage:** Different access patterns — use document DB
- **Metadata-heavy queries:** Need to query by non-time attributes primarily

#### Real-World Use Cases

**Use Case 1: Infrastructure Monitoring**

DevOps team monitors 5,000 servers with 50 metrics each (CPU, memory, disk, network). 250,000 metrics/minute. Need dashboards, alerting, historical analysis.

**Why TimescaleDB:**

- PostgreSQL extension — familiar SQL, existing tooling
- Continuous aggregates for automatic downsampling
- Hypertables for automatic partitioning
- Compression for storage efficiency
- Retention policies automated

**Implementation Pattern:**

```sql
-- Create hypertable: automatically partitioned by time
CREATE TABLE metrics (
    time TIMESTAMPTZ NOT NULL,
    server_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    tags JSONB
);

SELECT create_hypertable('metrics', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Index for efficient queries by server + time
CREATE INDEX idx_metrics_server_time ON metrics (server_id, time DESC);

-- Continuous aggregate: automatic hourly rollups
CREATE MATERIALIZED VIEW metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) as bucket,
    server_id,
    metric_name,
    avg(value) as value_avg,
    min(value) as value_min,
    max(value) as value_max,
    count(*) as sample_count
FROM metrics
GROUP BY bucket, server_id, metric_name;

-- Add refresh policy: auto-refresh last 2 hours
SELECT add_continuous_aggregate_policy('metrics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '5 minutes');

-- Retention policy: keep raw data 7 days, hourly aggregates 2 years
SELECT add_retention_policy('metrics', INTERVAL '7 days');
SELECT add_retention_policy('metrics_hourly', INTERVAL '2 years');

-- Query: CPU spike detection
SELECT server_id, time_bucket('1 minute', time) as bucket, max(value) as max_cpu
FROM metrics
WHERE metric_name = 'cpu_usage'
AND time > now() - INTERVAL '1 hour'
GROUP BY server_id, bucket
HAVING max(value) > 90
ORDER BY max_cpu DESC;
```

**Use Case 2: IoT Sensor Network**

Smart factory with 1,000 machines, each with 20 sensors (temperature, vibration, pressure). 10-second resolution. Need real-time alerting on anomalies, historical analysis for predictive maintenance.

**Why InfluxDB:**

- Purpose-built for time-series
- Line protocol for high-volume ingestion
- Downsample on ingestion (continuous queries)
- Built-in alerting
- Task automation

**Implementation Pattern:**

```influxql
-- Data model: measurements, tags (indexed), fields (values)
-- Write: line protocol
-- Example line protocol:
-- sensors,machine_id=m1,sensor_type=temperature value=85.5 1705334400000000000

-- Continuous query: downsample on ingest
CREATE CONTINUOUS QUERY "downsample_1m" ON "factory_db"
BEGIN
    SELECT 
        mean("value") as "value_avg",
        max("value") as "value_max",
        min("value") as "value_min",
        count("value") as "value_count"
    INTO "sensors_1m"
    FROM "sensors"
    GROUP BY time(1m), "machine_id", "sensor_type"
END

-- Retention: keep raw 24 hours, 1m aggregates 90 days
CREATE RETENTION POLICY "raw" ON "factory_db" DURATION 24h REPLICATION 1 DEFAULT
CREATE RETENTION POLICY "aggregated" ON "factory_db" DURATION 90d REPLICATION 1

-- Task: alert on temperature threshold
CREATE TASK "temp_alert" ON "factory_db"
BEGIN
    SELECT alert("machine_id", "sensor_type", "value", 90)
    FROM "sensors"
    WHERE "sensor_type" = 'temperature' AND "value" > 90
END
```

**Use Case 3: Financial Tick Data**

High-frequency trading system captures price ticks: symbol, price, volume, timestamp. 1 million ticks/second during market hours. Need real-time analytics, historical backtesting.

**Why QuestDB:**

- Time-series optimized column store
- Millisecond or nanosecond precision
- High compression
- SQL extensions for time-series
- Ingestion from Kafka, HTTP

**Implementation Pattern:**

```sql
-- Ticks table: append-only, partitioned by day
CREATE TABLE ticks (
    timestamp TIMESTAMP,
    symbol STRING,
    price DOUBLE,
    volume INT,
    side STRING
) TIMESTAMP(timestamp) PARTITION BY DAY;

-- Query: VWAP (volume-weighted average price) by minute
SELECT 
    timestamp_bucket(timestamp, '1m') as minute,
    symbol,
    sum(price * volume) / sum(volume) as vwap,
    sum(volume) as volume
FROM ticks
WHERE timestamp >= '2025-01-15'
AND symbol IN ('AAPL', 'GOOGL', 'MSFT')
GROUP BY minute, symbol
ORDER BY minute;

-- Query: latest price per symbol
SELECT LAST(price), LAST(timestamp)
FROM ticks
WHERE timestamp >= '2025-01-15 09:30:00'
SAMPLE BY LAST
LIMIT 20;
```

---

### 2.8 In-Memory Databases

#### Core Data Model

In-memory databases keep **all data in RAM**, providing **microsecond or nanosecond latency**. Primary use is caching, but some support persistence. Data structures vary — Redis offers rich data structures; others are simpler.

**Storage Architecture:**

- All data in RAM (DDR4/DDR5)
- Optionally persisted to disk (Redis AOF, Redis Cluster)
- RDD/persistence for durability
- Eviction policies: LRU, LFU, TTL
- Replica synchronization via network

**Query Model:**

- Key-value (GET/SET) or data structure commands
- Some support SQL (Hazelcast, SAP HANA)
- Transaction support (Redis MULTI/EXEC, Spark RDD)
- Lua scripting for atomic operations

#### Key Technical Strengths

1. **Ultra-Low Latency:** Memory access is nanoseconds. Sub-millisecond reads/writes guaranteed. Orders of magnitude faster than disk.

2. **High Throughput:** No disk I/O bottleneck. Can handle millions of operations/second.

3. **Rich Data Structures:** Redis offers strings, lists, sets, sorted sets, hashes, streams, bitmaps, hyperloglogs.

4. **Caching Simplification:** Direct cache layer without serialization to external caches. Shared cache for multiple applications.

5. **Real-Time Analytics:** In-memory aggregations for real-time dashboards without pre-computation.

#### Main Limitations and Trade-offs

1. **Memory Cost:** RAM is expensive. 1TB SSD ≈ $100; 1TB RAM ≈ $4,000+. Limits data volume.

2. **Data Durability Risk:** RAM loses data on power loss. Redis AOF provides durability but slower.

3. **Capacity Limits:** Can't store all enterprise data. Must select hot data carefully.

4. **Warm-up Time:** After restart, cache is cold. Application sees performance spike.

5. **Cluster Complexity:** Scaling memory requires cluster sharding. Adds operational complexity.

#### When to Use (Great Choice)

- **Hot data caching:** Frequently accessed data in front of slower databases
- **Session stores:** User sessions, shopping carts
- **Real-time leaderboards:** Sorted sets with frequent updates
- **Rate limiting:** Sliding window counters
- **Message queuing:** Pub/sub, streams for real-time messaging
- **In-memory analytics:** Real-time aggregations on recent data

#### When to Avoid (Bad Choice)

- **Data exceeding memory:** Not cost-effective for large datasets
- **Cold data storage:** Data accessed infrequently — disk cheaper
- **Durability-critical data:** Financial ledgers, critical records
- **Analytical workloads on historical data:** Columnar/OLAP better

#### Real-World Use Cases

**Use Case 1: Real-Time Session Store**

E-commerce platform stores shopping carts, recently viewed items, user preferences. 10 million concurrent users. Cart access is frequent, cart contents change often.

**Why Redis Cluster:**

- Sub-millisecond access
- TTL on cart (auto-expire after 30 days of inactivity)
- Hash for cart items (product_id → quantity)
- Horizontal scaling via clustering

**Implementation Pattern:**

```python
import redis
from redis.cluster import RedisCluster

# Redis Cluster for horizontal scaling
rc = RedisCluster(
    startup_nodes=[{"host": "redis-1", "port": 6379},
                   {"host": "redis-2", "port": 6379},
                   {"host": "redis-3", "port": 6379}],
    decode_responses=True
)

def get_cart(user_id: str) -> dict:
    """Get user's shopping cart"""
    cart_key = f"cart:{user_id}"
    cart_items = rc.hgetall(cart_key)
    return {k: int(v) for k, v in cart_items.items()}

def add_to_cart(user_id: str, product_id: str, quantity: int):
    """Add item to cart"""
    cart_key = f"cart:{user_id}"
    rc.hincrby(cart_key, product_id, quantity)
    rc.expire(cart_key, 30 * 24 * 60 * 60)  # 30-day TTL

def checkout(user_id: str):
    """Checkout: atomically get cart and clear"""
    cart_key = f"cart:{user_id}"
    pipe = rc.pipeline()
    
    # Get cart contents
    pipe.hgetall(cart_key)
    # Clear cart
    pipe.delete(cart_key)
    
    results = pipe.execute()
    cart = results[0]
    
    if cart:
        # Process order, clear cart
        process_order(user_id, cart)
        return {"status": "ordered", "items": len(cart)}
    return {"status": "empty"}
```

**Use Case 2: Distributed Locking**

Microservices need distributed locks for: preventing double-charge on payment, ensuring single consumer for message processing, coordinating deployment rolls.

**Why Redis:**

- SET with NX (only if not exists) for atomic lock acquisition
- EX for TTL (auto-release on failure)
- Lua scripting for multi-step operations

**Implementation Pattern:**

```python
import redis
import uuid
import time

class DistributedLock:
    def __init__(self, redis_client, lock_name, timeout=10):
        self.redis = redis_client
        self.lock_name = f"lock:{lock_name}"
        self.timeout = timeout
        self.token = str(uuid.uuid4())  # Unique lock holder ID
    
    def acquire(self, blocking=True, blocking_timeout=30):
        """Acquire lock with optional blocking"""
        start_time = time.time()
        
        while True:
            # SET NX: only set if not exists, return True if set
            acquired = self.redis.set(
                self.lock_name, 
                self.token, 
                nx=True,  # Only if not exists
                ex=self.timeout  # Auto-expire
            )
            
            if acquired:
                return True
            
            if not blocking:
                return False
            
            if time.time() - start_time >= blocking_timeout:
                return False
            
            time.sleep(0.01)  # Retry after 10ms
    
    def release(self):
        """Release lock only if we own it"""
        # Lua script: only delete if token matches (atomic)
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        self.redis.eval(lua_script, 1, self.lock_name, self.token)
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, *args):
        self.release()

# Usage
with DistributedLock(redis_client, "payment:order:12345") as lock:
    if lock.acquired():
        # Process payment - guaranteed single execution
        process_payment("order_12345")
    else:
        raise Exception("Could not acquire lock")
```

---

### 2.9 NewSQL / HTAP Systems

#### Core Data Model

NewSQL systems combine **distributed scalability** of NoSQL with **ACID guarantees** and **SQL interface** of traditional RDBMS. HTAP (Hybrid Transactional/Analytical Processing) systems run both OLTP and OLAP on the same platform.

**Storage Architecture:**

- Distributed storage (TiKV, Spanner) across multiple nodes
- Raft consensus for replication and consistency
- Separate storage engines for OLTP (row) vs OLAP (column)
- SQL layer on top (PostgreSQL, MySQL wire protocols)
- Automatic sharding and rebalancing

**Query Model:**

- Standard SQL (PostgreSQL or MySQL dialect)
- Full ACID transactions
- Distributed queries with parallel execution
- Columnar storage for analytical queries

#### Key Technical Strengths

1. **Horizontal Scaling:** Add nodes to scale. Automatic sharding. No application changes.

2. **ACID in Distributed Systems:** Distributed transactions with strong consistency. Raft/Paxos consensus.

3. **HTAP on Single Cluster:** Run transactional and analytical workloads simultaneously. No separate ETL.

4. **Cloud-Native:** Kubernetes operators, managed services. Automatic failover, self-healing.

5. **PostgreSQL/MySQL Compatibility:** Existing tools, ORMs, and expertise work. Easy migration path.

6. **Geo-Distribution:** Multi-region deployments with strong consistency (Google Spanner).

#### Main Limitations and Trade-offs

1. **Younger Technology:** Less battle-tested than traditional RDBMS. Fewer operational patterns.

2. **Performance Overhead:** Distributed consensus adds latency. Not as fast as single-node RDBMS for small workloads.

3. **Query Optimizer Maturity:** Less mature than PostgreSQL/MySQL optimizers. Complex queries may be slower.

4. **Cost:** Managed services expensive. Self-hosted requires expertise.

5. **Feature Gaps:** Some RDBMS features not yet implemented. Must check compatibility.

#### When to Use (Great Choice)

- **Global applications requiring strong consistency:** Multi-region with consistent reads
- **Microservices needing scalable RDBMS:** Each service can scale independently
- **HTAP workloads:** Need both transactions and analytics without data movement
- **Modernizing from monolithic RDBMS:** Need to scale but can't migrate off SQL

#### When to Avoid (Bad Choice)

- **Simple single-node workloads:** Traditional RDBMS faster and cheaper
- **Legacy systems with specific RDBMS features:** Check compatibility first
- **Small scale:** Overhead not justified

#### Real-World Use Cases

**Use Case 1: Global Financial Ledger**

Fintech company needs financial ledger that:

- Scales to handle 1 million transactions/second globally
- Strong consistency (no double-spend)
- Multi-region: users in US, EU, Asia with low latency
- ACID for transaction integrity

**Why TiDB:**

- Distributed ACID transactions (Percolator)
- MySQL compatible
- Geo-partitioning: user data in their region
- HTAP for real-time reporting on transactional data

**Implementation Pattern:**

```sql
-- Account table with geo-partitioning
CREATE TABLE accounts (
    id BIGINT UNSIGNED AUTO_RANDOM,
    user_id VARCHAR(64) NOT NULL,
    region VARCHAR(10) NOT NULL,
    balance DECIMAL(20, 2) NOT NULL DEFAULT 0,
    currency VARCHAR(3) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    KEY idx_user_region (user_id, region)
) PARTITION BY LIST (region) (
    PARTITION us VALUES IN ('US', 'CA', 'MX'),
    PARTITION eu VALUES IN ('UK', 'DE', 'FR', 'NL'),
    PARTITION apac VALUES IN ('JP', 'SG', 'AU')
);

-- Transaction with balance check (distributed)
START TRANSACTION;
UPDATE accounts 
SET balance = balance - 100.00 
WHERE user_id = 'user_123' AND region = 'US' AND balance >= 100.00;

IF ROW_COUNT() = 0 THEN
    ROLLBACK;
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Insufficient funds';
END IF;

INSERT INTO transactions (from_user, to_user, amount)
VALUES ('user_123', 'user_456', 100.00);
COMMIT;

-- Real-time analytics on same cluster
-- HTAP: no need for separate analytical database
SELECT 
    region,
    currency,
    count(*) as transaction_count,
    sum(amount) as total_volume,
    avg(amount) as avg_amount
FROM transactions
WHERE created_at >= NOW() - INTERVAL 1 DAY
GROUP BY region, currency;
```

**Use Case 2: Real-Time Operational Analytics**

E-commerce platform needs to run both:

- Transactional: orders, inventory, payments (OLTP)
- Analytical: real-time dashboards for ops team (OLAP)

Without ETL to separate data warehouse.

**Why SingleStore:**

- MySQL compatible
- Rowstore for OLTP, columnstore for OLAP
- Auto-materialized views
- Real-time ingestion from Kafka

**Implementation Pattern:**

```sql
-- Rowstore table for transactional (hot data)
CREATE TABLE orders (
    order_id BIGINT AUTO_INCREMENT,
    customer_id BIGINT,
    order_total DECIMAL(10,2),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (order_id),
    KEY idx_customer (customer_id),
    KEY idx_status (status, created_at)
) ENGINE=InnoDB;

-- Columnstore table for analytical (aggregations)
CREATE TABLE orders_analytics (
    order_id BIGINT,
    customer_id BIGINT,
    order_total DECIMAL(10,2),
    status VARCHAR(20),
    created_at TIMESTAMP,
    order_hour TIMESTAMP NOT NULL,
    order_date DATE NOT NULL
) ENGINE=Columnstore;

-- Auto-materialized view for real-time dashboard
CREATE MATERIALIZED VIEW mv_orders_hourly
AS SELECT 
    order_hour,
    status,
    count(*) as order_count,
    sum(order_total) as revenue
FROM orders_analytics
GROUP BY order_hour, status;

-- Real-time query: orders in last hour
SELECT status, count(*), sum(order_total)
FROM orders
WHERE created_at >= NOW() - INTERVAL 1 HOUR
GROUP BY status;
```

---

### 2.10 Vector Databases

#### Core Data Model

Vector databases store **high-dimensional vectors** (embeddings) — numerical representations of data (text, images, audio) produced by ML models. Enable **semantic similarity search** — finding items "similar in meaning" not just exact matches.

**Storage Architecture:**

- Vectors stored with metadata
- Approximate Nearest Neighbor (ANN) indexes: HNSW, IVF, PQ
- HNSW: graph-based, high recall, moderate memory
- IVF: inverted index, faster at scale, slightly less accurate
- Product Quantization (PQ): compressed vectors for memory efficiency

**Query Model:**

- Similarity search: find nearest vectors (cosine, euclidean, dot product)
- Metadata filtering: filter results by metadata before/after vector search
- Hybrid search: combine vector + keyword (BM25) search
- Pagination with score thresholds

#### Key Technical Strengths

1. **Semantic Search:** Find results by meaning, not keyword matching. "Apple the fruit" vs "Apple the company."

2. **ANN Performance:** Sub-second search across billions of vectors. Traditional databases can't.

3. **Metadata Filtering:** Filter by category, date, author before/after vector search.

4. **Hybrid Search:** Combine vector similarity with keyword search for better relevance.

5. **Dedicated for Embeddings:** Purpose-built for ML inference at scale.

#### Main Limitations and Trade-offs

1. **Single-Model Use:** Outside RAG/similarity search, general-purpose databases better.

2. **Index Build Time:** Building HNSW indexes on billions of vectors takes hours.

3. **Memory Footprint:** Vectors consume significant memory. Quantization helps.

4. **Not ACID:** Most vector DBs sacrifice ACID for speed. Check consistency requirements.

5. **Evolvability:** Index types and embeddings models can change. Re-indexing expensive.

#### When to Use (Great Choice)

- **RAG (Retrieval-Augmented Generation):** Store document chunks, retrieve relevant context for LLM
- **Semantic search:** Find products, articles, content by meaning
- **Image/video similarity:** Find similar images, reverse image search
- **Recommendation systems:** Item similarity, collaborative filtering via embeddings
- **Fraud detection:** Find similar patterns, anomaly detection

#### When to Avoid (Bad Choice)

- **Simple keyword search:** Use Elasticsearch, OpenSearch, or full-text search
- **Exact match queries:** Use RDBMS or document store
- **Small datasets:** Overhead not worth it for < 10K vectors

#### Real-World Use Cases

**Use Case 1: RAG for Customer Support Chatbot**

Customer support system uses LLM + retrieval. Knowledge base: 50,000 support articles, policies, FAQs. LLM answers user questions by retrieving relevant context.

**Why Qdrant:**

- Open source, self-hostable
- Rust implementation, high performance
- PostgreSQL-like filtering
- Easy integration with LangChain, LlamaIndex
- Pay for vectors with quantization

**Implementation Pattern:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np

client = QdrantClient(host="localhost", port=6333)

# Create collection with HNSW index
client.recreate_collection(
    collection_name="support_articles",
    vectors_config=VectorParams(
        size=768,  # Embedding dimension (e.g., OpenAI text-embedding-3-large)
        distance=Distance.COSINE
    )
)

# Index articles with metadata
def index_articles(articles: list[dict]):
    points = []
    for i, article in enumerate(articles):
        # Generate embedding (example using OpenAI)
        embedding = generate_embedding(article["content"])
        
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={
                "title": article["title"],
                "content": article["content"],
                "category": article["category"],
                "last_updated": article["updated"]
            }
        ))
    
    client.upsert(
        collection_name="support_articles",
        points=points
    )

# Query: semantic search + metadata filter
def search_articles(query: str, category: str = None, limit: int = 5):
    query_embedding = generate_embedding(query)
    
    # Build filter
    must_filters = []
    if category:
        must_filters.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )
    
    results = client.search(
        collection_name="support_articles",
        query_vector=query_embedding,
        query_filter=Filter(must=must_filters) if must_filters else None,
        limit=limit,
        with_payload=True
    )
    
    return [
        {"title": r.payload["title"], 
         "content": r.payload["content"],
         "score": r.score}
        for r in results
    ]
```

**Use Case 2: Product Recommendations**

E-commerce recommendation system: find similar products based on embedding similarity. Embeddings encode: product description, category, user behavior.

**Why Pinecone:**

- Managed service, no ops
- Serverless option for variable workloads
- Real-time indexing
- Global replicas for low latency

**Implementation Pattern:**

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
index = pc.Index("product-recommendations")

def find_similar_products(product_id: str, top_k: int = 10):
    # Get embedding for the query product
    product_vec = get_product_embedding(product_id)
    
    # Search
    results = index.query(
        vector=product_vec,
        top_k=top_k,
        include_metadata=True,
        filter={"status": {"$eq": "active"}}  # Only active products
    )
    
    return [
        {"product_id": r.id, "score": r.score, "name": r.metadata["name"]}
        for r in results.matches
    ]

def update_product_recommendations(products: list[dict]):
    # Batch upsert product embeddings
    vectors = [
        {
            "id": p["id"],
            "values": p["embedding"],
            "metadata": {"name": p["name"], "category": p["category"]}
        }
        for p in products
    ]
    
    index.upsert(vectors=vectors)
```

**Use Case 3: Image Similarity Search**

Fashion e-commerce: customers upload product photo, find visually similar products in catalog. Pre-compute image embeddings using CNN/ViT, store in vector DB.

**Why Milvus:**

- Open source at scale
- Multiple index types (HNSW, IVF, ANNOY)
- Distributed for billions of vectors
- Strong supporting ecosystem

**Implementation Pattern:**

```python
from milvus import MilvusClient

client = MilvusClient(uri="milvus.db")

# Create collection
client.create_collection(
    collection_name="fashion_products",
    dimension=512,  # Image embedding dimension
    metric_type="COSINE",
    index_type="HNSW"
)

def search_similar_images(query_image_path: str, top_k: int = 20):
    # Generate embedding from image
    query_embedding = image_to_embedding(query_image_path)
    
    # Search
    results = client.search(
        collection_name="fashion_products",
        data=[query_embedding],
        limit=top_k,
        output_fields=["product_id", "image_url", "category", "price"]
    )
    
    return results[0]  # Returns list of matches

def index_product_images(products: list[dict]):
    # Batch insert product images
    data = [
        {
            "id": str(p["id"]),
            "embedding": p["embedding"],
            "product_id": p["product_id"],
            "image_url": p["image_url"],
            "category": p["category"],
            "price": p["price"]
        }
        for p in products
    ]
    
    client.insert(
        collection_name="fashion_products",
        data=data
    )
```

---

## 3. Practical Comparison Tables

### 3.1 Database Family Comparison

| Aspect | Relational | Document | Key-Value | Wide-Column | Graph | Columnar/OLAP | Time-Series | In-Memory | Vector |
|-------|------------|----------|-----------|-------------|-------|---------------|-------------|-----------|--------|
| **Data Model** | Tables/rows/columns | JSON documents | Key→Value | Column families | Nodes/edges | Column-oriented | Timestamped points | Memory-resident | Embeddings+metadata |
| **Primary Access** | SQL queries | Document queries | Key lookup | Row key, range scan | Graph traversal | SQL analytics | Time range | Key/value | Similarity search |
| **Schema** | Fixed | Flexible | None | Flexible | Flexible | Fixed | Fixed | Flexible | Fixed for vectors |
| **Scalability** | Vertical (sharding) | Horizontal sharding | Horizontal | Massive horizontal | Limited | Massive | Horizontal | Horizontal clusters | Horizontal |
| **Consistency** | ACID strong | Eventual/strong | Eventual | Tunable | Eventually consistent | Eventually consistent | Eventual | Tunable | Tunable |
| **Joins** | Native | Limited/denormalized | No | No | Native | Limited | No | No | No |
| **Query Flexibility** | High | Medium | Very low | Low | Medium | High | Medium | Low | Low |
| **Write Throughput** | Medium | High | Very high | Very high | Medium | Medium | High | Very high | Medium |
| **Read Latency** | ms | ms | μs | ms | ms | ms-seconds | ms | μs | ms |
| **Typical Use** | Transactions | Content, profiles | Cache, sessions | IoT, events | Relationships | Analytics | Metrics | Cache, real-time | RAG, similarity |

### 3.2 OLTP vs OLAP vs HTAP

| Aspect | OLTP | OLAP | HTAP |
|--------|------|------|------|
| **Full Form** | Online Transaction Processing | Online Analytical Processing | Hybrid Transactional/Analytical Processing |
| **Primary Purpose** | Day-to-day operational data | Historical analysis, business intelligence | Unified transactional + analytical |
| **Query Type** | Simple, point queries, updates | Complex aggregations, multi-join | Both in single system |
| **Data Volume** | GB-TB | TB-PB | TB-PB |
| **Query Frequency** | Thousands/second | Tens/hour (heavy) | Mixed |
| **Response Time** | < 100ms | Seconds-minutes | Varies |
| **Data Freshness** | Current | Historical (hours old) | Real-time |
| **Schema** | Normalized | Denormalized | Both normalized + denormalized |
| **Indexes** | Many, B-tree | Few, specialized | Multiple index types |
| **Typical Queries** | INSERT, UPDATE, SELECT single row | GROUP BY, JOIN, aggregate | Mixed |
| **Example Systems** | PostgreSQL, MySQL, Oracle | Snowflake, BigQuery, ClickHouse | TiDB, SingleStore, CockroachDB |
| **Example Workloads** | Order processing, payment, login | Monthly reports, dashboards, ad-hoc analysis | Real-time analytics, operational BI |

### 3.3 Vector Database Comparison

| Aspect | Qdrant | Pinecone | Weaviate | Milvus | pgvector |
|--------|--------|----------|-----------|--------|----------|
| **Type** | Open-source + Cloud | Managed (SaaS) | Open-source + Cloud | Open-source | PostgreSQL extension |
| **Language** | Rust | Python/Go | Go | Go | C |
| **Index Types** | HNSW, IVF | HNSW, Brute-force | HNSW, BM25 | HNSW, IVF, ANNOY | HNSW, IVF |
| **Deployment** | Self-hosted, Cloud | Serverless, Dedicated | Self-hosted, Cloud | Self-hosted, Cloud | Self-hosted |
| **Max Dimensions** | 65,536 | 20,000+ | 65,536 | 32,768 | 2,000 |
| **Scalability** | Excellent (sharding) | Excellent (serverless) | Good (replication) | Excellent (distributed) | Limited to PostgreSQL |
| **Filtering** | Pre/Post filter | Pre-filter | Pre/Post filter | Pre-filter | SQL WHERE clause |
| **Hybrid Search** | Yes (BM25) | No (use inference) | Yes (BM25, rerank) | Yes (sparse) | Yes (custom) |
| **Latency (1M vectors)** | < 10ms | < 20ms | < 15ms | < 15ms | < 50ms |
| **Operations** | Medium | Low (managed) | Medium | High | Low (same as Postgres) |
| **Integration** | LangChain, LlamaIndex | LangChain, Python | LangChain, GraphQL | LangChain, PyTorch | Django, SQLAlchemy |
| **Best For** | Self-hosted, performance | Managed, ease-of-use | GraphQL, hybrid | Scale, flexibility | Existing Postgres users |

### 3.4 Relational vs Document vs Key-Value Decision Matrix

| Requirement | Best Choice | Alternative |
|-------------|-------------|-------------|
| **ACID transactions critical** | PostgreSQL, MySQL | CockroachDB (distributed) |
| **Flexible schema** | MongoDB | PostgreSQL JSONB |
| **Massive write scale** | Cassandra, DynamoDB | ClickHouse (append) |
| **Rich queries, relationships** | PostgreSQL, Neo4j | ArangoDB (multi-model) |
| **Sub-millisecond cache** | Redis | Memcached |
| **Simple key-value, managed** | DynamoDB | Redis (self-hosted) |
| **Full-text search** | Elasticsearch, PostgreSQL FTS | MongoDB Atlas Search |
| **Geospatial** | PostgreSQL PostGIS | MongoDB Geospatial |
| **Time-series at scale** | TimescaleDB, ClickHouse | InfluxDB |
| **Graph relationships** | Neo4j | Amazon Neptune |
| **Vector similarity** | Pinecone, Qdrant | pgvector (in Postgres) |
| **Analytics, data warehouse** | Snowflake, BigQuery | ClickHouse |

---

## 4. Polyglot Persistence and Architecture Patterns

### 4.1 Concept: Why Polyglot Persistence?

Modern applications have diverse data requirements that no single database optimally satisfies:

- **Transactional integrity** requires ACID (RDBMS)
- **Flexible product catalogs** need schema flexibility (Document)
- **Session data** needs microsecond latency (In-Memory)
- **User behavior analytics** needs columnar storage (OLAP)
- **Similarity search** needs vector indexes (Vector DB)

**Polyglot Persistence** = using multiple specialized databases, each optimized for specific data types and access patterns.

### 4.2 Pattern 1: SaaS Application Architecture

Modern SaaS applications typically combine:

| Data Type | Database | Justification |
|-----------|----------|----------------|
| **Users, tenants, roles** | PostgreSQL | ACID for auth, complex queries, RLS |
| **Session data, tokens** | Redis | Microsecond access, TTL support |
| **Event logs, audit** | MongoDB | Flexible schema for varied events |
| **Analytics, metrics** | ClickHouse | Columnar for aggregations |
| **Vector search (AI)** | Qdrant | Semantic search capabilities |

**Data Flow:**

```
User Request → API Gateway → Auth Service (PostgreSQL)
                    ↓
              Session Cache (Redis)
                    ↓
           Business Logic + Data Access
           ├── Read user profile (PostgreSQL)
           ├── Read product catalog (MongoDB)
           ├── Cache frequently accessed (Redis)
           └── Log analytics event (ClickHouse)
                    ↓
              Response + Background Jobs
              ├── Batch analytics to ClickHouse
              └── Vector search for recommendations (Qdrant)
```

**Detailed Flow:**

1. **Authentication:** User login → validate credentials against PostgreSQL → create session token in Redis (24-hour TTL)
2. **Profile Access:** Subsequent requests use session token from Redis for microsecond auth → load full profile from PostgreSQL on cache miss
3. **Business Operations:** Orders written to PostgreSQL for ACID → session updated in Redis
4. **Analytics Events:** Each page view, button click → write to ClickHouse (async, fire-and-forget)
5. **Search:** Product search → PostgreSQL full-text + Qdrant semantic hybrid
6. **Recommendations:** Background job queries Qdrant → results cached in Redis

### 4.3 Pattern 2: E-Commerce Platform

E-commerce has some of the most demanding data requirements:

| Data Type | Database | Justification |
|-----------|----------|----------------|
| **Orders, payments, inventory** | PostgreSQL | ACID critical for financial |
| **Product catalog** | MongoDB | Flexible attributes per product type |
| **User sessions, cart** | Redis | Real-time read/write, TTL |
| **Product search** | Elasticsearch | Full-text + fuzzy match + facets |
| **Analytics, reporting** | BigQuery/Snowflake | Massive aggregations |
| **Recommendations** | Pinecone | Vector similarity |
| **Images, media** | S3 + CloudFront | Blob storage, CDN |

**Data Flow:**

```
Customer Browse
    ↓
[1] Product Search (Elasticsearch) → Relevance scoring
[2] Product Details (MongoDB) → Flexible product attributes
[3] Inventory Check (PostgreSQL) → Real-time stock
[4] Add to Cart (Redis) → Session-scoped cart
    ↓
Checkout Flow
    [5] Cart → Order (PostgreSQL, ACID transaction)
    [6] Payment (PostgreSQL + external payment gateway)
    [7] Inventory Decrement (PostgreSQL)
    [8] Session Clear (Redis)
    ↓
Post-Purchase
    [9] Analytics Event (ClickHouse - async)
    [10] Recommendation Update (Pinecone - background)
    [11] Email Notification (triggered from PostgreSQL)
```

**Specific Examples:**

- **Product Search:** Elasticsearch indexes product name, description, attributes. Fuzzy matching, synonyms, facets (category, price range). Results ranked by relevance.
- **Inventory Management:** PostgreSQL rows locked during checkout to prevent overselling. Real-time stock display updated via change data capture (CDC) to Redis.
- **Recommendations:** User behavior (views, purchases) → generate embeddings → store in Pinecone. "Similar products" query runs on page load, results cached in Redis.
- **Analytics:** Every order, view, cart action → write to ClickHouse via Kafka. Marketing dashboards query ClickHouse. No impact on transactional PostgreSQL.

### 4.4 Pattern 3: Advanced RAG Platform

AI applications with RAG (Retrieval-Augmented Generation) have specialized requirements:

| Data Type | Database | Justification |
|-----------|----------|----------------|
| **Users, tenants, permissions** | PostgreSQL | Multi-tenant isolation, RLS |
| **Documents (raw)** | MongoDB / S3 | Flexible document storage |
| **Chunks + embeddings** | Qdrant / Pinecone | Vector similarity search |
| **Metadata, tags** | PostgreSQL | Structured queries on metadata |
| **Chat history** | MongoDB | Flexible conversation schema |
| **Monitoring, logs** | TimescaleDB | Time-series metrics |
| **Cache** | Redis | LLM response caching |

**Data Flow:**

```
User Query
    ↓
[1] Auth + Tenant Check (PostgreSQL)
    ↓
[2] Embed Query (embedding model)
    ↓
[3] Vector Search (Qdrant)
    ├── Hybrid: keyword filter (PostgreSQL metadata)
    └── Re-rerank (cross-encoder)
    ↓
[4] Retrieve Context Chunks
    ↓
[5] Build Prompt with Context + Chat History (MongoDB)
    ↓
[6] LLM Inference (cache check Redis → if miss, call LLM)
    ↓
[7] Stream Response + Log Metrics (TimescaleDB)
```

**Detailed Components:**

1. **Document Ingestion Pipeline:**
   ```
   Document Upload → Chunking (LangChain) → Embedding (OpenAI/Cohere) 
   → Store in Qdrant (vector) + MongoDB (content) + PostgreSQL (metadata)
   ```

2. **Multi-Tenant Isolation:**
   - PostgreSQL RLS policies enforce tenant_id filter
   - Qdrant uses tenant_id as metadata filter on every query
   - Cache keys prefixed by tenant_id

3. **Hybrid Search Implementation:**
   - Vector search returns top-100 candidates
   - PostgreSQL metadata filter applied in-memory
   - Re-ranker (cross-encoder) scores top-20
   - Final results to LLM

4. **Monitoring:**
   - Query latency, token usage, cache hit rate → TimescaleDB
   - Alert on latency > P99 threshold
   - Dashboard shows retrieval quality metrics (recall, precision)

---

## 5. Learning Roadmap

This roadmap is designed for an experienced engineer already familiar with PostgreSQL, MongoDB, Redis, and basic vector databases. The goal is production-oriented depth across all database families.

### Phase 1: Relational Database Mastery (Weeks 1-3)

**Key Concepts to Master:**

- **Normalization forms:** 1NF through BCNF. When to denormalize for performance.
- **Indexing strategies:** B-tree, GiST, GIN, BRIN. Index-only scans, covering indexes.
- **Transactions and isolation levels:** READ COMMITTED vs SERIALIZABLE. Locking, deadlocks.
- **Query optimization:** EXPLAIN ANALYZE, query planning, statistics.
- **Advanced SQL:** Window functions, CTEs, recursive queries, lateral joins.
- **Replication:** Streaming replication, logical replication, multi-master.
- **Partitioning:** Range, list, hash partitioning. Partition pruning.

**Resources:**

- PostgreSQL Documentation: Query Planning, Locking, Partitioning
- "The Art of PostgreSQL" by François Husson
- Use The Index, Luke (use-the-index-luke.com)

**Mini-Project 1A: Transactional E-Commerce Backend**

Build order processing system with:
- Order creation with inventory reservation
- Payment processing (simulated)
- Concurrent order handling
- Deadlock detection and handling

```python
# Key skills: ACID transactions, row-level locking, isolation levels
# Deliverable: REST API handling 100+ concurrent orders/second
# Tech: PostgreSQL, FastAPI, SQLAlchemy async
```

**Mini-Project 1B: Query Optimization Lab**

Given a set of slow queries:
- Use EXPLAIN ANALYZE to diagnose
- Add appropriate indexes
- Refactor schema if needed
- Benchmark before/after

```sql
-- Example challenge query
SELECT o.id, o.created_at, c.name, SUM(oi.quantity * p.price) as total
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE o.created_at > '2024-01-01'
GROUP BY o.id, c.name
HAVING SUM(oi.quantity * p.price) > 1000
ORDER BY total DESC
LIMIT 100;
```

### Phase 2: NoSQL Deep Dive (Weeks 4-6)

**Key Concepts to Master:**

**MongoDB:**
- Document modeling patterns: embedded vs referenced
- Aggregation pipeline optimization
- Sharding strategies: shard key selection
- Change streams for CDC

**Redis:**
- Data structures: strings, lists, sets, sorted sets, hashes, streams, bitmaps
- Persistence: RDB vs AOF, hybrid approaches
- Clustering: Redis Cluster, hash slots
- Lua scripting for atomic operations

**Cassandra (Overview):**
- CQL data modeling: query-first design
- Partition keys, clustering columns
- Tunable consistency
- Compaction strategies

**Graph Databases (Overview):**
- Property graph model
- Cypher basics
- Graph modeling patterns

**Resources:**

- MongoDB University: M001, M121, M310
- Redis Documentation: Data Types, Clustering
- Neo4j Graph Data Modeling

**Mini-Project 2A: Event Logging System**

Build event collection API:
- High-throughput event ingestion (10K events/second)
- Flexible event schema (different fields per event type)
- Aggregation queries by time, user, event type

```python
# Key skills: MongoDB aggregation, indexing, TTL
# Deliverable: Ingest events, query by user/time/type with < 100ms p99
# Tech: MongoDB, Python async API
```

**Mini-Project 2B: Real-Time Leaderboard**

Implement global and per-region leaderboards:
- Sorted set operations
- Real-time rank updates
- "Players around me" queries

```python
# Key skills: Redis sorted sets, atomic operations
# Deliverable: Update scores, query top-100, get player rank
# Tech: Redis, Python
```

### Phase 3: OLAP and Time-Series (Weeks 7-9)

**Key Concepts to Master:**

**Columnar/OLAP:**
- Columnar storage fundamentals
- Data warehouse vs data lake
- Partitioning, clustering, compression
- ETL/ELT patterns
- Materialized views

**ClickHouse:**
- MergeTree engine family
- Query optimization
- Distributed queries
- Materialized views

**Time-Series:**
- Time-series data modeling
- Retention policies
- Downsampling/rollup
- Compression

**TimescaleDB / InfluxDB:**
- Hypertable/chunk management
- Continuous aggregates
- Compression settings

**Resources:**

- ClickHouse Documentation
- TimescaleDB Documentation
- "ClickHouse: The Definitive Guide"

**Mini-Project 3A: Analytics Pipeline**

Export PostgreSQL orders to ClickHouse:
- Set up ClickHouse
- Design analytical schema
- Build export pipeline (Airbyte or custom)
- Create dashboards

```python
# Key skills: Columnar modeling, ETL, aggregations
# Deliverable: Dashboards with sub-second queries on 10M+ orders
# Tech: ClickHouse, Metabase/Streamlit
```

**Mini-Project 3B: Metrics Dashboard**

Build infrastructure monitoring:
- Ingest metrics via HTTP
- Query by time range, host, metric
- Automatic retention
- Grafana visualization

```python
# Key skills: Time-series modeling, downsampling, retention
# Deliverable: Real-time dashboard, 30-day retention, < 10ms queries
# Tech: TimescaleDB or InfluxDB, Grafana
```

### Phase 4: NewSQL and HTAP (Weeks 10-11)

**Key Concepts to Master:**

- Distributed SQL architecture
- HTAP concepts and trade-offs
- Distributed transactions
- Geo-distribution
- Consistency models

**Systems to Explore:**

- **TiDB:** MySQL compatible, distributed
- **CockroachDB:** PostgreSQL compatible, geo-distributed
- **SingleStore:** MySQL compatible, HTAP

**Resources:**

- TiDB Documentation
- CockroachDB Architecture docs

**Mini-Project 4: Distributed SQL Experiment**

Deploy TiDB or CockroachDB:
- Run transactional workload
- Run analytical workload simultaneously
- Observe resource sharing

```python
# Key skills: Distributed SQL, HTAP, resource management
# Deliverable: Mixed OLTP/OLAP workload on single cluster
# Tech: TiDB or CockroachDB (managed tier)
```

### Phase 5: Vector Databases and RAG (Weeks 12-13)

**Key Concepts to Master:**

- Embeddings fundamentals
- ANN indexes: HNSW, IVF, PQ
- Vector search + metadata filtering
- Hybrid search
- RAG architecture patterns

**Resources:**

- Qdrant Documentation
- Pinecone Documentation
- LangChain/LlamaIndex vector stores

**Mini-Project 5: End-to-End RAG**

Build document Q&A system:
- Document ingestion pipeline
- Chunking and embedding
- Vector storage
- Query with hybrid search

```python
# Key skills: Vector search, embeddings, RAG architecture
# Deliverable: Q&A bot with > 85% retrieval accuracy
# Tech: Qdrant or Pinecone, LangChain, OpenAI
```

---

## 6. Concrete Mini-Projects

### Project 1: Order Management System with PostgreSQL

**Description:** Classic transactional system handling orders, customers, products, inventory.

**Tech Stack:** PostgreSQL, Python/FastAPI, SQLAlchemy

**Learning Objectives:**

- ACID transaction handling
- Complex joins and aggregations
- Indexing strategy
- Row-level security for multi-tenancy

**Directions:**

1. Design normalized schema: customers, orders, order_items, products, inventory
2. Implement order placement with inventory locking
3. Add queries: orders by customer, revenue by product/category, daily totals
4. Add indexes, use EXPLAIN ANALYZE to verify
5. Add RLS for multi-tenant isolation

---

### Project 2: Flexible Event Logging with MongoDB

**Description:** Collect and query application events with varying schemas.

**Tech Stack:** MongoDB, Python/FastAPI

**Learning Objectives:**

- Document modeling
- Flexible schema design
- Aggregation pipeline
- TTL indexes

**Directions:**

1. Create events collection with time-based partitioning
2. Implement event ingestion API supporting variable event types
3. Build aggregation queries: events per user, event distributions, time-based metrics
4. Set up TTL for automatic old event deletion

---

### Project 3: API Caching Layer with Redis

**Description:** Add caching to REST API for hot data.

**Tech Stack:** Redis, Python/FastAPI

**Learning Objectives:**

- Cache-aside pattern
- Redis data structures
- TTL and cache invalidation
- Distributed rate limiting

**Directions:**

1. Implement GET/PUT endpoints with Redis caching
2. Add cache invalidation on data updates
3. Implement rate limiting using sliding window
4. Add caching statistics endpoint (hit rate, etc.)

---

### Project 4: IoT Data Ingestion with Cassandra

**Description:** High-throughput sensor data collection.

**Tech Stack:** Apache Cassandra, Python

**Learning Objectives:**

- Wide-column modeling
- Partition key design
- CQL query patterns
- Time-series on Cassandra

**Directions:**

1. Design table for sensor data with partition by sensor_id + time bucket
2. Implement high-throughput write path
3. Query by sensor, time range
4. Add TTL for data expiration

---

### Project 5: Analytics Dashboard with ClickHouse

**Description:** Build analytical queries on order data.

**Tech Stack:** ClickHouse, Python, Streamlit

**Learning Objectives:**

- Columnar storage
- Aggregation optimization
- Materialized views
- Dashboard building

**Directions:**

1. Load sample e-commerce data
2. Create MergeTree table with partitioning
3. Build materialized views for common aggregations
4. Create Streamlit dashboard with interactive charts

---

### Project 6: Metrics Monitoring with TimescaleDB

**Description:** Infrastructure monitoring with automatic retention.

**Tech Stack:** TimescaleDB, Grafana, Python

**Learning Objectives:**

- Time-series modeling
- Hypertables and chunks
- Continuous aggregates
- Retention policies

**Directions:**

1. Create hypertable for metrics
2. Implement metric ingestion API
3. Add continuous aggregates for hourly/daily rollups
4. Set up retention policy
5. Build Grafana dashboard

---

### Project 7: Social Graph with Neo4j

**Description:** Social network with friend recommendations.

**Tech Stack:** Neo4j, Python

**Learning Objectives:**

- Graph modeling
- Cypher queries
- Graph algorithms
- Path finding

**Directions:**

1. Create nodes for users
2. Add relationship types (FRIEND, FOLLOW)
3. Implement "suggest friends" using Cypher
4. Build path-finding queries (shortest path)

---

### Project 8: RAG with Vector Database

**Description:** Document Q&A using retrieval.

**Tech Stack:** Qdrant/Pinecone, LangChain, OpenAI

**Learning Objectives:**

- Embeddings
- Vector search
- RAG architecture
- Hybrid search

**Directions:**

1. Implement document ingestion (chunk, embed, store)
2. Build query endpoint (embed, search, retrieve)
3. Add metadata filtering
4. Combine with LLM for full RAG pipeline

---

### Project 9: Polyglot SaaS Backend

**Description:** Combine multiple databases in one application.

**Tech Stack:** PostgreSQL + Redis + MongoDB + ClickHouse

**Learning Objectives:**

- Polyglot persistence
- Data flow between databases
- Transaction coordination
- Architecture design

**Directions:**

1. Design data flow: which data to which DB
2. Implement user/auth in PostgreSQL
3. Add session cache in Redis
4. Store flexible metadata in MongoDB
5. Add analytics in ClickHouse
6. Create API that coordinates across all

---

### Project 10: Distributed SQL Experiment

**Description:** Explore distributed transactions and HTAP.

**Tech Stack:** TiDB or CockroachDB

**Learning Objectives:**

- Distributed SQL
- HTAP workloads
- Consistency vs performance

**Directions:**

1. Deploy TiDB cluster (managed)
2. Run YCSB benchmark
3. Execute mixed OLTP/OLAP queries
4. Observe resource utilization

---

## 7. Study Methodology for Professionals

### 7.1 Balancing Theory and Practice

**The 70/30 Rule:** Spend 30% of time on theory, 70% on hands-on practice.

**Theory (30%):**

- Read documentation for the database
- Understand internals: storage engines, query planners
- Study query optimization techniques
- Review case studies and architecture patterns

**Practice (70%):**

- Build actual projects with each database type
- Run benchmarks comparing approaches
- Debug performance issues
- Deploy to production-like environments

### 7.2 Documenting What You Learn

**Note Structure:**

```
# [Database Type]: [Topic]

## Key Concepts
- Concept 1: Definition, when to use
- Concept 2: ...

## Code Examples
```sql
-- Example query pattern
```

## Trade-offs
- When to use: ...
- When to avoid: ...

## Common Pitfalls
- Pitfall 1: How to avoid

## Links
- Official docs
- Blog posts
```

**Repository Organization:**

```
databases-learning/
├── postgresql/
│   ├── notes/
│   ├── projects/
│   └── benchmarks/
├── mongodb/
├── redis/
├── clickhouse/
└── ...
```

### 7.3 Testing Understanding

**Self-Assessment Questions:**

- For a given scenario, which database would you choose and why?
- What are the trade-offs of your choice?
- What alternatives exist and when would they be better?

**Exercises:**

- Take an existing project and identify database selection issues
- Design database architecture for a given requirements document
- Benchmark two approaches, analyze results
- Debug a slow query, explain the fix

**Benchmarking Practice:**

- Compare query times across databases for the same workload
- Measure scaling behavior
- Test failure scenarios
- Profile resource consumption

### 7.4 Iterative Improvement

**Refactoring Loop:**

1. Implement initial schema/queries
2. Profile with realistic data volume
3. Identify bottlenecks (EXPLAIN ANALYZE, slow query log)
4. Apply optimization (index, denormalize, restructure)
5. Verify improvement with benchmarks
6. Document the fix

**Tools for Profiling:**

- PostgreSQL: pg_stat_statements, EXPLAIN ANALYZE
- MongoDB: explain(), profiler
- Redis: INFO, SLOWLOG, MEMORY
- ClickHouse: system.query_log, system.metric_log

**Schema Evolution:**

- Start simple
- Add complexity as requirements demand
- Test with production-like data volumes
- Document schema decisions and rationale

---

## 8. Summary and Next Steps

This guide has provided:

1. **High-level map** of all major database families with key products
2. **Deep dives** into each family: data models, strengths, limitations, use cases
3. **Comparison tables** for quick decision-making
4. **Polyglot persistence patterns** for real-world architectures
5. **Learning roadmap** with 5 phases over ~13 weeks
6. **10 concrete mini-projects** covering all database types
7. **Study methodology** for professional learning

### Immediate Next Steps:

1. **Assess Current Knowledge:** Which database families are you already proficient in?
2. **Start Phase 1:** Begin with relational mastery if needed
3. **Choose First Project:** Start with Project 1 (Order Management) or adapt to your current work
4. **Set Up Environment:** Spin up databases locally or use free tiers
5. **Join Communities:** Database-specific forums, subreddits, Discord servers

### Resources for Continued Learning:

- **Database Documentation:** PostgreSQL, MongoDB, Redis, ClickHouse, TiDB
- **Books:** "Designing Data-Intensive Applications" (Kleppmann)
- **Courses:** Coursera, Udemy, official vendor courses
- **Practice:** LeetCode Database section, HackerRank SQL

The database landscape continues evolving. Stay current with:

- New database releases and features
- Cloud-managed service capabilities
- AI/ML integration patterns (your existing vector DB knowledge is key)

Build, benchmark, and iterate. Mastery comes from practical experience.

---

*This curriculum integrates into the AI-Mastery-2026 framework, building on existing knowledge of PostgreSQL, MongoDB, Redis, and vector databases to create comprehensive database systems understanding.*
