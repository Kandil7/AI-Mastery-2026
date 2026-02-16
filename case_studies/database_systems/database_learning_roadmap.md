# Database Learning Roadmap

## For Experienced Engineers (13 Weeks)

This roadmap is designed for engineers with existing knowledge of PostgreSQL, MongoDB, Redis, and basic vector databases. The focus is production-oriented depth across all database families.

---

## Week-by-Week Schedule

### Phase 1: Relational Mastery (Weeks 1-3)

#### Week 1: Schema Design & Normalization

**Topics:**
- Normal forms: 1NF through BCNF
- When to denormalize for performance
- Primary keys, foreign keys, constraints
- Multi-tenant schema patterns (RLS)

**Activities:**
- [ ] Design normalized schema for e-commerce
- [ ] Identify performance bottlenecks in normalized design
- [ ] Implement denormalization where needed

**Resources:**
- PostgreSQL Documentation: Schema
- "Database Internals" - Chapter on storage

---

#### Week 2: Indexing & Query Optimization

**Topics:**
- B-tree index internals
- Index types: B-tree, GiST, GIN, BRIN, Hash
- Composite indexes and column order
- EXPLAIN ANALYZE deep dive
- Covering indexes and index-only scans

**Activities:**
- [ ] Run EXPLAIN ANALYZE on 5 complex queries
- [ ] Design indexing strategy for order processing
- [ ] Compare query plans before/after indexing

**Resources:**
- use-the-index-luke.com
- PostgreSQL EXPLAIN documentation

---

#### Week 3: Transactions & Concurrency

**Topics:**
- ACID properties
- Isolation levels: READ COMMITTED, REPEATABLE READ, SERIALIZABLE
- Locking: row-level, table-level, advisory locks
- Deadlock detection and prevention
- Savepoints and subtransactions

**Activities:**
- [ ] Implement concurrent order processing with proper locking
- [ ] Reproduce and fix deadlock scenarios
- [ ] Benchmark isolation level performance impact

**Resources:**
- PostgreSQL Locking Documentation
- "High Performance PostgreSQL"

---

### Phase 2: NoSQL Deep Dive (Weeks 4-6)

#### Week 4: Document Modeling (MongoDB)

**Topics:**
- Document model vs relational model
- Embedding vs referencing
- Aggregation pipeline
- Change streams for CDC
- Sharding strategies

**Activities:**
- [ ] Design document model for product catalog
- [ ] Build aggregation pipeline for analytics
- [ ] Implement change stream listener

**Resources:**
- MongoDB University M121 (Aggregation)
- MongoDB Documentation: Data Modeling

---

#### Week 5: Caching & Data Structures (Redis)

**Topics:**
- Data structures: strings, lists, sets, sorted sets, hashes, streams
- Persistence: RDB vs AOF
- Redis Cluster sharding
- Lua scripting
- Pub/Sub patterns

**Activities:**
- [ ] Implement distributed rate limiter
- [ ] Build real-time leaderboard with sorted sets
- [ ] Create pub/sub notification system

**Resources:**
- Redis Documentation: Data Types
- "Redis in Action" by Josiah Carlson

---

#### Week 6: Wide-Column & Graph Overview

**Topics:**
- Wide-column model (Cassandra)
- CQL and query patterns
- Graph databases (Neo4j)
- When to use each

**Activities:**
- [ ] Design Cassandra table for IoT data
- [ ] Build simple graph for social relationships
- [ ] Compare query patterns across models

**Resources:**
- Cassandra Documentation: Data Modeling
- Neo4j Documentation: Cypher

---

### Phase 3: OLAP & Time-Series (Weeks 7-9)

#### Week 7: Columnar Storage & OLAP

**Topics:**
- Columnar vs row-oriented storage
- Data warehouse concepts
- Partitioning and clustering
- Compression strategies
- Materialized views

**Activities:**
- [ ] Load sample data into ClickHouse
- [ ] Create materialized view for common query
- [ ] Benchmark vs PostgreSQL for analytics

**Resources:**
- ClickHouse Documentation
- "ClickHouse: The Definitive Guide"

---

#### Week 8: ETL/ELP Patterns

**Topics:**
- ETL vs ELT
- Data lake vs data warehouse
- CDC (Change Data Capture)
- Airbyte/Fivetran basics

**Activities:**
- [ ] Set up CDC from PostgreSQL to ClickHouse
- [ ] Build simple Airbyte pipeline
- [ ] Design incremental refresh strategy

**Resources:**
- Airbyte Documentation
- Fivetran Documentation

---

#### Week 9: Time-Series Databases

**Topics:**
- Time-series data modeling
- Retention policies
- Downsampling/rollup
- Compression
- TimescaleDB vs InfluxDB

**Activities:**
- [ ] Design hypertable for metrics
- [ ] Set up continuous aggregates
- [ ] Configure retention policy
- [ ] Build Grafana dashboard

**Resources:**
- TimescaleDB Documentation
- InfluxDB Documentation

---

### Phase 4: NewSQL & HTAP (Weeks 10-11)

#### Week 10: Distributed SQL

**Topics:**
- Distributed SQL architecture
- Raft consensus
- Sharding vs replication
- TiDB vs CockroachDB

**Activities:**
- [ ] Deploy TiDB cluster (managed)
- [ ] Run distributed transactions
- [ ] Test failover scenarios

**Resources:**
- TiDB Documentation
- CockroachDB Architecture Docs

---

#### Week 11: HTAP Concepts

**Topics:**
- HTAP definition and trade-offs
- Row store vs column store
- Resource allocation
- Workload isolation

**Activities:**
- [ ] Run mixed OLTP/OLAP workload
- [ ] Observe resource contention
- [ ] Optimize for both workloads

**Resources:**
- SingleStore Documentation
- HTAP Research Papers

---

### Phase 5: Vector Databases (Weeks 12-13)

#### Week 12: Vector Search Fundamentals

**Topics:**
- Embeddings overview
- ANN algorithms: HNSW, IVF, PQ
- Vector search with metadata filtering
- Similarity metrics

**Activities:**
- [ ] Generate embeddings for documents
- [ ] Build HNSW index
- [ ] Implement metadata filtering

**Resources:**
- Qdrant Documentation
- Pinecone Documentation

---

#### Week 13: RAG & Production

**Topics:**
- RAG architecture patterns
- Hybrid search
- Chunking strategies
- Evaluation metrics

**Activities:**
- [ ] Build complete RAG pipeline
- [ ] Implement hybrid search
- [ ] Evaluate retrieval quality

**Resources:**
- LangChain Vector Stores
- LlamaIndex Vector Indices

---

## Completion Checklist

By the end of this roadmap, you should be able to:

- [ ] Select appropriate database for any given scenario
- [ ] Design schemas optimized for specific access patterns
- [ ] Optimize queries using EXPLAIN ANALYZE
- [ ] Implement caching strategies with Redis
- [ ] Build analytical pipelines with ClickHouse
- [ ] Design time-series data models
- [ ] Understand distributed SQL trade-offs
- [ ] Build production RAG systems with vector databases
- [ ] Design polyglot persistence architectures

---

## Suggested Progression

```
Week 1-3:   ████████████ (Relational)
Week 4-6:   ████████████ (NoSQL)  
Week 7-9:   ████████████ (OLAP/Time-Series)
Week 10-11: █████ (NewSQL/HTAP)
Week 12-13: █████ (Vector/RAG)
```

---

## Additional Resources

### Books
- "Designing Data-Intensive Applications" - Martin Kleppmann
- "The Art of PostgreSQL" - François Husson
- "Database Internals" - Alex Petrov

### Courses
- PostgreSQL Academy
- MongoDB University
- Redis University

### Practice Platforms
- HackerRank SQL
- LeetCode Database
- DB Fiddle

---

*This roadmap is part of AI-Mastery-2026. Adjust pacing based on your experience and available time.*
