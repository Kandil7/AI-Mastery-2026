# Database Systems Learning Path

<div align="center">

![Database Types](https://img.shields.io/badge/Database_Types-10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![Prerequisite](https://img.shields.io/badge/Prerequisite-PostgreSQL_MongoDB_Redis-green.svg)

**Production-oriented database mastery curriculum for experienced engineers**

</div>

---

## Overview

This learning path builds on existing knowledge of PostgreSQL, MongoDB, Redis, and vector databases to create comprehensive understanding of all major database systems. Designed for experienced Full-Stack and AI engineers who want to make informed database architecture decisions.

---

## Curriculum Structure

### Core Documentation

| Section | Description | Priority |
|---------|-------------|----------|
| [Database Systems Mastery](database_systems_mastery.md) | Comprehensive guide covering all database families | Required |
| [Learning Roadmap](database_learning_roadmap.md) | Week-by-week study plan | Required |

### Quick Reference

| Cheat Sheet | Description |
|-------------|-------------|
| [Database Selection Guide](cheatsheets/database_selection.md) | When to use which database type |
| [SQL vs NoSQL Decision Tree](cheatsheets/sql_nosql_decision.md) | Visual decision framework |
| [Indexing Strategies](cheatsheets/indexing_strategies.md) | When and how to index |
| [Query Optimization](cheatsheets/query_optimization.md) | Common optimization patterns |

---

## Phase-Based Learning

### Phase 1: Relational Database Mastery (Weeks 1-3)

**Goal:** Deep understanding of PostgreSQL for production transactional systems

**Topics:**
- Normalization through BCNF
- Indexing strategies (B-tree, GiST, GIN, BRIN)
- Transactions and isolation levels
- Query optimization with EXPLAIN ANALYZE
- Advanced SQL (window functions, CTEs, recursive queries)
- Replication and partitioning

**Mini-Projects:**
- [Order Management System with PostgreSQL](../projects/database_order_management/README.md)

**Resources:**
- PostgreSQL Documentation: Query Planning, Locking, Partitioning
- "The Art of PostgreSQL" by François Husson
- Use The Index, Luke (use-the-index-luke.com)

---

### Phase 2: NoSQL Deep Dive (Weeks 4-6)

**Goal:** Master document stores, key-value stores, and understand wide-column systems

**Topics:**
- MongoDB aggregation pipeline and data modeling
- Redis data structures and caching patterns
- Cassandra wide-column modeling (overview)
- Graph database basics with Neo4j

**Mini-Projects:**
- [Event Logging System with MongoDB](../projects/database_event_logging/README.md)
- [API Caching Layer with Redis](../projects/database_redis_cache/README.md)

**Resources:**
- MongoDB University: M001, M121, M310
- Redis Documentation: Data Types, Clustering

---

### Phase 3: OLAP and Time-Series (Weeks 7-9)

**Goal:** Build analytical pipelines and understand time-series data

**Topics:**
- Columnar storage fundamentals
- ETL/ELT patterns
- ClickHouse or BigQuery for analytics
- Time-series modeling with TimescaleDB or InfluxDB

**Mini-Projects:**
- [Analytics Dashboard with ClickHouse](../projects/database_analytics_dashboard/README.md)
- [Metrics Monitoring with TimescaleDB](../projects/database_timeseries/README.md)

**Resources:**
- ClickHouse Documentation
- TimescaleDB Documentation

---

### Phase 4: NewSQL and HTAP (Weeks 10-11)

**Goal:** Understand distributed SQL and hybrid transactional/analytical processing

**Topics:**
- Distributed SQL architecture (TiDB, CockroachDB)
- HTAP concepts and trade-offs
- Geo-distribution

**Mini-Projects:**
- [Distributed SQL Experiment with TiDB](../projects/database_distributed_sql/README.md)

---

### Phase 5: Vector Databases and RAG (Weeks 12-13)

**Goal:** Master vector search for AI applications

**Topics:**
- Embeddings fundamentals
- ANN indexes (HNSW, IVF, PQ)
- Hybrid search
- RAG architecture patterns

**Mini-Projects:**
- [End-to-End RAG with Qdrant/Pinecone](../projects/database_rag/README.md)

**Resources:**
- Qdrant Documentation
- Pinecone Documentation

---

## Polyglot Projects

### Project 9: Polyglot SaaS Backend

**Description:** Combine PostgreSQL + Redis + MongoDB + ClickHouse in one application

**Objectives:**
- Design data flow between databases
- Implement transactions across services
- Coordinate cache invalidation
- Build unified API

**Tech Stack:** PostgreSQL, Redis, MongoDB, ClickHouse

---

## Benchmarking Tools

### Database Comparison Benchmarks

Run comparative benchmarks across database types:

```bash
# Relational benchmarks
python -m benchmarks.database.postgres_benchmark

# Document store benchmarks  
python -m benchmarks.database.mongodb_benchmark

# Cache benchmarks
python -m benchmarks.database.redis_benchmark
```

---

## Assessment and Validation

### Self-Assessment Questions

After each phase, verify understanding with these question types:

1. **Selection:** For a given scenario, which database and why?
2. **Trade-offs:** What are the limitations of your choice?
3. **Alternatives:** When would a different approach be better?
4. **Implementation:** How would you optimize for this workload?

### Performance Benchmarks

- Compare query times across databases for equivalent workloads
- Test scaling behavior under load
- Measure failure recovery times
- Profile resource consumption

---

## Integration with AI-Mastery-2026

This database curriculum complements other AI-Mastery-2026 modules:

| Database Type | AI-Mastery Module | Integration Point |
|---------------|-------------------|------------------|
| Vector DB | LLM Engineering | RAG systems, semantic search |
| Time-Series | ML Algorithms | Time series forecasting |
| Graph DB | ML Algorithms | GNN recommenders |
| OLAP | Production Systems | Feature stores, analytics |

---

## Contributing

Contributions welcome! To add new content:

1. Follow the case study format in `/case_studies/`
2. Add mini-project specs to `/projects/`
3. Include code examples with explanations
4. Update this index with new resources

---

## Quick Links

- [Main Case Study](database_systems_mastery.md)
- [Learning Roadmap](database_learning_roadmap.md)
- [Database Selection Guide](cheatsheets/database_selection.md)
- [GitHub Repository](https://github.com/Kandil7/AI-Mastery-2026)

