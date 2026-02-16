### 3.1 Data Model & Operational Trade-offs (Enhanced)

| Dimension | Relational | Document | Key-Value | Wide-Column | Graph |
|----------|------------|----------|-----------|-------------|-------|
| **Data Model** | Tables, rows, FKs | JSON docs, collections | `(k → v)` pairs | Partition key + clustering cols | Nodes, edges, properties |
| **Query Model** | SQL (joins, aggregations) | Query language + aggregation pipeline | GET/SET/DEL + Lua | CQL (SELECT WHERE pk = ? AND ck > ?) | Cypher/Gremlin (traversals) |
| **Scalability** | Vertical (or sharded manually) | Horizontal (sharding on `_id`) | Horizontal (consistent hashing) | Horizontal (partition-aware) | Hard (mostly vertical) |
| **Consistency** | Strong (ACID) | Tunable (session/strong) | Strong (single key), eventual (multi-key) | Tunable (QUORUM, ONE, ALL) | Strong (within cluster) |
| **Best For** | Transactions, reporting, integrity | Hierarchical data, rapid iteration | Caching, sessions, counters | Time-series, high-write logs | Relationships, paths, networks |
| **Real-World Examples** | Netflix (accounts), Capital One (core banking), Stripe (ledger) | Spotify (user profiles), Adobe (settings), Uber (driver state) | Twitter (timelines), Discord (presence), Netflix (EVCache) | Netflix (event logging), Apple (iMessage metadata), Tesla (telemetry) | PayPal (fraud rings), LinkedIn (network), GitHub (dependency graphs) |
| **Performance Metrics** | 10K-100K TPS (OLTP), <10ms P99 | 50K-200K TPS, <20ms P99 | 100K-1M+ TPS, <1ms P99 | 250K+ writes/sec, <15ms P99 | 10K-50K traversals/sec, O(1) per hop |
| **Pros** | Mature, standards, tooling | Flexibility, developer velocity | Speed, simplicity, atomic ops | Write scalability, time-range scans | Intuitive for networks |
| **Cons** | JOIN cost, rigid schema | No joins, index bloat, eventual consistency | No queries, no relations | No JOINs, complex modeling | Poor for aggregates, scaling hard |

### 3.2 OLTP vs OLAP vs HTAP (Enhanced)

| Aspect | OLTP | OLAP | HTAP |
|--------|------|------|------|
| **Goal** | Fast transactions, low latency | Complex analytics, high throughput scans | Real-time analytics on live data |
| **Workload** | Short, frequent reads/writes (e.g., `INSERT`, `UPDATE`, `SELECT` by PK) | Long-running scans, aggregations (`GROUP BY`, `JOIN`, window funcs) | Mix: point queries + aggregations on same dataset |
| **Data Layout** | Row-oriented | Column-oriented | Hybrid (e.g., delta store + columnar read store) |
| **Consistency** | Strong (ACID) | Eventual or snapshot isolation | Strong (linearizable reads) or near-real-time |
| **Tech Examples** | PostgreSQL, MySQL, CockroachDB | ClickHouse, Snowflake, Redshift, DuckDB | TiDB, CockroachDB (with `EXPERIMENTAL` OLAP), SingleStore, Google Spanner |
| **Real-World Performance** | Netflix MySQL: 99.99% availability, sub-100ms | Cloudflare ClickHouse: 1M+ events/sec, real-time dashboards | Shopee TiDB: 100K+ TPS during flash sales, real-time inventory |
| **Latency Profile** | P99: 10-100ms | P99: 100ms-2s (complex queries) | P99: 50-200ms (hybrid workloads) |
| **Scaling Strategy** | Vertical + sharding | Horizontal (columnar compression) | Horizontal + distributed transactions |

### 3.3 Vector DBs: Dedicated vs Extensions (Enhanced)

| Criterion | Dedicated (Qdrant, Milvus, Pinecone) | Extensions (pgvector, RedisSearch, Elasticsearch KNN) |
|----------|----------------------------------------|--------------------------------------------------------|
| **Performance** | Optimized ANN (HNSW/IVF-PQ), sub-10ms @ 100M vectors | Slower (e.g., pgvector: 100–500ms @ 10M), CPU-bound without GPU |
| **Scalability** | Native sharding, auto-balancing, cloud-managed | Limited by underlying DB (PostgreSQL max ~100M vectors without tuning) |
| **Ops Complexity** | Medium–High: manage clusters, index tuning | Low: leverage existing DB ops (backup, HA, monitoring) |
| **Integration** | Separate service; requires orchestration | Tight: same connection, transactions, ACLs |
| **Hybrid Search** | First-class (metadata filters + vector) | Possible but often clunky (e.g., `WHERE metadata = ? AND <->` in PG) |
| **Cost** | Higher (dedicated infra) | Lower (reuse existing DB) |
| **When to Choose** | >10M vectors, strict latency SLAs, production RAG | <5M vectors, prototyping, tight coupling with app data, budget constraints |
| **Real-World Benchmarks** | Qdrant: 0.94 recall@10, 18ms P99 (1M vectors) | pgvector: 0.87 recall@10, 85ms P99 (1M vectors) |
| **Production Use Cases** | Amazon Bedrock RAG, Notion AI search, GitHub Copilot | Internal tools, MVP prototypes, small-scale applications |

> ✅ **Rule of Thumb**: Start with `pgvector` for MVP; migrate to Qdrant/Milvus when latency >50ms or scale >10M vectors. For production RAG with strict SLAs, Qdrant provides best balance of recall, latency, and operational maturity.