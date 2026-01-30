# ADR-001: Postgres FTS vs Elasticsearch for Keyword Search

## 📝 Context
To implement **Hybrid Search**, we need a strong Keyword Search (Lexical) component. The industry standard is typically Elasticsearch (or OpenSearch), but our stack already relies on Postgres for relational data.

## 🏗️ Decision
We decided to use **Postgres Full-Text Search (tsvector)** instead of a dedicated Elasticsearch cluster.

## 💡 Rationale

### 1. Operational Complexity
*   **Elasticsearch**: Requires JVM, large heavy containers, and separate maintenance pipelines. For a "Mini" engine or mid-sized startup, this is massive overhead.
*   **Postgres**: We already run Postgres for metadata. Adding a `tsvector` column is effectively "free" operationally.

### 2. Synchronization (The "Dual Write" Problem)
*   Keeping metadata (Postgres) in sync with Search indices (Elastic) is a notorious source of bugs (data inconsistency).
*   By using Postgres for *both*, updates are **Atomic**. Transactional integrity is guaranteed out of the box.

### 3. Performance
*   For datasets under ~10 Million chunks, Postgres GIN indexes perform exceptionally well (millisec latency).
*   Elasticsearch only starts justifying its cost at massive scale (>50M+ docs) or for complex fuzzy/aggregation queries we don't need yet.

## ⚠️ Consequences
*   **Pro**: Single "Source of Truth" database.
*   **Pro**: Simplified Docker Compose and deployment.
*   **Con**: Less advanced linguistic features (custom analyzers, specific language stemming) compared to Elastic.
*   **Con**: Scaling writes might be harder eventually, but solving for scale <10M now is premature optimization.

## ✅ Status
Accepted & Implemented in `src/adapters/postgres/keyword_store.py`.
