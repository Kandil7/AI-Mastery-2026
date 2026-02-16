# Relational Database Internals: How PostgreSQL/MySQL Work Under the Hood

## Executive Summary

Understanding the internal architecture of relational databases is essential for AI/ML engineers who need to optimize data pipelines, feature stores, and model serving systems. This system design explores the fundamental components of PostgreSQL and MySQL, revealing how these databases achieve ACID compliance, handle concurrency, and optimize query execution—knowledge that directly impacts ML infrastructure performance and reliability.

## Technical Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                          PostgreSQL Architecture              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Client Connections (Backend Processes)                 │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │ │
│  │  │ Backend 1   │   │ Backend 2   │   │ Backend N   │    │ │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘    │ │
│  │         │                 │                 │           │ │
│  │  ┌──────▼─────────────────▼─────────────────▼────────┐ │ │
│  │  │                Shared Memory Area                │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │ │
│  │  │  │ Shared Buffers│ │ WAL Buffers │ │ Caches      │ │ │ │
│  │  │  │ (8MB pages)  │ │ (Write-ahead)│ │ (Catalog, etc.)│ │ │ │
│  │  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ │ │ │
│  │  │         │               │               │        │ │ │
│  │  │  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐ │ │ │
│  │  │  │ Background  │ │ Checkpointer│ │ Autovacuum  │ │ │ │
│  │  │  │ Writer      │ │ Process     │ │ Worker      │ │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ │ │ │
│  │  └───────────────────────────────────────────────────┘ │ │
│  │                                                       │ │
│  │  ┌─────────────────────────────────────────────────┐ │ │
│  │  │                Storage Engine                   │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ │ │
│  │  │  │ Heap Files  │ │ Index Files │ │ TOAST       │ │ │ │
│  │  │  │ (Main table)│ │ (B-tree, etc.)│ │ (Large objects)│ │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

## Implementation Details

### Core Components Breakdown

#### 1. Connection Management
- **PostgreSQL**: One process per connection (fork-based)
- **MySQL**: Thread-per-connection model (with connection pooling)

**ML Impact**: Connection pooling is critical for ML inference servers handling thousands of concurrent requests.

#### 2. Query Processing Pipeline
```
Client Query → Parser → Rewriter → Planner → Executor → Result
                     ↑              ↑
                Rule-based      Cost-based
                transformations  optimization
```

**Query Planning for ML Workloads**:
- **Cost Estimation**: Uses statistics (ANALYZE command) to estimate row counts
- **Join Order Optimization**: Critical for feature engineering queries
- **Index Selection**: Chooses optimal indexes based on selectivity

#### 3. Storage Engine Architecture

**Heap Files (Main Table Storage)**:
- **Page-based storage**: 8KB pages in PostgreSQL, 16KB in MySQL
- **Tuple structure**: Header + data + null bitmap
- **Visibility map**: Tracks which pages have only visible tuples

**Index Structures**:
- **B-tree**: Default for most indexes (balanced tree)
- **Hash**: For equality lookups (PostgreSQL 10+)
- **GiST/GIN**: For full-text search, geometric data, JSON

#### 4. Write-Ahead Logging (WAL)
- **Mechanism**: Log changes before applying to data files
- **Checkpointing**: Periodic synchronization of dirty buffers
- **Crash Recovery**: Replay WAL from last checkpoint

**ML-Specific Importance**: Ensures model training data integrity during failures.

#### 5. Concurrency Control

**PostgreSQL MVCC (Multi-Version Concurrency Control)**:
- **Snapshot isolation**: Each transaction sees a consistent snapshot
- **xmin/xmax**: Transaction IDs mark tuple visibility
- **Vacuum**: Removes dead tuples and updates visibility map

**MySQL InnoDB Locking**:
- **Record locks**: Row-level locking
- **Gap locks**: Prevent phantom reads
- **Next-key locks**: Combination of record + gap locks

### Performance Optimization Mechanisms

#### Buffer Pool Management
- **LRU Algorithm**: Least Recently Used page replacement
- **Checkpoints**: Flush dirty pages to disk periodically
- **Prefetching**: Read-ahead for sequential scans

#### Query Execution Strategies
- **Sequential Scan**: Full table scan (good for small tables)
- **Index Scan**: Use index to find rows
- **Index Only Scan**: Return data directly from index (covering indexes)
- **Bitmap Index Scan**: Combine multiple indexes

#### Statistics and Autovacuum
- **ANALYZE**: Collects column statistics for query planner
- **Autovacuum**: Automatic cleanup of dead tuples
- **Statistics Target**: Controls granularity of statistics collection

## Performance Metrics and Trade-offs

| Component | PostgreSQL | MySQL (InnoDB) | ML Impact |
|-----------|------------|----------------|-----------|
| Connection Overhead | High (process-based) | Lower (thread-based) | MySQL better for high-concurrency inference |
| Query Planning | Advanced cost-based | Good cost-based | PostgreSQL better for complex feature queries |
| MVCC Implementation | Snapshot isolation | Read views + locking | PostgreSQL has better concurrency for ML workloads |
| WAL Performance | Tunable sync levels | Group commit optimization | Both good for durability requirements |
| Index Types | Rich variety (B-tree, GiST, GIN) | B-tree, Hash, Full-text | PostgreSQL better for specialized ML indexing |

**Throughput Benchmarks**:
- **Simple SELECT**: PostgreSQL 15K ops/sec, MySQL 20K ops/sec
- **Complex JOIN**: PostgreSQL 8K ops/sec, MySQL 6K ops/sec
- **Bulk INSERT**: PostgreSQL 5K ops/sec, MySQL 10K ops/sec
- **Concurrent Queries**: PostgreSQL 12K ops/sec, MySQL 8K ops/sec

## Key Lessons for AI/ML Systems

1. **Connection Pooling is Essential**: ML inference servers should use connection pools (pgbouncer, mysql-connector pooling).

2. **Statistics Matter**: Run ANALYZE regularly on ML feature tables to keep query plans optimal.

3. **Index Strategy**: Use covering indexes for frequent ML query patterns.

4. **MVCC Understanding**: Dead tuples accumulate during bulk ML operations; monitor vacuum activity.

5. **WAL Configuration**: Tune wal_level and synchronous_commit for ML workload requirements.

## Real-World Industry Examples

**Stripe**: Uses PostgreSQL with extensive indexing for payment processing and ML fraud detection

**Uber**: PostgreSQL for trip data and ML feature stores, leveraging MVCC for concurrent access

**Airbnb**: MySQL for booking system, PostgreSQL for analytics and ML workloads

**Netflix**: PostgreSQL for recommendation metadata, optimized with custom extensions

**Tesla**: PostgreSQL for vehicle telemetry analysis and ML model training data

## Measurable Outcomes

- **Query Optimization**: Proper indexing reduces ML feature query time from 500ms to 50ms
- **Connection Efficiency**: Connection pooling increases throughput by 3-5x for inference servers
- **Vacuum Management**: Regular autovacuum reduces bloat by 40-60% in ML data tables
- **WAL Tuning**: Optimized WAL settings improve bulk insert performance by 2-3x

**ML Impact Metrics**:
- Feature engineering pipeline: 60% faster with optimized query plans
- Model training data preparation: 40% reduction in ETL time
- Real-time inference: 99.9% availability with proper connection management

## Practical Guidance for AI/ML Engineers

1. **Monitor Key Metrics**: Track `pg_stat_activity`, `pg_stat_user_tables`, and WAL statistics.

2. **Use EXPLAIN ANALYZE**: Profile ML query performance regularly.

3. **Optimize for Your Workload**: 
   - Batch processing → PostgreSQL with MVCC
   - High-concurrency serving → MySQL with connection pooling

4. **Implement Proper Indexing**: Create composite indexes for common ML query patterns.

5. **Tune Memory Settings**: Adjust shared_buffers, work_mem based on ML workload size.

6. **Automate Maintenance**: Schedule regular ANALYZE and VACUUM for ML data tables.

7. **Consider Extensions**: PostgreSQL extensions like pg_partman for time-series ML data partitioning.

Understanding relational database internals empowers AI/ML engineers to design and optimize data infrastructure that delivers the performance, reliability, and scalability required for modern ML systems.