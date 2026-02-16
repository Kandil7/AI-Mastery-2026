# Database Storage Engines Comparison

Choosing the right storage engine is a critical architectural decision that impacts performance, concurrency, features, and operational characteristics. This document provides a comprehensive comparison of major database storage engines, their internal architectures, and practical guidance for selection.

## Table of Contents

1. [Storage Engine Overview](#1-storage-engine-overview)
2. [PostgreSQL Storage Engines](#2-postgresql-storage-engines)
3. [MySQL Storage Engines](#3-mysql-storage-engines)
4. [MongoDB WiredTiger Storage Engine](#4-mongodb-wiredtiger-storage-engine)
5. [Specialized Storage Engines](#5-specialized-storage-engines)
6. [Choosing the Right Storage Engine](#6-choosing-the-right-storage-engine)
7. [Performance Comparison](#7-performance-comparison)
8. [Migration Considerations](#8-migration-considerations)

---

## 1. Storage Engine Overview

### 1.1 What is a Storage Engine?

A storage engine is the underlying software component that manages how data is stored, retrieved, and managed at the physical level. Different storage engines are optimized for different use cases, from transactional processing to analytical workloads.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Database Storage Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    SQL Interface Layer                     │  │
│  │              (Query Parser, Optimizer, Executor)          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Storage Engine Interface                     │  │
│  │         (Common API for multiple engines)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│          ┌─────────────────┼─────────────────┐                   │
│          ▼                 ▼                 ▼                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │  Engine A   │   │  Engine B   │   │  Engine C   │           │
│  │  (InnoDB)   │   │  (MyISAM)   │   │  (Memory)   │           │
│  │             │   │             │   │             │           │
│  │ - MVCC      │   │ - Table lock│   │ - In-memory │           │
│  │ - Row lock  │   │ - Full-text │   │ - Hash idx  │           │
│  │ - ACID      │   │ - No ACID   │   │ - No ACID   │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Disk Subsystem                          │  │
│  │              (File System, Block Device)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Comparison Dimensions

| Dimension | Description | Key Questions |
|-----------|-------------|---------------|
| **Concurrency** | How engine handles simultaneous operations | Row-level vs table-level locking, MVCC support |
| **Durability** | Data persistence guarantees | Write-ahead logging, crash recovery |
| **Performance** | Read/write characteristics | Index types, caching, storage format |
| **Features** | Special capabilities | Full-text search, spatial data, JSON |
| **Storage** | Physical data organization | Row vs columnar, compression |
| **Scalability** | Growth characteristics | Vertical vs horizontal |

---

## 2. PostgreSQL Storage Engines

### 2.1 Heap Storage (Default)

PostgreSQL's default storage uses heap files with MVCC:

```
┌─────────────────────────────────────────────────────────────────┐
│                PostgreSQL Heap Storage                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Table Structure                        │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Page 0 (8KB)                                         │ │  │
│  │  │ ┌─────────────────────────────────────────────────┐ │ │  │
│  │  │ │ PageHeader: LSN, checksum, items, free space   │ │ │  │
│  │  │ ├─────────────────────────────────────────────────┤ │ │  │
│  │  │ │ ItemIdData: offset, length                      │ │ │  │
│  │  │ ├─────────────────────────────────────────────────┤ │ │  │
│  │  │ │ Tuple 1: t_xmin, t_xmax, t_cid, t_ctid, data  │ │ │  │
│  │  │ │ Tuple 2: t_xmin, t_xmax, t_cid, t_ctid, data  │ │ │  │
│  │  │ │ ...                                              │ │ │  │
│  │  │ └─────────────────────────────────────────────────┘ │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Page 1 (8KB)                                         │ │  │
│  │  │ └────────────────────────────────────────────────────┘ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Characteristics:                                                │
│  - MVCC with XID-based visibility                               │
│  - Heap-only tuples (HOT) for same-page updates                 │
│  - Free space map for new tuple insertion                       │
│  - Visibility map for vacuum optimization                       │
│                                                                  │
│  Index Structure:                                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 B-tree Index                               │  │
│  │                                                            │  │
│  │  Root → Internal pages → Leaf pages (with TIDs)         │  │
│  │                                                            │  │
│  │  Leaf entry: (key value, TID)                            │  │
│  │  TID = (block number, offset)                            │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 PostgreSQL Table Access Methods

PostgreSQL allows pluggable table access methods:

```sql
-- Check available table access methods
SELECT amname FROM pg_access_method;

-- Check current table's access method
SELECT relname, relkind, relam FROM pg_class
WHERE relname = 'orders';

-- Create table with specific access method
CREATE TABLE orders_heap () USING heap;
CREATE TABLE orders_zheap () USING zheap;  -- If available

-- zheap (experimental): More efficient MVCC storage
-- - Stores undo information within the table
-- - Reduces bloat
-- - Currently in development
```

### 2.3 PostgreSQL Index Types

| Index Type | Best For | Not Suitable For |
|------------|----------|------------------|
| **B-tree** | Equality, range queries, sorted output | Complex data types |
| **Hash** | Equality only | Range queries, durability issues |
| **GIN** | Composite types, arrays, JSONB | Simple key-value |
| **GiST** | Geometric, full-text, range types | General purpose |
| **BRIN** | Time-series, naturally ordered data | Random access patterns |
| **SP-GiST** | Large datasets with non-balanced trees | General purpose |

```sql
-- B-tree (default)
CREATE INDEX idx1 ON orders (customer_id);

-- Hash
CREATE INDEX idx2 ON orders USING HASH (customer_id);

-- GIN for arrays
CREATE INDEX idx3 ON orders (tags);

-- GiST for spatial
CREATE INDEX idx4 ON locations USING GIST (position);

-- BRIN for time-series
CREATE INDEX idx5 ON sensor_data USING BRIN (timestamp);

-- Composite index with included columns
CREATE INDEX idx6 ON orders (customer_id, status) INCLUDE (total);
```

---

## 3. MySQL Storage Engines

### 3.1 InnoDB (Default)

InnoDB is MySQL's ACID-compliant storage engine:

```
┌─────────────────────────────────────────────────────────────────┐
│                    InnoDB Storage Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Tablespace Structure                      │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ System Tablespace (ibdata1)                        │ │  │
│  │  │ ├─ Doublewrite Buffer                              │ │  │
│  │  │ ├─ Change Buffer                                   │ │  │
│  │  │ ├─ Undo Logs                                       │ │  │
│  │  │ └─ Data Dictionary                                 │ │  │
│  │  ├─────────────────────────────────────────────────────┤ │  │
│  │  │ File-per-table tablespaces (*.ibd)                 │ │  │
│  │  │ ├─ Index Root Page                                 │ │  │
│  │  │ ├─ Leaf Pages (B+tree)                            │ │  │
│  │  │ └─ Row Data                                        │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Row Format Options:                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ COMPACT (default):                                        │  │
│  │ ┌─────────┬──────────────┬────────────┬───────────────┐   │  │
│  │ │ NULL flg│ Column 1..N  │ External   │ Transaction ID│   │  │
│  │ │ (1 byte)│ (no NULL)    │ (optional) │ Rollback ptr  │   │  │
│  │ └─────────┴──────────────┴────────────┴───────────────┘   │  │
│  │                                                            │  │
│  │ DYNAMIC (MySQL 5.7+):                                      │  │
│  │ - All columns stored off-page if > 768 bytes              │  │
│  │ - 20-byte pointer in row                                  │  │
│  │                                                            │  │
│  │ COMPRESSED:                                                │  │
│  │ - Page compression (zlib)                                 │  │
│  │ - Smaller page size (16K default)                        │  │
│  │                                                            │  │
│  │ REDUNDANT (old format):                                   │  │
│  │ - More overhead per column                                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 MyISAM

MyISAM was MySQL's default engine before InnoDB:

```sql
-- Create MyISAM table
CREATE TABLE logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP
) ENGINE = MyISAM;

-- MyISAM characteristics:
-- - Table-level locking
-- - No transaction support
-- - Index statistics
-- - Full-text search (legacy)
-- - Fast reads, slow writes under contention

-- Status:
-- Pros: Fast full-text search, low memory footprint
-- Cons: No ACID, table locks, crashes can corrupt
-- Use: Read-heavy, non-critical applications
```

### 3.3 MySQL Memory Engine

```sql
-- Memory storage engine (heap tables)
CREATE TABLE sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id INT,
    data JSON,
    expires_at TIMESTAMP
) ENGINE = Memory;

-- Characteristics:
-- - All data in memory (lost on restart)
-- - Hash indexes (default) or B-tree
-- - Fixed-row format
-- - Very fast for temporary data

-- Best for:
-- - Session storage with Redis alternative unavailable
-- - Temporary lookup tables
-- - Caching layers
```

### 3.4 Other MySQL Engines

```sql
-- CSV Engine
CREATE TABLE archive (
    id INT,
    data VARCHAR(255)
) ENGINE = CSV;
-- Data stored in CSV files, good for data exchange

-- Archive Engine
CREATE TABLE logs_archive (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP
) ENGINE = Archive;
-- Compressed storage, insert-only
-- Good for logging, audit trails

-- Federated Engine (deprecated in 8.0)
-- Allows访问远程 MySQL tables

-- Blackhole Engine
CREATE TABLE blackhole_test (
    id INT
) ENGINE = Blackhole;
-- Writes discarded, good for replication filtering
```

### 3.5 MySQL Engine Comparison

| Feature | InnoDB | MyISAM | Memory | Archive |
|---------|--------|--------|--------|---------|
| **ACID** | Yes | No | No | No |
| **Row-level Lock** | Yes | No | N/A | No |
| **Table-level Lock** | No | Yes | No | Yes |
| **MVCC** | Yes | No | No | No |
| **Transactions** | Yes | No | No | No |
| **Crash Recovery** | Yes | Limited | None | None |
| **Full-text Index** | Yes (5.6+) | Yes | No | No |
| **Spatial Data** | Yes | Yes | No | No |
| **Compression** | Yes | Limited | No | Yes |
| **Foreign Keys** | Yes | No | No | No |

---

## 4. MongoDB WiredTiger Storage Engine

### 4.1 WiredTiger Architecture

WiredTiger is MongoDB's default storage engine:

```
┌─────────────────────────────────────────────────────────────────┐
│               WiredTiger Storage Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              WiredTiger Cache (In-Memory)                 │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │                 Document Store                       │ │  │
│  │  │                                                      │ │  │
│  │  │  ┌───────────────────────────────────────────────┐  │ │  │
│  │  │  │ WT_UPDATE:                                    │  │ │  │
│  │  │  │ ┌─────────────────────────────────────────┐   │  │ │  │
│  │  │  │ │ timestamp: Update timestamp             │   │  │ │  │
│  │  │  │ │ type: insert/update/prepare             │   │  │ │  │
│  │  │  │ │ size: Data size                          │   │  │ │  │
│  │  │  │ │ data: Actual document or diff          │   │  │ │  │
│  │  │  │ └─────────────────────────────────────────┘   │  │ │  │
│  │  │  └───────────────────────────────────────────────┘  │  │
│  │  └─────────────────────────────────────────────────────┘  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │                 Index Store                           │ │  │
│  │  │                                                      │ │  │
│  │  │  B-tree indexes for each index                      │ │  │
│  │  │                                                      │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  │  Default: 50% of RAM minus 1GB                          │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Journal (Write-Ahead Log)                    │  │
│  │                                                            │  │
│  │  - Every operation written to journal                    │  │
│  │  - Default: flush every 100ms                            │  │
│  │  - After checkpoint, journal can be cleaned               │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Data Files (*.wt)                     │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Collection: WiredTiger file                         │ │  │
│  │  │ - B-tree structure                                  │ │  │
│  │  │ - Compression enabled                               │ │  │
│  │  │ - Prefix compression                               │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Index: WiredTiger file                              │ │  │
│  │  │ - B-tree structure                                  │ │  │
│  │  │ - Separate from data                               │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 WiredTiger Concurrency

```javascript
// WiredTiger supports document-level locking
// Multiple readers can access same collection
// Writers only block other writers on same document

// Checkpoint behavior
// - Default: every 60 seconds or every 2GB of journal
// - Creates consistent snapshot
// - All data up to checkpoint is recoverable

// Configuration
db.adminCommand({
    setParameter: 1,
    wiredTigerConcurrentWriteTransactions: 128
});
db.adminCommand({
    setParameter: 1,
    wiredTigerConcurrentReadTransactions: 128
});
```

### 4.3 WiredTiger Compression

```javascript
// WiredTiger compression configuration
// mongod.conf

storage:
  dbPath: /var/lib/mongodb
  wiredTiger:
    engineConfig:
      cacheSizeGB: 4          # Cache size
      journalCompressor: snappy  # Journal compression
      directoryForIndexes: false # Separate index directory
    collectionConfig:
      blockCompressor: snappy    # Data compression
      prefixCompression: true    # Index prefix compression
    indexConfig:
      prefixCompression: true    # Index compression

// Compression algorithms
// - snappy: Default, balanced speed/ratio
// - zlib: Higher compression, slower
// - zstd: (MongoDB 4.2+) Balanced, good ratio
// - none: No compression
```

---

## 5. Specialized Storage Engines

### 5.1 Columnar Storage Engines

```
┌─────────────────────────────────────────────────────────────────┐
│               Columnar vs Row-Oriented Storage                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Row-Oriented (OLTP):                                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Row 1: id=1, name=John, age=30, city=NYC                  │  │
│  │ Row 2: id=2, name=Jane, age=25, city=LA                   │  │
│  │ Row 3: id=3, name=Bob, age=35, city=NYC                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  └───────────────────────────────────────────────────────────┐  │
│  │ All columns for a row stored together                     │  │
│  │ Best for: Transactional workloads, point queries          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Columnar (OLAP):                                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ id: 1, 2, 3, 4, 5, ...                                    │  │
│  │ name: John, Jane, Bob, ...                                │  │
│  │ age: 30, 25, 35, 28, ...                                 │  │
│  │ city: NYC, LA, NYC, SF, ...                              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  └───────────────────────────────────────────────────────────┐  │
│  │ All values for a column stored together                    │  │
│  │ Best for: Analytics, aggregation, column scans            │  │
│  │ - Better compression (similar values)                     │  │
│  │ - Only read needed columns                                 │  │
│  │ - Vectorized processing                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Engines:                                                        │
│  - ClickHouse: Native columnar                                   │
│  - Apache Parquet: Columnar file format                          │
│  - Amazon Redshift: Columnar storage                            │
│  - PostgreSQL cstore_fdw: Columnar extension                    │
│  - MySQL HeatWave: In-memory columnar                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 LSM-Tree Based Engines

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSM-Tree Storage                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              LSM-Tree Structure                             │  │
│  │                                                            │  │
│  │              ┌───────────────┐                            │  │
│  │              │  MemTable      │                            │  │
│  │              │  (In-memory)  │                            │  │
│  │              └───────┬───────┘                            │  │
│  │                      │ Flush                               │  │
│  │                      ▼                                    │  │
│  │              ┌───────────────┐                            │  │
│  │              │   SSTable 0   │                            │  │
│  │              │   (Level 0)   │                            │  │
│  │              └───────┬───────┘                            │  │
│  │                      │ Compaction                         │  │
│  │                      ▼                                    │  │
│  │              ┌───────────────┐                            │  │
│  │              │   SSTable 1   │                            │  │
│  │              │   (Level 1)   │                            │  │
│  │              └───────┬───────┘                            │  │
│  │                      │ Compaction                         │  │
│  │                      ▼                                    │  │
│  │              ┌───────────────┐                            │  │
│  │              │   SSTable 2   │                            │  │
│  │              │   (Level 2)   │                            │  │
│  │              └───────────────┘                            │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Write Path:                                                     │
│  1. Write to WAL (for durability)                               │
│  2. Write to MemTable (fast, in-memory)                        │
│  3. When MemTable is full:                                      │
│     - Flush to SSTable (sorted file on disk)                   │
│     - Background compaction merges levels                      │
│                                                                  │
│  Read Path:                                                     │
│  1. Check MemTable                                             │
│  2. Check SSTables (newest to oldest)                          │
│  3. Merge results                                              │
│                                                                  │
│  Advantages:                                                    │
│  - Fast writes (append-only)                                   │
│  - Good for write-heavy workloads                              │
│  - Automatic compaction                                         │
│                                                                  │
│  Disadvantages:                                                 │
│  - Read amplification (multiple files to check)               │
│  - Compaction overhead                                         │
│                                                                  │
│  Engines using LSM:                                             │
│  - RocksDB (Facebook, embedded)                                │
│  - Cassandra                                                   │
│  - ScyllaDB                                                    │
│  - MongoRocks (deprecated)                                      │
│  - Apache Druid                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 In-Memory Storage Engines

| Engine | Type | Use Case | Features |
|--------|------|----------|----------|
| **Redis** | Key-value | Caching, session | Strings, lists, sets, streams |
| **Memcached** | Key-value | Caching | Simple, no persistence |
| **SAP HANA** | In-memory | Analytics | Columnar, ACID |
| **MemSQL (SingleStore)** | Hybrid | HTAP | Disk + memory, SQL |
| **Apache Ignite** | Distributed | Caching, compute | ACID, SQL |

---

## 6. Choosing the Right Storage Engine

### 6.1 Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              Storage Engine Selection Framework                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 1: Identify Primary Workload                         │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │ Transactional (OLTP)     │    Analytical (OLAP)      │ │  │
│  │  │                           │                           │ │  │
│  │  │ - Point updates         │ - Complex aggregations    │ │  │
│  │  │ - ACID required          │ - Large scans             │ │  │
│  │  │ - Many concurrent users │ - Few users               │ │  │
│  │  │ - Low latency           │ - High throughput         │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 2: Evaluate Key Requirements                          │  │
│  │                                                            │  │
│  │  □ ACID compliance needed?                                 │  │
│  │    Yes → InnoDB, PostgreSQL, WiredTiger                   │  │
│  │    No  → MyISAM, Memory, some NoSQL                       │  │
│  │                                                            │  │
│  │  □ Row-level locking required?                            │  │
│  │    Yes → InnoDB, PostgreSQL                                │  │
│  │    No  → MyISAM                                           │  │
│  │                                                            │  │
│  │  □ JSON/document support?                                 │  │
│  │    Yes → InnoDB (MySQL 5.7+), MongoDB, PostgreSQL JSON    │  │
│  │                                                            │  │
│  │  □ Full-text search?                                      │  │
│  │    Yes → InnoDB, MyISAM, Elasticsearch, PostgreSQL        │  │
│  │                                                            │  │
│  │  □ Spatial data?                                          │  │
│  │    Yes → InnoDB, PostgreSQL (PostGIS), MongoDB           │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 3: Consider Operational Factors                      │  │
│  │                                                            │  │
│  │  □ Team expertise                                         │  │
│  │  □ Infrastructure (cloud, on-prem)                        │  │
│  │  □ Scalability requirements                               │  │
│  │  □ Maintenance capabilities                                │  │
│  │  □ Budget                                                 │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Quick Reference Matrix

| Requirement | Primary Choice | Alternatives |
|-------------|----------------|--------------|
| **General OLTP** | PostgreSQL, InnoDB | MySQL (all engines) |
| **Heavy Writes** | PostgreSQL, Cassandra, RocksDB | MySQL InnoDB |
| **Heavy Reads** | PostgreSQL, MySQL + Cache | MongoDB |
| **JSON Documents** | PostgreSQL JSON, MongoDB | MySQL JSON |
| **Full-Text Search** | Elasticsearch, PostgreSQL | InnoDB, MyISAM |
| **Time-Series** | TimescaleDB, InfluxDB | PostgreSQL + BRIN |
| **Spatial/GIS** | PostgreSQL + PostGIS | MySQL InnoDB |
| **Analytical** | ClickHouse, Redshift | PostgreSQL + cstore |
| **Caching** | Redis, Memcached | MySQL Memory |
| **HTAP** | SingleStore, TiDB | PostgreSQL + Citus |

### 6.3 Use Case Recommendations

#### E-commerce Application

```sql
-- PostgreSQL recommendation for e-commerce
-- Primary: Transactional workload

-- Orders table (ACID critical)
CREATE TABLE orders (
    order_id BIGSERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    status VARCHAR(20) NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
) WITH (fillfactor = 80);  -- Leave room for updates

-- Products (frequent reads, occasional writes)
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    inventory INT NOT NULL DEFAULT 0
);

-- Optimized indexes
CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_status_created ON orders (status, created_at);
CREATE INDEX idx_products_category ON products (category_id);

-- Full-text search for product descriptions
ALTER TABLE products ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', name || ' ' || coalesce(description, ''))) STORED;
CREATE INDEX idx_products_search ON products USING GIN (search_vector);
```

#### Logging/Analytics Platform

```sql
-- ClickHouse or TimescaleDB for time-series analytics
-- Example: ClickHouse

CREATE TABLE events (
    event_time DateTime,
    event_type String,
    user_id UInt32,
    properties Map(String, String),
    value Float64
) ENGINE = MergeTree()
ORDER BY (event_type, event_time, user_id)
PARTITION BY toYYYYMM(event_time)
TTL event_time + INTERVAL 3 MONTH;

-- Aggregation for fast queries
CREATE MATERIALIZED VIEW events_by_type
ENGINE = SummingMergeTree()
ORDER BY (event_type, toStartOfHour(event_time))
AS SELECT
    event_type,
    toStartOfHour(event_time) AS hour,
    count() AS cnt,
    sum(value) AS total
FROM events
GROUP BY event_type, hour;
```

#### Session Store

```python
# Redis for session storage
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# Session storage
session_data = {
    'user_id': 12345,
    'name': 'John Doe',
    'roles': ['admin', 'user'],
    'login_time': '2025-01-15T10:30:00Z'
}

# Store session with expiration
r.setex(f'session:{session_id}', 3600, json.dumps(session_data))

# Retrieve session
session = r.get(f'session:{session_id}')
if session:
    data = json.loads(session)
```

---

## 7. Performance Comparison

### 7.1 Benchmark Comparison

| Operation | PostgreSQL | MySQL InnoDB | MongoDB | Redis |
|-----------|-----------|--------------|---------|-------|
| **Simple INSERT** | Fast | Fast | Fast | Very Fast |
| **Bulk INSERT** | Good | Good | Good | N/A |
| **Point SELECT** | Good | Good | Good | Excellent |
| **Range Query** | Excellent | Good | Good | N/A |
| **Aggregation** | Good | Good | Good | Good |
| **JOIN** | Excellent | Good | Limited | N/A |
| **Full-Text Search** | Good | Good | Good | N/A |
| **JSON Operations** | Excellent | Good | Excellent | N/A |
| **Point UPDATE** | Good | Good | Good | Excellent |

### 7.2 Concurrency Characteristics

```
┌─────────────────────────────────────────────────────────────────┐
│              Concurrency Performance Comparison                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Write-Heavy Workload:                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            │ 10 threads │ 50 threads │ 100 threads       │  │
│  │────────────┼────────────┼────────────┼───────────────────┤  │
│  │ PostgreSQL │  10,000    │  35,000    │  50,000          │  │
│  │ MySQL InnoDB│  12,000    │  40,000    │  55,000          │  │
│  │ MongoDB     │  15,000    │  45,000    │  60,000          │  │
│  │ RocksDB     │  50,000    │ 100,000    │ 150,000          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Read-Heavy Workload:                                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            │ 10 threads │ 50 threads │ 100 threads       │  │
│  │────────────┼────────────┼────────────┼───────────────────┤  │
│  │ PostgreSQL │  50,000    │ 180,000    │ 250,000          │  │
│  │ MySQL InnoDB│  60,000    │ 200,000    │ 300,000          │  │
│  │ MongoDB     │  55,000    │ 190,000    │ 280,000          │  │
│  │ Redis       │ 200,000    │ 800,000    │ 1,200,000       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Mixed Workload (70% read, 30% write):                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            │ 10 threads │ 50 threads │ 100 threads       │  │
│  │────────────┼────────────┼────────────┼───────────────────┤  │
│  │ PostgreSQL │  30,000    │ 100,000    │ 150,000          │  │
│  │ MySQL InnoDB│  35,000    │ 110,000    │ 160,000          │  │
│  │ MongoDB     │  40,000    │ 120,000    │ 170,000          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  * Values are approximate ops/sec, vary by hardware             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Memory Requirements

| Engine | Cache Size | Memory per Connection | Notes |
|--------|------------|----------------------|-------|
| PostgreSQL | 25% RAM | ~5-10MB | Shared buffer pool |
| MySQL InnoDB | 50-80% RAM | ~1-2MB | Buffer pool + connections |
| MongoDB WiredTiger | 50% RAM -1GB | ~1MB | Cache + connections |
| Redis | N/A | Variable | All data in memory |

---

## 8. Migration Considerations

### 8.1 Migrating Between Engines

#### MySQL: MyISAM to InnoDB

```sql
-- Convert MyISAM to InnoDB
ALTER TABLE mytable ENGINE = InnoDB;

-- Best practices:
-- 1. Ensure primary key exists
-- 2. Analyze table for optimal row format
ALTER TABLE mytable ENGINE = InnoDB
    ROW_FORMAT = DYNAMIC
    KEY_BLOCK_SIZE = 8;

-- For large tables, use online DDL
ALTER TABLE mytable ENGINE = InnoDB,
    ALGORITHM = COPY,
    LOCK = NONE;
```

#### PostgreSQL to MySQL

```sql
-- Data type mapping considerations
-- PostgreSQL → MySQL
-- SERIAL → AUTO_INCREMENT
-- BYTEA → BLOB
-- JSON → JSON
-- TEXT → LONGTEXT
-- TIMESTAMP → DATETIME
-- UUID → CHAR(36) or VARCHAR(64)

-- Migration approach:
-- 1. Export using pg_dump
-- 2. Transform data types
-- 3. Import to MySQL
-- 4. Verify data integrity

-- Example using mysqldump
mysqldump --compatible=postgresql source_db > dump.sql
```

### 8.2 MongoDB to PostgreSQL Migration

```javascript
// MongoDB aggregation to find schemas
db.collection.aggregate([
    { $sample: { size: 1000 } },
    { $project: { arrayOfKeys: { $objectToArray: "$$ROOT" } } },
    { $unwind: "$arrayOfKeys" },
    { $group: { _id: "$arrayOfKeys.k", types: { $addToSet: { $type: "$arrayOfKeys.v" } } } }
]);

// Migration strategy:
// 1. Analyze MongoDB documents
// 2. Design PostgreSQL schema
// 3. Use tools like pgloader or mongosql
// 4. Test with subset of data
```

---

## Related Documentation

- [PostgreSQL Internals](postgresql_internals.md)
- [MySQL InnoDB Internals](mysql_innodb_internals.md)
- [Time-Series Databases](01_time_series_databases.md)
- [Vector Databases](01_vector_databases.md)

---

## References

- PostgreSQL Documentation: https://www.postgresql.org/docs/
- MySQL Documentation: https://dev.mysql.com/doc/refman/8.0/en/
- MongoDB Documentation: https://docs.mongodb.com/
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "High Performance MySQL" by Baron Schwartz et al.
