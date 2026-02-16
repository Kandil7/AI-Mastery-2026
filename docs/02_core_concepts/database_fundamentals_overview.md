# Database Fundamentals Overview for AI/ML Engineers

This document provides a comprehensive overview of database fundamentals essential for AI/ML engineers who need to understand not just how to use databases, but how they work under the hood.

## What is a Database?

A database is a structured collection of data that allows for efficient storage, retrieval, and management of information. For AI/ML engineers, databases serve as the foundation for:
- Training data storage and versioning
- Model metadata and provenance tracking
- Feature stores and online feature serving
- Experiment tracking and model registry
- Real-time inference data pipelines

## Core Database Components

### 1. Storage Engine
The storage engine manages how data is physically stored on disk and in memory. Key aspects include:
- **Page-based storage**: Data is organized into fixed-size pages (typically 4KB-16KB)
- **Buffer pool**: In-memory cache of frequently accessed pages
- **Write-ahead logging (WAL)**: Ensures durability by logging changes before applying them

### 2. Query Processor
Translates high-level queries into executable plans:
- **Parser**: Converts SQL into abstract syntax tree
- **Optimizer**: Generates efficient execution plans using cost models
- **Executor**: Runs the query plan against the storage engine

### 3. Transaction Manager
Ensures ACID properties (Atomicity, Consistency, Isolation, Durability):
- **Two-phase locking (2PL)**: Manages concurrent access
- **Multi-version concurrency control (MVCC)**: Allows read consistency without blocking writes
- **Isolation levels**: From READ UNCOMMITTED to SERIALIZABLE

## Data Models and Paradigms

### Relational Model
- Tables with rows and columns
- Primary keys, foreign keys, constraints
- SQL as the standard query language
- Strong consistency guarantees

### NoSQL Models
- **Document**: JSON-like documents (MongoDB)
- **Key-Value**: Simple key-value pairs (Redis, DynamoDB)
- **Wide-column**: Column-family storage (Cassandra, ScyllaDB)
- **Graph**: Nodes, edges, and properties (Neo4j, Amazon Neptune)

### Specialized Models
- **Time-series**: Optimized for timestamped data (TimescaleDB, InfluxDB)
- **Vector**: Optimized for similarity search (Qdrant, Milvus, Pinecone)
- **Event-sourcing**: Append-only event logs (Apache Kafka, EventStore)

## Database Architecture Patterns

### Single-node vs Distributed
- **Single-node**: Simpler, ACID-compliant, limited scalability
- **Distributed**: Horizontal scaling, eventual consistency, partition tolerance

### Shared-nothing Architecture
- Each node has independent CPU, memory, and storage
- Communication via network messages
- Enables horizontal scaling (e.g., Cassandra, CockroachDB)

### Log-structured Merge-tree (LSM-tree)
- Write-optimized storage engine
- Data written to memtable → SSTables → compaction
- Used in: LevelDB, RocksDB, Cassandra, ScyllaDB

### B-tree and Variants
- Read-optimized storage engine
- Balanced tree structure for efficient lookups
- Used in: PostgreSQL, MySQL, SQLite

## Performance Considerations for ML Workloads

### Throughput vs Latency Trade-offs
- **Batch processing**: High throughput, acceptable latency (training data loading)
- **Real-time inference**: Low latency, moderate throughput (online serving)
- **Interactive analysis**: Balanced requirements (exploratory data analysis)

### Data Locality and Caching
- **Columnar storage**: Better compression and scan performance for analytical queries
- **In-memory databases**: Ultra-low latency for real-time features
- **Caching layers**: Redis/Memcached for hot data

### Indexing Strategies
- **B-tree indexes**: Excellent for equality and range queries
- **Hash indexes**: Optimal for exact match lookups
- **Bitmap indexes**: Efficient for low-cardinality columns
- **Vector indexes**: HNSW, IVF, ANNOY for approximate nearest neighbor search

## Common Pitfalls for AI/ML Engineers

1. **Ignoring data skew**: Uneven distribution causing hot partitions
2. **Over-indexing**: Slowing down writes and increasing storage
3. **N+1 query problems**: Multiple round trips for related data
4. **Transaction misuse**: Long-running transactions blocking others
5. **Ignoring WAL configuration**: Affecting durability and recovery time
6. **Assuming ACID everywhere**: Not understanding eventual consistency trade-offs

## Best Practices for ML Systems

1. **Separate concerns**: Use different databases for different workloads
   - OLTP for operational data
   - OLAP for analytics
   - Vector DB for RAG systems
   - Time-series DB for monitoring/metrics

2. **Design for observability**: Include metrics, tracing, and logging
3. **Version your data schemas**: Like code versioning for reproducibility
4. **Consider data gravity**: Move computation to data, not vice versa
5. **Plan for scale**: Design with sharding and partitioning from day one

## Visual Diagrams

### Database Architecture Layers
```
┌─────────────────────────────────────────────────────┐
│                  Application Layer                    │
│  (AI/ML Models, APIs, Feature Engineering Pipelines) │
└─────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│                Query Processing Layer                 │
│  (Parser → Optimizer → Executor → Result Formatter) │
└─────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│               Transaction Management Layer            │
│  (ACID Guarantees, Concurrency Control, Recovery)   │
└─────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│                Storage Engine Layer                   │
│  (Buffer Pool, WAL, Page Management, Index Structures)│
└─────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────┐
│                 Physical Storage Layer                │
│        (Disk, SSD, Memory, Network Attached Storage) │
└─────────────────────────────────────────────────────┘
```

### Data Flow in Modern ML Systems
```
Training Data → [Feature Store] → Model Training → [Model Registry]
       ↑               ↓                     ↓
[Data Lake] ← [Online Features] ← [Real-time Inference]
       ↓               ↑                     ↑
[Monitoring] → [Drift Detection] → [Retraining Triggers]
```

## Further Reading
- [Database Internals](https://www.amazon.com/Designing-Data-Intensive-Applications-Reliable-Maintainable/dp/1449373321) by Martin Kleppmann
- [The Art of Computer Programming, Vol 3](https://www-cs-faculty.stanford.edu/~knuth/taocp.html) (Sorting and Searching)
- [ACID vs BASE](https://en.wikipedia.org/wiki/ACID) comparison
- [CAP Theorem](https://en.wikipedia.org/wiki/CAP_theorem) implications

This overview provides the foundation for understanding more specialized database topics covered in subsequent documents.