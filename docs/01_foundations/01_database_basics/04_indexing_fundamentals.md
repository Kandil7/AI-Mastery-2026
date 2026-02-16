# Indexing Fundamentals

Indexes are data structures that speed up data retrieval operations. Understanding different index types and their characteristics is essential for optimizing database performance in AI/ML applications.

## Overview

Indexing is one of the most important techniques for improving database performance. Proper indexing can reduce query execution time from seconds to milliseconds, making it critical for production AI systems that require low-latency data access.

## What Are Indexes?

An index is a data structure that allows the database to quickly locate rows without scanning the entire table. Think of it as a book's index that helps you find information without reading every page.

### Key Concepts
- **Key**: The column(s) being indexed
- **Value**: Pointer to the actual data location
- **Selectivity**: Ratio of unique values to total rows (higher = better)
- **Cardinality**: Number of unique values in the indexed column

## Common Index Types

### B-Tree Indexes

B-Tree (Balanced Tree) indexes are the most common index type, supporting efficient range queries, equality comparisons, and sorted results.

#### Characteristics
- **Structure**: Balanced tree with logarithmic search time
- **Operations**: Efficient for equality, range, and sorting
- **Storage**: Moderate overhead (typically 20-30% of table size)
- **Updates**: O(log n) for inserts/deletes

#### Best For
- Most use cases, especially when querying ranges or sorting
- Primary keys and foreign keys
- High-selectivity columns

#### Example
```sql
-- Simple B-Tree index
CREATE INDEX idx_users_email ON users(email);

-- Composite index (covering multiple columns)
CREATE INDEX idx_orders_customer_date
ON orders(customer_id, order_date DESC);

-- Partial index (only for active records)
CREATE INDEX idx_active_orders
ON orders(customer_id)
WHERE status = 'active';
```

### Hash Indexes

Hash indexes provide O(1) lookup time for equality comparisons but cannot handle range queries.

#### Characteristics
- **Structure**: Hash table mapping keys to row locations
- **Operations**: Only equality comparisons (no ranges)
- **Storage**: Low overhead
- **Updates**: O(1) for inserts/deletes

#### Best For
- Exact match lookups only
- High-cardinality columns with frequent equality queries
- Session storage, caching layers

#### Example
```sql
CREATE INDEX idx_sessions_token
USING hash (session_token);
```

### GIN Indexes (Generalized Inverted Index)

GIN indexes are ideal for full-text search, arrays, and JSONB columns.

#### Characteristics
- **Structure**: Inverted index mapping values to document IDs
- **Operations**: Efficient for containment, overlap, and full-text search
- **Storage**: Higher overhead than B-Tree
- **Updates**: Slower than B-Tree due to inverted structure

#### Best For
- Full-text search
- Array columns
- JSONB data
- Multi-value attributes

#### Example
```sql
-- Index for array columns
CREATE INDEX idx_products_tags
ON products USING GIN (tags);

-- Index for JSONB
CREATE INDEX idx_orders_metadata
ON orders USING GIN (metadata);

-- Full-text search index
CREATE INDEX idx_articles_content
ON articles
USING GIN (to_tsvector('english', content));
```

### BRIN Indexes (Block Range Index)

BRIN indexes are extremely compact and efficient for naturally ordered data like time-series.

#### Characteristics
- **Structure**: Summarizes blocks of data rather than individual rows
- **Operations**: Efficient for range queries on ordered data
- **Storage**: Very low overhead (often <1% of table size)
- **Updates**: Very fast

#### Best For
- Time-series data
- Append-only tables
- Very large datasets with natural ordering
- Date/time columns

#### Example
```sql
CREATE INDEX idx_measurements_time
ON measurements USING BRIN (recorded_at);
```

## Index Selection Guidelines

### When to Use Each Index Type

| Index Type | Equality | Range | Sorting | Full-Text | Arrays | Storage Overhead |
|------------|----------|-------|---------|-----------|--------|------------------|
| B-Tree | ✓ | ✓ | ✓ | ✗ | ✗ | Medium |
| Hash | ✓ | ✗ | ✗ | ✗ | ✗ | Low |
| GIN | ✓ | ✗ | ✗ | ✓ | ✓ | High |
| BRIN | ✓ | ✓ | ✗ | ✗ | ✗ | Very Low |

### Selectivity Rules

1. **High selectivity** (>10% unique values): B-Tree usually best
2. **Low selectivity** (<1% unique values): Consider BRIN or no index
3. **Very high cardinality**: Hash indexes for equality only
4. **Multi-value data**: GIN indexes

### Composite Index Design

When creating composite indexes (multiple columns), order matters:

```sql
-- Good: Leading column has high selectivity
CREATE INDEX idx_orders_customer_status
ON orders(customer_id, status);

-- Bad: Leading column has low selectivity
CREATE INDEX idx_orders_status_customer
ON orders(status, customer_id); -- status likely has low selectivity
```

## Index Maintenance

Indexes need regular maintenance as data changes:

### Monitoring Unused Indexes
```sql
-- Check unused indexes
SELECT indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0;
```

### Rebuilding Fragmented Indexes
```sql
-- Rebuild fragmented index
REINDEX INDEX CONCURRENTLY idx_orders_customer_id;

-- Analyze table to update statistics
ANALYZE orders;
```

### Index Size Monitoring
```sql
-- Check index sizes
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

## AI/ML Engineering Considerations

For AI/ML applications, indexing strategies are particularly important for:

### Vector Search Optimization
- **IVF (Inverted File)**: Partition vectors into clusters for faster search
- **HNSW (Hierarchical Navigable Small World)**: Graph-based indexing for approximate nearest neighbors
- **PQ (Product Quantization)**: Compress vectors for memory efficiency

### Time-Series Data
- **Time-based partitioning**: Natural BRIN index candidates
- **Composite indexes**: (device_id, timestamp) for IoT data
- **Materialized views**: Pre-aggregated metrics for fast dashboards

### Real-time Analytics
- **Covering indexes**: Include all needed columns to avoid table lookups
- **Partial indexes**: Focus on active data segments
- **Function-based indexes**: Index computed values (e.g., `LOWER(email)`)

## Common Indexing Mistakes

1. **Over-indexing**: Too many indexes slow down writes
2. **Under-indexing**: Missing critical indexes for query patterns
3. **Wrong column order**: In composite indexes
4. **Ignoring statistics**: Not updating query planner statistics
5. **Not monitoring usage**: Keeping unused indexes

## Related Resources

- [Query Processing Basics] - How indexes affect query execution plans
- [Performance Optimization] - Advanced indexing techniques
- [Vector Databases] - Specialized indexing for AI/ML workloads
- [Database Tuning] - Comprehensive performance optimization guide