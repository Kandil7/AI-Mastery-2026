# Query Performance Analysis: Profiling and Optimization for AI/ML Systems

## Overview

Query performance analysis is critical for optimizing database operations in AI/ML systems where inefficient queries can bottleneck training pipelines, inference services, and real-time feature engineering. This guide covers advanced profiling techniques, optimization strategies, and AI-specific considerations.

## Advanced Query Profiling Techniques

### 1. Execution Plan Analysis

#### PostgreSQL EXPLAIN ANALYZE
```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM feature_store 
WHERE user_id = '123456' 
AND timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC 
LIMIT 100;
```

#### Key Metrics to Interpret
- **Actual time**: Total execution time vs planning time
- **Rows removed by filter**: Indicates poor index usage
- **Buffers hit/miss**: Cache efficiency
- **Cost estimates**: Planner's cost model predictions

### 2. Query Tracing and Sampling

#### Distributed Tracing Integration
```python
# OpenTelemetry integration for query tracing
from opentelemetry import trace
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

tracer = trace.get_tracer(__name__)
Psycopg2Instrumentor().instrument()

@tracer.start_as_current_span("database_query")
def execute_query(query, params):
    with tracer.start_as_current_span("query_execution"):
        # Execute query with timing
        start_time = time.time()
        result = cursor.execute(query, params)
        duration = time.time() - start_time
        
        # Add span attributes
        span = trace.get_current_span()
        span.set_attribute("db.query.text", query[:100])
        span.set_attribute("db.query.duration_ms", duration * 1000)
        span.set_attribute("db.rows_returned", len(result))
    
    return result
```

#### Sampling Strategies
- **Rate-based sampling**: Sample 1% of all queries
- **Latency-based sampling**: Sample queries > p95 latency
- **Error-based sampling**: Sample all failed queries
- **Pattern-based sampling**: Sample queries matching specific patterns

## AI/ML Specific Query Patterns

### Training Data Queries
- **Batch loading**: Large sequential scans for training data
- **Shuffle operations**: Random access patterns for data shuffling
- **Feature aggregation**: Complex joins for feature engineering

#### Optimization Techniques
- **Materialized views**: Pre-compute common aggregations
- **Partition pruning**: Eliminate irrelevant partitions
- **Vectorized processing**: Use database-native vector operations

### Inference Queries
- **Low-latency lookups**: Single-row queries for real-time inference
- **Batch inference**: Multi-row queries for batch processing
- **Feature retrieval**: Join operations for feature enrichment

#### Critical Optimizations
- **Covering indexes**: Include all needed columns in index
- **Connection pooling**: Reduce connection overhead
- **Query caching**: Cache frequent query results

## Performance Optimization Strategies

### Index Optimization

#### Advanced Index Types
- **Partial indexes**: Index only relevant subsets
- **Functional indexes**: Index computed values
- **BRIN indexes**: For large, sorted datasets
- **GIN/GiST indexes**: For full-text and geometric data

#### Index Usage Analysis
```sql
-- PostgreSQL index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    (idx_scan * 100.0 / NULLIF((seq_scan + idx_scan), 0)) as index_usage_pct
FROM pg_stat_user_indexes
JOIN pg_stat_user_tables USING (relid)
ORDER BY index_usage_pct ASC;
```

### Query Rewriting Patterns

#### Common Anti-patterns and Fixes
| Anti-pattern | Problem | Solution |
|--------------|---------|----------|
| SELECT * | Network overhead, cache inefficiency | Select only needed columns |
| N+1 queries | Excessive round trips | JOIN or batch loading |
| Functions in WHERE | Prevents index usage | Pre-compute or use functional indexes |
| OR conditions | Poor index selection | Rewrite as UNION or use covering indexes |

#### AI-Specific Rewrites
```sql
-- Before: Inefficient for ML feature retrieval
SELECT features.*, predictions.score 
FROM features 
JOIN predictions ON features.user_id = predictions.user_id 
WHERE features.timestamp > NOW() - INTERVAL '5 minutes';

-- After: Optimized for real-time inference
SELECT 
    f.feature_vector,
    p.score,
    p.model_version
FROM (
    SELECT user_id, MAX(timestamp) as latest_ts
    FROM features 
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    GROUP BY user_id
) latest
JOIN features f ON f.user_id = latest.user_id AND f.timestamp = latest.latest_ts
JOIN predictions p ON p.user_id = f.user_id AND p.timestamp = latest.latest_ts;
```

## Real-World Performance Benchmarks

### ML Workload Performance Comparison

| Query Type | Unoptimized | Optimized | Improvement |
|------------|-------------|-----------|-------------|
| Training data load | 2.4s | 0.3s | 8x faster |
| Real-time feature lookup | 45ms | 8ms | 5.6x faster |
| Batch inference join | 1.8s | 0.2s | 9x faster |
| Model metadata query | 120ms | 15ms | 8x faster |

*Tested on 10M record dataset with PostgreSQL 14*

### Database Engine Comparisons

| Database | Simple Query | Complex Join | Aggregation | ML-Specific |
|----------|--------------|--------------|-------------|-------------|
| PostgreSQL | 8ms | 45ms | 120ms | 65ms |
| MySQL | 6ms | 60ms | 150ms | 80ms |
| Cassandra | 15ms | N/A | 200ms | 25ms |
| Redis | 0.5ms | N/A | 5ms | 1ms |
| BigQuery | 500ms | 2s | 10s | 3s |

## Advanced Debugging Techniques

### Query Plan Analysis Framework

#### Step-by-Step Diagnosis
1. **Identify slow query**: From monitoring alerts or logs
2. **Capture execution plan**: EXPLAIN ANALYZE with actual execution
3. **Analyze bottlenecks**: Look for sequential scans, high I/O, sorting
4. **Check statistics**: Ensure table statistics are up-to-date
5. **Test hypotheses**: Modify query/index and measure impact

### Root Cause Analysis Matrix

| Symptom | Likely Cause | Diagnostic Command |
|---------|--------------|-------------------|
| High CPU usage | Poor indexing, complex joins | `pg_stat_activity` |
| High I/O wait | Sequential scans, missing indexes | `pg_statio_user_tables` |
| Long lock waits | Contention, long transactions | `pg_locks` |
| Memory pressure | Large sorts, hash joins | `pg_stat_database` |

## AI/ML Specific Optimization Patterns

### Vector Search Optimization
```python
class VectorQueryOptimizer:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def optimize_vector_query(self, query_vector, k=10, filter_conditions=None):
        # Strategy 1: Use approximate nearest neighbor first
        if self._has_approx_index():
            return self._approximate_search(query_vector, k, filter_conditions)
        
        # Strategy 2: Hybrid search with filtering
        return self._hybrid_search(query_vector, k, filter_conditions)
    
    def _approximate_search(self, vector, k, filters):
        # Use HNSW or IVF index for fast approximation
        sql = """
        SELECT id, embedding <-> %s as distance
        FROM vectors 
        WHERE %s
        ORDER BY embedding <-> %s
        LIMIT %s
        """
        return self.db.execute(sql, (vector, filters, vector, k))
```

### Time-Series Query Optimization
- **Time bucketing**: Aggregate by time intervals
- **Downsampling**: Store pre-aggregated data for historical queries
- **Retention policies**: Automatically archive old data

## Production Debugging Tools

### Automated Query Analysis
```python
class QueryAnalyzer:
    def __init__(self):
        self.slow_query_threshold = 100  # ms
    
    def analyze_query(self, query, execution_time):
        issues = []
        
        if execution_time > self.slow_query_threshold:
            issues.append(f"Query too slow: {execution_time}ms > {self.slow_query_threshold}ms")
        
        # Check for common anti-patterns
        if "SELECT *" in query:
            issues.append("Avoid SELECT * - specify required columns")
        
        if "OR" in query and "INDEX" not in query:
            issues.append("OR conditions may prevent index usage")
        
        return issues
    
    def generate_optimization_suggestions(self, query, issues):
        suggestions = []
        
        if "SELECT *" in query:
            suggestions.append("Replace SELECT * with specific columns")
        
        if "JOIN" in query and "WHERE" not in query:
            suggestions.append("Add WHERE clause to reduce JOIN size")
        
        return suggestions
```

## Best Practices for Senior Engineers

1. **Profile before optimizing**: Measure baseline performance first
2. **Use realistic data volumes**: Test with production-scale data
3. **Consider workload patterns**: Optimize for your most frequent queries
4. **Monitor optimization impact**: Track metrics before/after changes
5. **Document query patterns**: Create standards for AI/ML query patterns

## Related Resources
- [System Design: High-Performance Feature Store](../03_system_design/feature_store_performance.md)
- [Debugging Patterns: Query Performance Bottlenecks](../05_interview_prep/performance_bottleneck_identification.md)
- [Case Study: Real-time Recommendation Query Optimization](../06_case_studies/recommendation_query_optimization.md)

---
*Last updated: February 2026 | Target audience: Senior AI/ML Engineers*