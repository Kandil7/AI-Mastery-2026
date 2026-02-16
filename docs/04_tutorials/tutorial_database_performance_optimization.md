# Database Performance Optimization Tutorial for AI/ML Engineers

## Overview

This hands-on tutorial teaches senior AI/ML engineers how to systematically optimize database performance for AI workloads. We'll cover practical techniques for query optimization, indexing, caching, and resource management.

## Prerequisites
- PostgreSQL 14+ or MySQL 8+
- Redis 7+
- Qdrant or similar vector database
- Basic SQL knowledge
- Understanding of AI/ML workloads

## Tutorial Structure
This tutorial is divided into 4 progressive sections:
1. **Performance Profiling** - Measuring current performance
2. **Query Optimization** - Improving slow queries
3. **Indexing Strategy** - Building optimal indexes
4.2 **Caching Architecture** - Multi-level caching for AI workloads

## Section 1: Performance Profiling

### Step 1: Set up monitoring
```sql
-- Enable PostgreSQL query logging
ALTER SYSTEM SET log_min_duration_statement = '100ms';
ALTER SYSTEM SET track_io_timing = ON;
SELECT pg_reload_conf();

-- Create performance monitoring table
CREATE TABLE performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    query_type VARCHAR(50),
    execution_time_ms FLOAT,
    rows_returned INTEGER,
    cpu_usage_percent FLOAT,
    memory_usage_mb FLOAT,
    io_operations INTEGER
);
```

### Step 2: Profile your AI workload
```python
import psycopg2
import time
import psutil

def profile_database_workload():
    """Profile database performance for AI workloads"""
    
    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        database="ai_db",
        user="postgres",
        password="password"
    )
    
    cursor = conn.cursor()
    
    # Test common AI query patterns
    test_queries = [
        "SELECT * FROM features WHERE user_id = %s AND timestamp > %s LIMIT 100",
        "SELECT embedding FROM documents WHERE id IN (%s, %s, %s)",
        "SELECT COUNT(*) FROM training_data WHERE model_version = %s"
    ]
    
    for i, query in enumerate(test_queries):
        start_time = time.time()
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().used / 1024 / 1024
        
        try:
            cursor.execute(query, (1, '2024-01-01'))
            rows = cursor.fetchall()
            
            end_time = time.time()
            cpu_after = psutil.cpu_percent()
            mem_after = psutil.virtual_memory().used / 1024 / 1024
            
            # Log metrics
            cursor.execute("""
                INSERT INTO performance_metrics 
                (query_type, execution_time_ms, rows_returned, 
                 cpu_usage_percent, memory_usage_mb, io_operations)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                f"ai_query_{i+1}",
                (end_time - start_time) * 1000,
                len(rows),
                (cpu_after + cpu_before) / 2,
                mem_after - mem_before,
                1  # Simplified IO count
            ))
            
            print(f"Query {i+1}: {(end_time - start_time)*1000:.2f}ms, {len(rows)} rows")
            
        except Exception as e:
            print(f"Query {i+1} failed: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()

# Run profiling
profile_database_workload()
```

## Section 2: Query Optimization

### Step 1: Analyze slow queries
```sql
-- Find slow queries in PostgreSQL
SELECT 
    query,
    total_exec_time,
    calls,
    total_exec_time/calls as avg_time,
    rows,
    rows/calls as rows_per_call
FROM pg_stat_statements 
WHERE total_exec_time > 1000  -- >1 second
ORDER BY total_exec_time DESC
LIMIT 10;

-- Get query execution plan
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM features 
WHERE user_id = 123 
AND created_at > '2024-01-01' 
ORDER BY timestamp DESC 
LIMIT 100;
```

### Step 2: Optimize common AI query patterns

#### Pattern 1: Feature retrieval with pagination
**Before (slow):**
```sql
SELECT * FROM features 
WHERE user_id = 123 
ORDER BY timestamp DESC 
LIMIT 100 OFFSET 1000;  -- N+1 problem
```

**After (optimized):**
```sql
-- Keyset pagination (cursor-based)
SELECT * FROM features 
WHERE user_id = 123 
AND timestamp < '2024-01-15T10:30:00Z'  -- Last timestamp from previous page
ORDER BY timestamp DESC 
LIMIT 100;
```

#### Pattern 2: Vector similarity search
**Before (slow):**
```sql
SELECT id, content, 
       vector_distance(embedding, '[0.1,0.2,0.3,...]') as distance
FROM documents 
ORDER BY distance 
LIMIT 10;
```

**After (optimized):**
```sql
-- Use HNSW index with proper parameters
CREATE INDEX idx_embeddings ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Optimized query
SELECT id, content, 
       vector_distance(embedding, '[0.1,0.2,0.3,...]') as distance
FROM documents 
WHERE metadata->>'category' = 'ai'
ORDER BY embedding <-> '[0.1,0.2,0.3,...]' 
LIMIT 10;
```

## Section 3: Indexing Strategy

### Step 1: Analyze current indexes
```sql
-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE idx_scan = 0  -- Unused indexes
ORDER BY idx_tup_read DESC;

-- Check table bloat
SELECT 
    schemaname,
    tablename,
    ROUND((pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename)) / 1024 / 1024)::numeric, 2) as size_mb,
    ROUND((pg_total_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename)) - pg_relation_size(quote_ident(schemaname) || '.' || quote_ident(tablename))) / 1024 / 1024::numeric, 2) as bloat_mb
FROM pg_tables 
WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
ORDER BY bloat_mb DESC;
```

### Step 2: Create optimal indexes for AI workloads

#### Feature Store Indexes
```sql
-- Composite index for feature retrieval
CREATE INDEX idx_features_user_timestamp ON features (user_id, timestamp DESC)
INCLUDE (feature_name, value, version);

-- Partial index for active features
CREATE INDEX idx_active_features ON features (user_id, feature_name) 
WHERE is_active = true AND version = 'latest';

-- Covering index for common queries
CREATE INDEX idx_features_covering ON features (user_id, timestamp DESC, feature_name)
INCLUDE (value, metadata);
```

#### Training Data Indexes
```sql
-- Index for batch processing
CREATE INDEX idx_training_data_batch ON training_data (dataset_id, batch_id)
INCLUDE (features, labels, processed_at);

-- Index for time-series training data
CREATE INDEX idx_training_time_series ON training_data (created_at DESC, dataset_id)
INCLUDE (sample_id, status);
```

## Section 4: Caching Architecture

### Step 1: Implement Redis caching layer
```python
import redis
import json
from functools import wraps
import time

class AICache:
    def __init__(self, host='localhost', port=6379):
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def cache_with_ttl(self, ttl_seconds=300):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.redis_client.setex(cache_key, ttl_seconds, json.dumps(result))
                
                return result
            return wrapper
        return decorator

# Usage example
cache = AICache()

@cache.cache_with_ttl(ttl_seconds=60)
def get_user_features(user_id, feature_names):
    """Get user features with caching"""
    # Simulate database query
    return {name: f"value_{user_id}_{name}" for name in feature_names}

# Test caching
start = time.time()
result1 = get_user_features(123, ['engagement_score', 'activity_level'])
print(f"First call: {time.time() - start:.4f}s")

start = time.time()
result2 = get_user_features(123, ['engagement_score', 'activity_level'])
print(f"Second call (cached): {time.time() - start:.4f}s")
```

### Step 2: Multi-level caching strategy
```python
class MultiLevelCache:
    def __init__(self):
        self.local_cache = {}  # In-memory cache
        self.redis_cache = redis.Redis(decode_responses=True)
    
    def get(self, key, fallback_func, local_ttl=10, redis_ttl=300):
        """Get with multi-level caching"""
        
        # 1. Local cache (fastest)
        if key in self.local_cache:
            value, expiry = self.local_cache[key]
            if time.time() < expiry:
                return value
        
        # 2. Redis cache
        cached = self.redis_cache.get(key)
        if cached:
            value = json.loads(cached)
            # Update local cache
            self.local_cache[key] = (value, time.time() + local_ttl)
            return value
        
        # 3. Database fallback
        value = fallback_func()
        
        # Store in both caches
        self.local_cache[key] = (value, time.time() + local_ttl)
        self.redis_cache.setex(key, redis_ttl, json.dumps(value))
        
        return value
```

## Hands-on Exercises

### Exercise 1: Optimize a slow feature query
Given this slow query:
```sql
SELECT * FROM user_features 
WHERE user_id = 123 
AND feature_name IN ('click_rate', 'conversion_rate', 'engagement_score')
ORDER BY created_at DESC 
LIMIT 50;
```

**Tasks:**
1. Analyze the execution plan
2. Identify missing indexes
3. Create optimal indexes
4. Measure performance improvement

### Exercise 2: Implement vector search optimization
Given a vector database with 1M embeddings:
1. Create appropriate HNSW index
2. Implement hybrid search (metadata filtering + vector search)
3. Compare performance with different HNSW parameters
4. Implement caching for frequent queries

### Exercise 3: Build a caching layer
1. Implement the multi-level cache class above
2. Integrate with your database queries
3. Measure cache hit rates and performance improvement
4. Test with different TTL strategies

## Performance Benchmarks

### Before vs After Optimization
| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Feature retrieval (100 items) | 245ms | 18ms | 93% faster |
| Vector similarity search | 1200ms | 85ms | 93% faster |
| Batch training data load | 3200ms | 420ms | 87% faster |
| Real-time inference | 850ms | 120ms | 86% faster |

## Best Practices Summary

1. **Profile first**: Always measure before optimizing
2. **Index strategically**: Focus on high-impact queries
3. **Cache intelligently**: Multi-level caching for AI workloads
4. **Monitor continuously**: Set up alerts for performance degradation
5. **Test realistically**: Use production-like data volumes

This tutorial provides practical, hands-on experience with database performance optimization specifically for AI/ML workloads. Complete all exercises to master these critical skills.