# Caching Strategies

Caching is a fundamental technique for improving database performance by storing frequently accessed data in faster storage layers. For AI/ML applications, effective caching strategies are critical for reducing latency and scaling systems.

## Overview

Caching sits between the application and database, reducing database load and improving response times. Senior AI/ML engineers need to understand different caching strategies to build high-performance data systems.

## Cache Layers

### Application-Level Caching
- **Location**: Within application code
- **Implementation**: In-memory data structures (HashMap, Redis client)
- **Advantages**: Full control, low latency
- **Disadvantages**: Memory consumption, cache invalidation complexity

### Database-Level Caching
- **Location**: Database buffer pool, query cache
- **Implementation**: Built-in database features
- **Advantages**: Transparent to application, automatic
- **Disadvantages**: Limited control, database-specific

### Proxy-Level Caching
- **Location**: Between application and database (e.g., Redis, Memcached)
- **Implementation**: Dedicated caching service
- **Advantages**: Shared across instances, scalable
- **Disadvantages**: Network overhead, additional infrastructure

## Common Caching Patterns

### Read-Through Caching
- **Pattern**: Cache miss → load from DB → store in cache
- **Implementation**: 
```python
def get_user(user_id):
    cached = cache.get(f"user:{user_id}")
    if cached:
        return cached
    
    user = db.query("SELECT * FROM users WHERE id = %s", [user_id])
    cache.setex(f"user:{user_id}", 300, user)  # 5 minute TTL
    return user
```

### Write-Through Caching
- **Pattern**: Write to cache and DB simultaneously
- **Implementation**:
```python
def update_user(user_id, data):
    db.execute("UPDATE users SET ... WHERE id = %s", [user_id, data])
    cache.setex(f"user:{user_id}", 300, data)
```

### Write-Behind Caching
- **Pattern**: Write to cache first, flush to DB asynchronously
- **Implementation**: Background worker processes
- **Advantages**: High write throughput
- **Disadvantages**: Data loss risk, eventual consistency

### Cache-Aside Pattern
- **Pattern**: Application manages cache explicitly
- **Implementation**: Most common pattern for complex scenarios
- **Advantages**: Fine-grained control
- **Disadvantages**: More complex application logic

## AI/ML Specific Caching Patterns

### Model Metadata Caching
Cache frequently accessed model metadata to reduce database load.

```python
# Model registry caching
class ModelCache:
    def __init__(self):
        self.cache = redis.Redis()
    
    def get_model_version(self, model_id, version=None):
        if version is None:
            # Get latest version
            cache_key = f"model:{model_id}:latest"
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Query database
            model = db.query("""
                SELECT mv.* FROM model_versions mv
                INNER JOIN models m ON mv.model_id = m.model_id
                WHERE m.model_id = %s
                ORDER BY mv.created_at DESC
                LIMIT 1
            """, [model_id])
            
            # Cache result
            self.cache.setex(cache_key, 300, json.dumps(model))
            return model
```

### Feature Store Caching
Optimize feature retrieval for ML inference.

```sql
-- Time-series feature caching
CREATE TABLE feature_cache (
    feature_id UUID NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    ttl TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (feature_id, entity_id, timestamp)
);

-- Cache invalidation strategy
CREATE OR REPLACE FUNCTION invalidate_feature_cache(feature_id UUID, entity_id VARCHAR)
RETURNS VOID AS $$
BEGIN
    DELETE FROM feature_cache 
    WHERE feature_id = $1 AND entity_id = $2;
END;
$$ LANGUAGE plpgsql;
```

### Query Result Caching
Cache expensive query results for dashboard and reporting.

```python
# Dashboard query caching
def get_daily_metrics():
    cache_key = "daily_metrics:" + datetime.now().date().isoformat()
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Expensive query
    metrics = db.execute("""
        SELECT 
            DATE(order_date) as day,
            COUNT(*) as orders,
            SUM(total_amount) as revenue,
            AVG(total_amount) as avg_order
        FROM orders
        WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(order_date)
        ORDER BY day DESC
    """)
    
    # Cache with short TTL for freshness
    redis.setex(cache_key, 60, json.dumps(metrics))  # 1 minute TTL
    return metrics
```

## Cache Invalidation Strategies

### Time-Based Expiration
- **TTL**: Set time-to-live for cache entries
- **Advantages**: Simple, automatic cleanup
- **Disadvantages**: Stale data until expiration

### Event-Driven Invalidation
- **Pub/Sub**: Invalidate on data changes
- **Implementation**: Database triggers or application events
- **Advantages**: Fresh data, immediate consistency
- **Disadvantages**: Complex setup, potential race conditions

### Write-Invalidate Pattern
- **Pattern**: Invalidate cache on write operations
- **Implementation**: Application-level coordination
- **Advantages**: Strong consistency
- **Disadvantages**: Higher latency on writes

### Example - Event-Driven Invalidation
```sql
-- Database trigger for cache invalidation
CREATE OR REPLACE FUNCTION invalidate_user_cache()
RETURNS TRIGGER AS $$
BEGIN
    -- Publish cache invalidation event
    PERFORM pg_notify('cache_invalidate', 'user:' || NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_cache_invalidate
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION invalidate_user_cache();
```

## Performance Considerations

### Cache Hit Ratio Monitoring
```sql
-- Redis cache statistics
redis-cli info stats | grep -E "(keyspace|hit|miss)"

-- Application-level monitoring
SELECT 
    'cache_hit_ratio' as metric,
    (hits::float / NULLIF(hits + misses, 0)) * 100 as percentage
FROM cache_stats;
```

### Memory Management
- **LRU eviction**: Remove least recently used items
- **LFU eviction**: Remove least frequently used items
- **TTL-based**: Expire based on time
- **Size-based**: Limit total cache size

### Multi-Level Caching
Combine multiple cache layers for optimal performance:

```
Application Memory → Redis (Local) → Redis (Cluster) → Database
      ↑                    ↑                   ↑
   Fastest              Medium               Slowest
   Smallest           Medium size          Largest
```

## Best Practices

1. **Start simple**: Begin with basic TTL-based caching
2. **Monitor hit ratios**: Optimize cache effectiveness
3. **Consider consistency requirements**: Choose appropriate invalidation strategy
4. **Test under load**: Ensure caching doesn't become bottleneck
5. **Implement fallbacks**: Handle cache failures gracefully
6. **Document cache policies**: What's cached and why

## Related Resources

- [Database Performance Tuning] - Comprehensive performance optimization
- [Query Optimization] - How caching affects query patterns
- [AI/ML System Design] - Caching in ML system architecture
- [Redis for Realtime] - Practical Redis implementation guide