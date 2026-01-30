# Caching Strategies for RAG Engine

## Overview

Comprehensive caching strategies to optimize RAG performance across multiple layers.

استراتيجيات التخزين المؤقت لتحسين أداء RAG Engine

---

## Cache Types

### 1. Embedding Cache (Already Implemented)

```python
# CachedEmbeddings in src/application/services/embedding_cache.py
# - In-memory cache for embeddings
# - TTL: 1 hour
# - LRU eviction policy
```

### 2. Query Response Cache

```python
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import hashlib
import json

log = logging.getLogger(__name__)


@dataclass
class QueryCacheEntry:
    """Cache entry for query results."""
    answer: str
    sources: List[str]
    retrieval_time_ms: float
    llm_time_ms: float
    timestamp: datetime
    hit_count: int


class QueryResponseCache:
    """Cache for query responses."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize query response cache.
        
        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live in seconds (default: 1 hour)
        """
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, QueryCacheEntry] = {}
    
    def _generate_key(
        self,
        question: str,
        k: int,
        document_id: Optional[str],
    ) -> str:
        """Generate cache key from parameters."""
        key_parts = [question, str(k)]
        if document_id:
            key_parts.append(document_id)
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(
        self,
        question: str,
        k: int,
        document_id: Optional[str] = None,
    ) -> Optional[QueryCacheEntry]:
        """
        Get cached query response.
        
        Args:
            question: User question
            k: Number of chunks retrieved
            document_id: Optional document ID for chat mode
        
        Returns:
            Cached response or None
        
        الحصول على استجابة مخبأة للسؤال
        """
        key = self._generate_key(question, k, document_id)
        entry = self._cache.get(key)
        
        if not entry:
            return None
        
        # Check TTL
        if datetime.utcnow() - entry.timestamp > self._ttl:
            log.debug("Cache entry expired", key=key)
            del self._cache[key]
            return None
        
        # Increment hit count
        entry.hit_count += 1
        log.info("Cache hit", key=key, hit_count=entry.hit_count)
        
        return entry
    
    def set(
        self,
        question: str,
        k: int,
        answer: str,
        sources: List[str],
        retrieval_time_ms: float,
        llm_time_ms: float,
        document_id: Optional[str] = None,
    ) -> None:
        """
        Cache query response.
        
        Args:
            question: User question
            k: Number of chunks
            answer: LLM-generated answer
            sources: Document IDs used
            retrieval_time_ms: Retrieval duration
            llm_time_ms: LLM generation duration
            document_id: Optional document ID for chat mode
        
        تخزين استجابة السؤال
        """
        key = self._generate_key(question, k, document_id)
        
        entry = QueryCacheEntry(
            answer=answer,
            sources=sources,
            retrieval_time_ms=retrieval_time_ms,
            llm_time_ms=llm_time_ms,
            timestamp=datetime.utcnow(),
            hit_count=0,
        )
        
        # LRU eviction if cache is full
        if len(self._cache) >= self._max_size:
            # Remove oldest entry (simple implementation)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            log.debug("Cache evicted (LRU)", key=oldest_key)
        
        self._cache[key] = entry
        log.info("Cache set", key=key, answer_length=len(answer))
    
    def invalidate(self, question: str, k: int, document_id: Optional[str] = None):
        """Invalidate cache entry."""
        key = self._generate_key(question, k, document_id)
        
        if key in self._cache:
            del self._cache[key]
            log.debug("Cache invalidated", key=key)
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        log.info("Cache cleared", entries_before=len(self._cache))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._cache.values())
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "total_hits": total_hits,
            "hit_rate": total_hits / len(self._cache) if self._cache else 0,
        }
```

### 3. Document Metadata Cache

```python
class DocumentMetadataCache:
    """Cache for document metadata."""
    
    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata from cache."""
        return self._cache.get(document_id)
    
    def set_document(self, document_id: str, metadata: Dict[str, Any]):
        """Cache document metadata."""
        # LRU eviction
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[document_id] = metadata
```

### 4. Distributed Cache (Redis)

```python
import redis
import json

class RedisCache:
    """Distributed cache using Redis."""
    
    def __init__(self, redis_url: str):
        """Initialize Redis connection."""
        self._pool = redis.ConnectionPool(
            host=redis_url.split(':')[0].replace('redis://', ''),
            port=int(redis_url.split(':')[1].split('/')[0]),
            db=int(redis_url.split('/')[2]),
            max_connections=50,
            decode_responses=True,
        )
        self._client = redis.Redis(connection_pool=self._pool)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            log.error("Redis GET failed", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set value in Redis with TTL."""
        try:
            serialized = json.dumps(value)
            self._client.setex(key, ttl_seconds, serialized)
            log.debug("Redis SET", key=key, ttl=ttl_seconds)
        except Exception as e:
            log.error("Redis SET failed", key=key, error=str(e))
    
    def delete(self, key: str):
        """Delete key from Redis."""
        try:
            self._client.delete(key)
            log.debug("Redis DELETE", key=key)
        except Exception as e:
            log.error("Redis DELETE failed", key=key, error=str(e))
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            log.error("Redis EXISTS failed", key=key, error=str(e))
            return False
```

---

## Cache Configuration

### Multi-Layer Caching

```python
from functools import lru_cache
from typing import Callable

# Layer 1: In-memory (fastest)
@lru_cache(maxsize=100)
def cached_embedding(text: str) -> List[float]:
    """In-memory cache for embeddings (LRU)."""
    # ... implementation
    pass

# Layer 2: Redis (shared across instances)
redis_cache = RedisCache(redis_url="redis://localhost:6379/0")

# Layer 3: Database (fallback)
def get_embedding_from_db(text: str) -> List[float]:
    """Get embedding from database (slowest)."""
    # ... implementation
    pass

def get_embedding(text: str) -> List[float]:
    """Multi-layer caching strategy."""
    # Try in-memory first
    embedding = cached_embedding(text)
    if embedding:
        return embedding
    
    # Try Redis second
    redis_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
    embedding = redis_cache.get(redis_key)
    if embedding:
        return embedding
    
    # Fallback to database
    return get_embedding_from_db(text)
```

---

## Cache Invalidation Strategies

### TTL-Based Invalidation

```python
# Set TTL for all cache entries
CACHE_TTLS = {
    "embedding": 3600,  # 1 hour
    "query_response": 1800,  # 30 minutes
    "document_metadata": 86400,  # 24 hours
}
```

### Event-Based Invalidation

```python
def invalidate_document_cache(document_id: str):
    """Invalidate all caches for a document."""
    # Invalidate query cache
    query_cache.invalidate_by_document(document_id)
    
    # Invalidate metadata cache
    metadata_cache.invalidate_document(document_id)
    
    # Invalidate Redis cache
    redis_cache.delete(f"doc:{document_id}")
    redis_cache.delete(f"doc:{document_id}:metadata")
```

---

## Cache Monitoring

### Cache Metrics

```python
from prometheus_client import Counter, Histogram

CACHE_HITS = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
CACHE_LATENCY = Histogram('cache_lookup_duration_seconds', 'Cache lookup latency', ['cache_type'])

def track_cache_hit(cache_type: str):
    """Track cache hit."""
    CACHE_HITS.labels(cache_type=cache_type).inc()

def track_cache_miss(cache_type: str):
    """Track cache miss."""
    CACHE_MISSES.labels(cache_type=cache_type).inc()

def track_cache_latency(cache_type: str, duration_seconds: float):
    """Track cache lookup latency."""
    CACHE_LATENCY.labels(cache_type=cache_type).observe(duration_seconds)
```

---

## Summary

| Cache Type | Purpose | Implementation | TTL |
|-------------|---------|-----------------|-----|
| **Embedding** | LLM API cost reduction | In-memory + Redis | 1 hour |
| **Query Response** | Reduce latency | Redis | 30 min |
| **Document Metadata** | Fast lookups | Redis | 24 hours |
| **Search Results** | Performance | In-memory LRU | 5 min |

---

## Best Practices

1. **Layered Caching**: In-memory → Redis → Database
2. **Appropriate TTLs**: Balance freshness and performance
3. **LRU Eviction**: Remove least recently used items
4. **Cache Warming**: Pre-populate cache on startup
5. **Monitoring**: Track hit rates and latency
6. **Invalidation**: Clear cache on updates
7. **Distributed**: Use Redis for multi-instance setups

---

## Further Reading

- [Redis Caching Best Practices](https://redis.io/topics/lru-cache)
- [Caching Patterns](https://docs.microsoft.com/en-us/azure/architecture/patterns/caching)
- [Cache Invalidation](https://docs.aws.amazon.com/AmazonElastiCache/latest/UserGuide/DataInvalidation.html)
