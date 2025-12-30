"""
Production Caching Module
=========================
Caching strategies for ML systems including LRU cache,
Redis integration, and embedding caches.

Performance Considerations:
- Memory efficiency
- Cache invalidation strategies
- TTL management
- Distributed caching

Author: AI-Mastery-2026
"""

import os
import time
import json
import hashlib
import logging
from typing import Any, Optional, Dict, Callable, TypeVar, Generic, List
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import pickle
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================
# CACHE INTERFACES
# ============================================================

class CacheInterface(Generic[T]):
    """Abstract interface for cache implementations."""
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL (seconds)."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError
    
    def size(self) -> int:
        """Get number of items in cache."""
        raise NotImplementedError


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: Optional[int] = None
    hits: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_set_operations: int = 0
    total_get_operations: int = 0
    memory_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "evictions": self.evictions,
            "total_operations": self.total_set_operations + self.total_get_operations,
            "memory_bytes": self.memory_bytes
        }


# ============================================================
# LRU CACHE (In-Memory)
# ============================================================

class LRUCache(CacheInterface[T]):
    """
    Least Recently Used (LRU) Cache.
    
    Thread-safe implementation with eviction policy.
    
    Algorithm:
        - Uses OrderedDict for O(1) access and ordering
        - Most recently used items moved to end
        - Least recently used items evicted from start
    
    Example:
        >>> cache = LRUCache(max_size=100)
        >>> cache.set("key1", "value1")
        >>> cache.get("key1")
        'value1'
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        """
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[T]:
        """Get value, moving to end (most recently used)."""
        with self._lock:
            self._stats.total_get_operations += 1
            
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1
            
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        with self._lock:
            self._stats.total_set_operations += 1
            
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Remove existing key first
            if key in self._cache:
                del self._cache[key]
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                # Remove oldest (first) item
                evicted_key, _ = self._cache.popitem(last=False)
                self._stats.evictions += 1
                logger.debug(f"Evicted cache key: {evicted_key}")
            
            # Add new entry
            self._cache[key] = CacheEntry(value=value, ttl=ttl)
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False
            
            return True
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            return True
    
    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.memory_bytes = sum(
            len(pickle.dumps(entry.value)) 
            for entry in self._cache.values()
        )
        return self._stats
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1
            
            return len(expired_keys)


# ============================================================
# REDIS CACHE
# ============================================================

class RedisCache(CacheInterface[T]):
    """
    Redis-backed distributed cache.
    
    Features:
        - Distributed caching across multiple nodes
        - Automatic serialization/deserialization
        - Connection pooling
        - Key prefixing for namespacing
    
    Example:
        >>> cache = RedisCache(url="redis://localhost:6379", prefix="ml_cache:")
        >>> cache.set("model_prediction", result, ttl=3600)
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        prefix: str = "cache:",
        default_ttl: int = 3600,
        serializer: str = "json"
    ):
        """
        Args:
            url: Redis URL (defaults to REDIS_URL env var)
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
            serializer: Serialization method ("json" or "pickle")
        """
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.serializer = serializer
        self._client = None
        self._stats = CacheStats()
        
        self._connect()
    
    def _connect(self):
        """Initialize Redis connection."""
        try:
            import redis
            self._client = redis.from_url(self.url, decode_responses=False)
            self._client.ping()
            logger.info(f"Connected to Redis at {self.url}")
        except ImportError:
            logger.error("redis package not installed. Run: pip install redis")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._client = None
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.serializer == "json":
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize stored value."""
        if data is None:
            return None
        if self.serializer == "json":
            return json.loads(data.decode('utf-8'))
        else:
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[T]:
        """Get value from Redis."""
        if not self._client:
            return None
        
        self._stats.total_get_operations += 1
        
        try:
            data = self._client.get(self._make_key(key))
            if data is None:
                self._stats.misses += 1
                return None
            
            self._stats.hits += 1
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.misses += 1
            return None
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        if not self._client:
            return False
        
        self._stats.total_set_operations += 1
        ttl = ttl or self.default_ttl
        
        try:
            data = self._serialize(value)
            self._client.setex(self._make_key(key), ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        if not self._client:
            return False
        
        try:
            result = self._client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._client:
            return False
        
        try:
            return self._client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all keys with our prefix."""
        if not self._client:
            return False
        
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=f"{self.prefix}*")
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    def size(self) -> int:
        """Get number of keys with our prefix."""
        if not self._client:
            return 0
        
        try:
            cursor = 0
            count = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=f"{self.prefix}*")
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception as e:
            logger.error(f"Redis size error: {e}")
            return 0
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


# ============================================================
# EMBEDDING CACHE
# ============================================================

class EmbeddingCache:
    """
    Specialized cache for embeddings with content hashing.
    
    Features:
        - Content-based hashing for deduplication
        - Batch get/set operations
        - Optional persistence
    
    Example:
        >>> cache = EmbeddingCache(cache_backend=LRUCache(max_size=10000))
        >>> embeddings = cache.get_or_compute(texts, embed_fn)
    """
    
    def __init__(
        self,
        cache_backend: Optional[CacheInterface] = None,
        hash_algorithm: str = "sha256"
    ):
        """
        Args:
            cache_backend: Underlying cache implementation
            hash_algorithm: Hash algorithm for content keys
        """
        self.cache = cache_backend or LRUCache(max_size=10000)
        self.hash_algorithm = hash_algorithm
    
    def _hash_content(self, content: str) -> str:
        """Create hash key from content."""
        hasher = hashlib.new(self.hash_algorithm)
        hasher.update(content.encode('utf-8'))
        return hasher.hexdigest()[:16]  # Use first 16 chars
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        key = self._hash_content(text)
        return self.cache.get(key)
    
    def set(self, text: str, embedding: List[float], ttl: Optional[int] = None) -> bool:
        """Set embedding for text."""
        key = self._hash_content(text)
        return self.cache.set(key, embedding, ttl)
    
    def get_batch(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get embeddings for multiple texts."""
        return {text: self.get(text) for text in texts}
    
    def set_batch(
        self, 
        texts: List[str], 
        embeddings: List[List[float]],
        ttl: Optional[int] = None
    ) -> int:
        """Set embeddings for multiple texts. Returns count of successful sets."""
        assert len(texts) == len(embeddings), "Texts and embeddings must have same length"
        
        success_count = 0
        for text, embedding in zip(texts, embeddings):
            if self.set(text, embedding, ttl):
                success_count += 1
        
        return success_count
    
    def get_or_compute(
        self,
        texts: List[str],
        embed_fn: Callable[[List[str]], List[List[float]]],
        ttl: Optional[int] = None
    ) -> List[List[float]]:
        """
        Get embeddings from cache or compute and cache.
        
        Efficiently handles mixed cache hits and misses.
        
        Args:
            texts: Texts to embed
            embed_fn: Function to compute embeddings
            ttl: TTL for new cache entries
        
        Returns:
            List of embeddings in same order as texts
        """
        results = [None] * len(texts)
        texts_to_compute = []
        indices_to_compute = []
        
        # Check cache
        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        # Compute missing embeddings
        if texts_to_compute:
            computed = embed_fn(texts_to_compute)
            
            for idx, text, embedding in zip(indices_to_compute, texts_to_compute, computed):
                results[idx] = embedding
                self.set(text, embedding, ttl)
        
        return results
    
    @property
    def stats(self) -> CacheStats:
        """Get underlying cache statistics."""
        if hasattr(self.cache, 'stats'):
            return self.cache.stats
        return CacheStats()


# ============================================================
# CACHE DECORATORS
# ============================================================

def cached(
    cache: CacheInterface,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None
):
    """
    Decorator to cache function results.
    
    Example:
        >>> cache = LRUCache(max_size=100)
        >>> @cached(cache, ttl=3600)
        ... def expensive_computation(x, y):
        ...     return x + y
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{fn.__name__}:{hashlib.md5(str((args, sorted(kwargs.items()))).encode()).hexdigest()}"
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = fn(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator


def async_cached(
    cache: CacheInterface,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None
):
    """
    Async version of cached decorator.
    
    Example:
        >>> @async_cached(cache, ttl=3600)
        ... async def async_expensive_computation(x):
        ...     return await some_async_call(x)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{fn.__name__}:{hashlib.md5(str((args, sorted(kwargs.items()))).encode()).hexdigest()}"
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = await fn(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator


# ============================================================
# TIERED CACHE
# ============================================================

class TieredCache(CacheInterface[T]):
    """
    Multi-tier caching with L1 (memory) and L2 (Redis) layers.
    
    Pattern:
        1. Check L1 (fast, local memory)
        2. Check L2 (distributed, slower)
        3. On L2 hit, populate L1
    
    Example:
        >>> cache = TieredCache(
        ...     l1_cache=LRUCache(max_size=100),
        ...     l2_cache=RedisCache()
        ... )
    """
    
    def __init__(
        self,
        l1_cache: CacheInterface,
        l2_cache: CacheInterface,
        l1_ttl: Optional[int] = 60
    ):
        """
        Args:
            l1_cache: Fast, local cache (e.g., LRU)
            l2_cache: Slower, distributed cache (e.g., Redis)
            l1_ttl: TTL for L1 entries (shorter than L2)
        """
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.l1_ttl = l1_ttl
    
    def get(self, key: str) -> Optional[T]:
        """Get from L1, then L2."""
        # Try L1
        result = self.l1.get(key)
        if result is not None:
            return result
        
        # Try L2
        result = self.l2.get(key)
        if result is not None:
            # Populate L1
            self.l1.set(key, result, self.l1_ttl)
            return result
        
        return None
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> bool:
        """Set in both L1 and L2."""
        # Set in L1 with shorter TTL
        l1_ttl = min(self.l1_ttl, ttl) if ttl else self.l1_ttl
        self.l1.set(key, value, l1_ttl)
        
        # Set in L2 with full TTL
        return self.l2.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete from both caches."""
        self.l1.delete(key)
        return self.l2.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check in L1, then L2."""
        return self.l1.exists(key) or self.l2.exists(key)
    
    def clear(self) -> bool:
        """Clear both caches."""
        self.l1.clear()
        return self.l2.clear()
    
    def size(self) -> int:
        """Return L2 size (authoritative)."""
        return self.l2.size()


# ============================================================
# MODEL PREDICTION CACHE
# ============================================================

class PredictionCache:
    """
    Specialized cache for ML model predictions.
    
    Features:
        - Feature hashing for cache keys
        - Batch prediction caching
        - Model version tracking
    
    Example:
        >>> cache = PredictionCache(model_version="v1.0")
        >>> prediction = cache.get_or_predict(features, model.predict)
    """
    
    def __init__(
        self,
        cache_backend: Optional[CacheInterface] = None,
        model_version: str = "v1",
        feature_keys: Optional[List[str]] = None
    ):
        """
        Args:
            cache_backend: Underlying cache
            model_version: Model version for cache key namespace
            feature_keys: Ordered list of feature names for consistent hashing
        """
        self.cache = cache_backend or LRUCache(max_size=10000)
        self.model_version = model_version
        self.feature_keys = feature_keys
    
    def _hash_features(self, features: Dict[str, Any]) -> str:
        """Create hash from feature dictionary."""
        # Use consistent ordering
        if self.feature_keys:
            ordered = [(k, features.get(k)) for k in self.feature_keys]
        else:
            ordered = sorted(features.items())
        
        feature_str = json.dumps(ordered, sort_keys=True)
        hash_val = hashlib.md5(feature_str.encode()).hexdigest()[:16]
        
        return f"{self.model_version}:{hash_val}"
    
    def get(self, features: Dict[str, Any]) -> Optional[Any]:
        """Get cached prediction."""
        key = self._hash_features(features)
        return self.cache.get(key)
    
    def set(
        self,
        features: Dict[str, Any],
        prediction: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache a prediction."""
        key = self._hash_features(features)
        return self.cache.set(key, prediction, ttl)
    
    def get_or_predict(
        self,
        features: Dict[str, Any],
        predict_fn: Callable[[Dict[str, Any]], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or compute prediction."""
        cached = self.get(features)
        if cached is not None:
            return cached
        
        prediction = predict_fn(features)
        self.set(features, prediction, ttl)
        return prediction
    
    def invalidate_model_version(self):
        """Clear all predictions for current model version."""
        # This is approximate - full clear for simplicity
        # Production would use prefix-based deletion
        self.cache.clear()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Interfaces
    'CacheInterface', 'CacheEntry', 'CacheStats',
    # Implementations
    'LRUCache', 'RedisCache', 'TieredCache',
    # Specialized caches
    'EmbeddingCache', 'PredictionCache',
    # Decorators
    'cached', 'async_cached',
]
