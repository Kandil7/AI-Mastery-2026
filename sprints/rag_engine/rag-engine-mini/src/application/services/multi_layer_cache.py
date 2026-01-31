"""
Multi-Layer Caching Service
=========================
Service for managing caching at multiple layers.

خدمة إدارة التخزين المؤقت في طبقات متعددة
"""

from typing import Optional, Any
from functools import lru_cache
from datetime import datetime, timedelta


class CacheLayer:
    """Cache layer types."""

    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


class CacheService:
    """Multi-layer caching service."""

    def __init__(self, memory_cache, redis_cache, db_cache):
        self._memory = memory_cache
        self._redis = redis_cache
        self._database = db_cache

    async def get(self, key: str, layer: Optional[str] = None) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            layer: Specific layer (default: check all layers)

        Returns:
            Cached value or None
        """
        # Layer 1: In-memory cache (fastest)
        value = self._memory.get(key)
        if value is not None:
            return value

        # Layer 2: Redis cache (fast)
        if layer != CacheLayer.MEMORY:
            value = await self._redis.get(key)
            if value is not None:
                return value

        # Layer 3: Database cache (slowest)
        if layer != CacheLayer.MEMORY and layer != CacheLayer.REDIS:
            value = await self._database.get(key)
            if value is not None:
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600,
        layer: str = CacheLayer.REDIS,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (default: 3600)
            layer: Cache layer to use (default: Redis)
        """
        expiry = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        if layer == CacheLayer.MEMORY or layer is None:
            self._memory.set(key, value)

        if layer == CacheLayer.REDIS or layer is None:
            await self._redis.set(key, value, ttl=expiry)

        if layer == CacheLayer.DATABASE or layer is None:
            await self._database.set(key, value, expiry)

    async def delete(self, key: str, layer: str = None) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key
            layer: Specific layer (default: all layers)

        Returns:
            True if deleted
        """
        success = False

        if layer == CacheLayer.MEMORY or layer is None:
            try:
                del self._memory[key]
                success = True
            except KeyError:
                pass

        if layer == CacheLayer.REDIS or layer is None:
            await self._redis.delete(key)
            success = True

        if layer == CacheLayer.DATABASE or layer is None:
            await self._database.delete(key)
            success = True

        return success

    async def clear(self, layer: str = None) -> int:
        """
        Clear all cache or specific layer.

        Args:
            layer: Cache layer to clear (default: all)

        Returns:
            Number of keys cleared
        """
        count = 0

        if layer == CacheLayer.MEMORY or layer is None:
            self._memory.cache_clear()
            count += 1

        if layer == CacheLayer.REDIS or layer is None:
            await self._redis.flush()
            count += 1

        if layer == CacheLayer.DATABASE or layer is None:
            await self._database.flush()
            count += 1

        return count

    def get_hit_rate(self, hits: int, total: int) -> float:
        """
        Calculate cache hit rate.

        Args:
            hits: Number of cache hits
            total: Total number of lookups

        Returns:
            Hit rate as percentage (0.0 to 1.0)
        """
        if total == 0:
            return 0.0
        return hits / total


# In-memory cache using LRU
@lru_cache(maxsize=1000)
def _memory_cache_get(key: str) -> Optional[Any]:
    """In-memory cache get."""
    # Implemented by lru_cache decorator
    pass
