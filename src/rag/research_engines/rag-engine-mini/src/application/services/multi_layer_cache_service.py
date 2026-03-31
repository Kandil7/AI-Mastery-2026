"""
Multi-Layer Caching Service

This module implements a multi-layer caching strategy for the RAG Engine,
with support for L1 (in-memory), L2 (Redis), and L3 (persistent) caches.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import asyncio
import hashlib
import pickle
import json
from dataclasses import dataclass


class CacheLayer(str, Enum):
    """Cache layers in the multi-layer cache system."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_PERSISTENT = "l3_persistent"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    layer: Optional[CacheLayer] = None
    size_bytes: Optional[int] = None


class MultiLayerCachePort(ABC):
    """Abstract port for multi-layer caching services."""

    @abstractmethod
    async def get(self, key: str, layer: Optional[CacheLayer] = None) -> Optional[Any]:
        """Get a value from cache. If layer is None, tries all layers in order."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None, target_layers: Optional[List[CacheLayer]] = None) -> bool:
        """Set a value in cache. If target_layers is None, sets in all layers."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from all cache layers."""
        pass

    @abstractmethod
    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all keys with a given prefix across all layers."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[CacheLayer, Dict[str, Any]]:
        """Get statistics for all cache layers."""
        pass

    @abstractmethod
    async def warm_up(self, data: Dict[str, Any], ttl: Optional[timedelta] = None) -> bool:
        """Warm up the cache with a batch of data."""
        pass

    @abstractmethod
    async def clear_layer(self, layer: CacheLayer) -> bool:
        """Clear all entries in a specific cache layer."""
        pass


class MultiLayerCacheService(MultiLayerCachePort):
    """Concrete implementation of the multi-layer caching service."""

    def __init__(self, redis_client=None, persistent_store=None):
        """Initialize the multi-layer cache with different storage backends."""
        # L1: In-memory cache (simple dict with size limits)
        self._l1_cache: Dict[str, CacheEntry] = {}
        self._l1_max_size = 1000  # Max items in L1 cache
        self._l1_current_size = 0  # Current size in bytes (approximated)
        self._l1_max_bytes = 10 * 1024 * 1024  # 10 MB limit

        # L2: Redis cache (passed in via dependency injection)
        self._l2_cache = redis_client

        # L3: Persistent cache (could be database, file system, etc.)
        self._l3_cache = persistent_store

        # Statistics
        self._hits = {layer: 0 for layer in CacheLayer}
        self._misses = {layer: 0 for layer in CacheLayer}

    def _calculate_size(self, obj: Any) -> int:
        """Calculate approximate size of an object in bytes."""
        try:
            # For strings, return length
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            # For other objects, serialize and measure
            serialized = pickle.dumps(obj)
            return len(serialized)
        except:
            # If serialization fails, return a default size
            return 1024

    def _generate_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Generate a full cache key with optional namespace."""
        if namespace:
            return f"{namespace}:{key}"
        return key

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry is expired."""
        if entry.expires_at is None:
            return False
        return datetime.now() > entry.expires_at

    async def _clean_expired_l1_entries(self):
        """Remove expired entries from L1 cache."""
        expired_keys = []
        for key, entry in self._l1_cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            entry = self._l1_cache[key]
            self._l1_current_size -= entry.size_bytes or 0
            del self._l1_cache[key]

    async def get(self, key: str, layer: Optional[CacheLayer] = None) -> Optional[Any]:
        """Get a value from cache. If layer is None, tries all layers in order (L1 -> L2 -> L3)."""
        # If a specific layer is requested, check only that layer
        if layer == CacheLayer.L1_MEMORY:
            await self._clean_expired_l1_entries()
            entry = self._l1_cache.get(key)
            if entry and not self._is_expired(entry):
                self._hits[CacheLayer.L1_MEMORY] += 1
                return entry.value
            self._misses[CacheLayer.L1_MEMORY] += 1
            return None

        elif layer == CacheLayer.L2_REDIS:
            if self._l2_cache:
                try:
                    value = await self._l2_cache.get(key)
                    if value is not None:
                        self._hits[CacheLayer.L2_REDIS] += 1
                        return pickle.loads(value) if isinstance(value, bytes) else value
                except Exception:
                    pass  # Fall through to next layer
            self._misses[CacheLayer.L2_REDIS] += 1
            return None

        elif layer == CacheLayer.L3_PERSISTENT:
            if self._l3_cache:
                try:
                    # In a real implementation, this would query the persistent store
                    # For now, we'll simulate with a placeholder
                    return self._l3_cache.get(key) if hasattr(self._l3_cache, 'get') else None
                except Exception:
                    pass  # Fall through to return None
            self._misses[CacheLayer.L3_PERSISTENT] += 1
            return None

        # If no specific layer requested, check all layers in hierarchy
        # L1 first
        await self._clean_expired_l1_entries()
        entry = self._l1_cache.get(key)
        if entry and not self._is_expired(entry):
            self._hits[CacheLayer.L1_MEMORY] += 1
            return entry.value

        # L2 second
        if self._l2_cache:
            try:
                value = await self._l2_cache.get(key)
                if value is not None:
                    # Promote to L1 if found in L2
                    deserialized_value = pickle.loads(value) if isinstance(value, bytes) else value
                    await self.set(key, deserialized_value, target_layers=[CacheLayer.L1_MEMORY])
                    self._hits[CacheLayer.L2_REDIS] += 1
                    return deserialized_value
            except Exception:
                pass  # Continue to L3

        # L3 last
        if self._l3_cache:
            try:
                value = self._l3_cache.get(key) if hasattr(self._l3_cache, 'get') else None
                if value is not None:
                    # Promote to L1 and L2 if found in L3
                    await self.set(key, value, target_layers=[CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS])
                    self._hits[CacheLayer.L3_PERSISTENT] += 1
                    return value
            except Exception:
                pass

        # Cache miss for all layers
        self._misses[CacheLayer.L1_MEMORY] += 1
        self._misses[CacheLayer.L2_REDIS] += 1
        self._misses[CacheLayer.L3_PERSISTENT] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None, target_layers: Optional[List[CacheLayer]] = None) -> bool:
        """Set a value in cache. If target_layers is None, sets in all layers."""
        # Default to all layers if none specified
        if target_layers is None:
            target_layers = [CacheLayer.L1_MEMORY, CacheLayer.L2_REDIS, CacheLayer.L3_PERSISTENT]

        success_count = 0

        # Set in L1 (memory cache)
        if CacheLayer.L1_MEMORY in target_layers:
            await self._clean_expired_l1_entries()

            # Check if we need to evict items due to size constraints
            value_size = self._calculate_size(value)
            while (self._l1_current_size + value_size > self._l1_max_bytes or 
                   len(self._l1_cache) >= self._l1_max_size) and self._l1_cache:
                # Simple eviction: remove oldest item
                oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k].created_at)
                old_entry = self._l1_cache.pop(oldest_key)
                self._l1_current_size -= old_entry.size_bytes or 0

            # Calculate expiration time
            expires_at = datetime.now() + ttl if ttl else None

            # Create and store the entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                layer=CacheLayer.L1_MEMORY,
                size_bytes=value_size
            )

            old_entry = self._l1_cache.get(key)
            if old_entry:
                self._l1_current_size -= old_entry.size_bytes or 0

            self._l1_cache[key] = entry
            self._l1_current_size += value_size
            success_count += 1

        # Set in L2 (Redis cache)
        if CacheLayer.L2_REDIS in target_layers and self._l2_cache:
            try:
                serialized_value = pickle.dumps(value)
                expire_seconds = int(ttl.total_seconds()) if ttl else None
                await self._l2_cache.set(key, serialized_value, ex=expire_seconds)
                success_count += 1
            except Exception:
                pass  # Continue to next layer

        # Set in L3 (Persistent cache)
        if CacheLayer.L3_PERSISTENT in target_layers and self._l3_cache:
            try:
                # In a real implementation, this would store to the persistent store
                # For now, we'll simulate with a placeholder
                if hasattr(self._l3_cache, 'set'):
                    self._l3_cache.set(key, value, ttl)
                    success_count += 1
            except Exception:
                pass  # Continue to next step

        return success_count > 0

    async def delete(self, key: str) -> bool:
        """Delete a value from all cache layers."""
        success_count = 0

        # Delete from L1
        if key in self._l1_cache:
            entry = self._l1_cache.pop(key)
            self._l1_current_size -= entry.size_bytes or 0
            success_count += 1

        # Delete from L2
        if self._l2_cache:
            try:
                result = await self._l2_cache.delete(key)
                if result > 0:
                    success_count += 1
            except Exception:
                pass

        # Delete from L3
        if self._l3_cache and hasattr(self._l3_cache, 'delete'):
            try:
                self._l3_cache.delete(key)
                success_count += 1
            except Exception:
                pass

        return success_count > 0

    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all keys with a given prefix across all layers."""
        invalidated_count = 0

        # Invalidate in L1
        keys_to_remove = [key for key in self._l1_cache.keys() if key.startswith(prefix)]
        for key in keys_to_remove:
            entry = self._l1_cache.pop(key)
            self._l1_current_size -= entry.size_bytes or 0
            invalidated_count += 1

        # Invalidate in L2
        if self._l2_cache:
            try:
                # Get all keys matching the prefix
                pattern = f"{prefix}*"
                keys = await self._l2_cache.keys(pattern)
                if keys:
                    result = await self._l2_cache.delete(*keys)
                    invalidated_count += result
            except Exception:
                pass

        # Invalidate in L3
        if self._l3_cache and hasattr(self._l3_cache, 'invalidate_by_prefix'):
            try:
                count = self._l3_cache.invalidate_by_prefix(prefix)
                invalidated_count += count
            except Exception:
                pass

        return invalidated_count

    async def get_stats(self) -> Dict[CacheLayer, Dict[str, Any]]:
        """Get statistics for all cache layers."""
        stats = {}

        # L1 stats
        await self._clean_expired_l1_entries()
        l1_size = len(self._l1_cache)
        l1_memory_usage = self._l1_current_size
        stats[CacheLayer.L1_MEMORY] = {
            "size": l1_size,
            "memory_usage_bytes": l1_memory_usage,
            "hit_count": self._hits[CacheLayer.L1_MEMORY],
            "miss_count": self._misses[CacheLayer.L1_MEMORY],
            "hit_rate": self._hits[CacheLayer.L1_MEMORY] / max(1, self._hits[CacheLayer.L1_MEMORY] + self._misses[CacheLayer.L1_MEMORY]),
            "max_size_items": self._l1_max_size,
            "max_size_bytes": self._l1_max_bytes
        }

        # L2 stats (Redis)
        if self._l2_cache:
            try:
                info = await self._l2_cache.info()
                stats[CacheLayer.L2_REDIS] = {
                    "size": info.get("db0", {}).get("keys", 0) if "db0" in info else 0,
                    "memory_usage_bytes": info.get("used_memory", 0),
                    "hit_count": self._hits[CacheLayer.L2_REDIS],
                    "miss_count": self._misses[CacheLayer.L2_REDIS],
                    "hit_rate": self._hits[CacheLayer.L2_REDIS] / max(1, self._hits[CacheLayer.L2_REDIS] + self._misses[CacheLayer.L2_REDIS]),
                    "connected": True
                }
            except Exception as e:
                stats[CacheLayer.L2_REDIS] = {
                    "error": str(e),
                    "connected": False
                }
        else:
            stats[CacheLayer.L2_REDIS] = {"available": False}

        # L3 stats (Persistent)
        if self._l3_cache:
            stats[CacheLayer.L3_PERSISTENT] = {
                "hit_count": self._hits[CacheLayer.L3_PERSISTENT],
                "miss_count": self._misses[CacheLayer.L3_PERSISTENT],
                "hit_rate": self._hits[CacheLayer.L3_PERSISTENT] / max(1, self._hits[CacheLayer.L3_PERSISTENT] + self._misses[CacheLayer.L3_PERSISTENT]),
                "available": True
            }
        else:
            stats[CacheLayer.L3_PERSISTENT] = {"available": False}

        return stats

    async def warm_up(self, data: Dict[str, Any], ttl: Optional[timedelta] = None) -> bool:
        """Warm up the cache with a batch of data."""
        success_count = 0

        for key, value in data.items():
            if await self.set(key, value, ttl):
                success_count += 1

        return success_count == len(data)

    async def clear_layer(self, layer: CacheLayer) -> bool:
        """Clear all entries in a specific cache layer."""
        if layer == CacheLayer.L1_MEMORY:
            self._l1_cache.clear()
            self._l1_current_size = 0
            return True

        elif layer == CacheLayer.L2_REDIS and self._l2_cache:
            try:
                await self._l2_cache.flushdb()
                return True
            except Exception:
                return False

        elif layer == CacheLayer.L3_PERSISTENT and self._l3_cache:
            try:
                if hasattr(self._l3_cache, 'clear'):
                    self._l3_cache.clear()
                    return True
                return False
            except Exception:
                return False

        return False