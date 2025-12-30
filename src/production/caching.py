"""
Caching Module

This module implements various caching strategies for ML models in production,
including in-memory caching, Redis-like caching, and model-specific caching.
"""

import time
import hashlib
import pickle
import json
from typing import Any, Dict, Optional, Union, Callable, List
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio
from abc import ABC, abstractmethod


class CacheType(Enum):
    """Enumeration of cache types."""
    LRU = "lru"
    TTL = "ttl"
    LFU = "lfu"
    FIFO = "fifo"


@dataclass
class CacheItem:
    """Represents an item in the cache."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    ttl: Optional[float] = None  # Time-to-live in seconds


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set a value in the cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Delete a value from the cache."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all values from the cache."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the number of items in the cache."""
        pass


class LRUCache(BaseCache):
    """Least Recently Used (LRU) cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Move to end (most recently used)
            item = self.cache.pop(key)
            self.cache[key] = item
            
            # Update access count
            item.access_count += 1
            
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set a value in the cache."""
        with self.lock:
            # Check if key already exists
            if key in self.cache:
                # Update existing item
                item = self.cache.pop(key)
                item.value = value
                item.timestamp = time.time()
                item.ttl = ttl
            else:
                # Create new item
                item = CacheItem(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl
                )
            
            # Add to cache
            self.cache[key] = item
            
            # Remove oldest if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def delete(self, key: str):
        """Delete a value from the cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self):
        """Clear all values from the cache."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get the number of items in the cache."""
        with self.lock:
            return len(self.cache)
    
    def cleanup_expired(self):
        """Remove expired items from the cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, item in self.cache.items()
                if item.ttl is not None and (current_time - item.timestamp) > item.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]


class TTLCache(BaseCache):
    """Time-To-Live (TTL) cache implementation."""
    
    def __init__(self, default_ttl: float = 3600):  # 1 hour default
        """
        Initialize TTL cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheItem] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            item = self.cache[key]
            
            # Check if item is expired
            current_time = time.time()
            if item.ttl is not None and (current_time - item.timestamp) > item.ttl:
                del self.cache[key]
                return None
            
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set a value in the cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        with self.lock:
            self.cache[key] = CacheItem(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
    
    def delete(self, key: str):
        """Delete a value from the cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
    
    def clear(self):
        """Clear all values from the cache."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get the number of items in the cache."""
        with self.lock:
            # Clean up expired items first
            self.cleanup_expired()
            return len(self.cache)
    
    def cleanup_expired(self):
        """Remove expired items from the cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, item in self.cache.items()
                if item.ttl is not None and (current_time - item.timestamp) > item.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]


class LFUCache(BaseCache):
    """Least Frequently Used (LFU) cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LFU cache.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache: Dict[str, CacheItem] = {}
        self.freq_buckets: Dict[int, OrderedDict[str, None]] = {}  # frequency -> ordered keys
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            item = self.cache[key]
            
            # Remove from current frequency bucket
            old_freq = item.access_count
            del self.freq_buckets[old_freq][key]
            if not self.freq_buckets[old_freq]:
                del self.freq_buckets[old_freq]
            
            # Increment access count
            item.access_count += 1
            item.timestamp = time.time()
            
            # Add to new frequency bucket
            new_freq = item.access_count
            if new_freq not in self.freq_buckets:
                self.freq_buckets[new_freq] = OrderedDict()
            self.freq_buckets[new_freq][key] = None
            
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set a value in the cache."""
        with self.lock:
            if key in self.cache:
                # Update existing item
                item = self.cache[key]
                item.value = value
                item.timestamp = time.time()
                item.ttl = ttl
            else:
                # Check if cache is full
                if len(self.cache) >= self.max_size and key not in self.cache:
                    # Remove least frequently used item
                    min_freq = min(self.freq_buckets.keys())
                    lfu_key = next(iter(self.freq_buckets[min_freq]))
                    
                    del self.cache[lfu_key]
                    del self.freq_buckets[min_freq][lfu_key]
                    if not self.freq_buckets[min_freq]:
                        del self.freq_buckets[min_freq]
                
                # Create new item
                item = CacheItem(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    access_count=0,
                    ttl=ttl
                )
                self.cache[key] = item
            
            # Add to frequency bucket (frequency 0 for new items)
            if 0 not in self.freq_buckets:
                self.freq_buckets[0] = OrderedDict()
            self.freq_buckets[0][key] = None
    
    def delete(self, key: str):
        """Delete a value from the cache."""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                freq = item.access_count
                
                del self.cache[key]
                del self.freq_buckets[freq][key]
                if not self.freq_buckets[freq]:
                    del self.freq_buckets[freq]
    
    def clear(self):
        """Clear all values from the cache."""
        with self.lock:
            self.cache.clear()
            self.freq_buckets.clear()
    
    def size(self) -> int:
        """Get the number of items in the cache."""
        with self.lock:
            return len(self.cache)


class ModelCache:
    """Cache specifically designed for ML model predictions."""
    
    def __init__(self, cache_type: CacheType = CacheType.LRU, max_size: int = 10000):
        """
        Initialize model cache.
        
        Args:
            cache_type: Type of cache to use
            max_size: Maximum number of items to store
        """
        self.cache_type = cache_type
        
        if cache_type == CacheType.LRU:
            self.cache = LRUCache(max_size)
        elif cache_type == CacheType.TTL:
            self.cache = TTLCache(max_size)
        elif cache_type == CacheType.LFU:
            self.cache = LFUCache(max_size)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        self.stats_lock = threading.Lock()
    
    def _generate_key(self, model_name: str, input_data: Any) -> str:
        """Generate a unique key for caching model predictions."""
        # Create a hash of the input data to use as cache key
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        key_str = f"{model_name}:{input_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_prediction(self, model_name: str, input_data: Any) -> Optional[Any]:
        """Get a cached prediction."""
        key = self._generate_key(model_name, input_data)
        result = self.cache.get(key)
        
        with self.stats_lock:
            if result is not None:
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
        
        return result
    
    def cache_prediction(self, model_name: str, input_data: Any, prediction: Any, ttl: Optional[float] = None):
        """Cache a model prediction."""
        key = self._generate_key(model_name, input_data)
        self.cache.set(key, prediction, ttl)
    
    def invalidate_model_cache(self, model_name: str):
        """Invalidate all cache entries for a specific model."""
        # This is a simplified implementation
        # In a real system, you'd need to track which keys belong to which model
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        stats["size"] = self.cache.size()
        if (stats["hits"] + stats["misses"]) > 0:
            stats["hit_rate"] = stats["hits"] / (stats["hits"] + stats["misses"])
        else:
            stats["hit_rate"] = 0.0
        
        return stats


class CacheManager:
    """Manages multiple caches and provides high-level caching operations."""
    
    def __init__(self):
        self.caches: Dict[str, BaseCache] = {}
        self.default_cache: Optional[str] = None
        self.model_cache = ModelCache()
    
    def create_cache(self, name: str, cache_type: CacheType, **kwargs) -> BaseCache:
        """Create a new cache."""
        if cache_type == CacheType.LRU:
            cache = LRUCache(**kwargs)
        elif cache_type == CacheType.TTL:
            cache = TTLCache(**kwargs)
        elif cache_type == CacheType.LFU:
            cache = LFUCache(**kwargs)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")
        
        self.caches[name] = cache
        
        # Set as default if this is the first cache
        if self.default_cache is None:
            self.default_cache = name
        
        return cache
    
    def get_cache(self, name: str = None) -> BaseCache:
        """Get a cache by name or return default."""
        if name is None:
            name = self.default_cache
        
        if name not in self.caches:
            raise ValueError(f"Cache {name} does not exist")
        
        return self.caches[name]
    
    def get_model_prediction(self, model_name: str, input_data: Any) -> Optional[Any]:
        """Get a cached model prediction."""
        return self.model_cache.get_prediction(model_name, input_data)
    
    def cache_model_prediction(self, model_name: str, input_data: Any, prediction: Any, ttl: Optional[float] = None):
        """Cache a model prediction."""
        self.model_cache.cache_prediction(model_name, input_data, prediction, ttl)
    
    def invalidate_cache(self, name: str = None):
        """Invalidate a cache."""
        cache = self.get_cache(name)
        cache.clear()
    
    def get_cache_stats(self, name: str = None) -> Dict[str, Any]:
        """Get cache statistics."""
        if name is None and self.default_cache:
            cache = self.caches[self.default_cache]
            return {"size": cache.size()}
        elif name in self.caches:
            cache = self.caches[name]
            return {"size": cache.size()}
        else:
            return self.model_cache.get_stats()


class AsyncCache:
    """Asynchronous cache implementation."""
    
    def __init__(self, cache_type: CacheType = CacheType.LRU, max_size: int = 10000):
        """
        Initialize async cache.
        
        Args:
            cache_type: Type of cache to use
            max_size: Maximum number of items to store
        """
        self.cache_type = cache_type
        
        if cache_type == CacheType.LRU:
            self.cache = LRUCache(max_size)
        elif cache_type == CacheType.TTL:
            self.cache = TTLCache(max_size)
        elif cache_type == CacheType.LFU:
            self.cache = LFUCache(max_size)
        else:
            raise ValueError(f"Unsupported cache type: {cache_type}")
        
        self.lock = asyncio.Lock()
        self.stats = {"hits": 0, "misses": 0}
    
    async def get(self, key: str) -> Optional[Any]:
        """Asynchronously get a value from the cache."""
        async with self.lock:
            value = self.cache.get(key)
            if value is not None:
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Asynchronously set a value in the cache."""
        async with self.lock:
            self.cache.set(key, value, ttl)
    
    async def batch_get(self, keys: List[str]) -> List[Optional[Any]]:
        """Asynchronously get multiple values from the cache."""
        async with self.lock:
            results = []
            for key in keys:
                value = self.cache.get(key)
                if value is not None:
                    self.stats["hits"] += 1
                else:
                    self.stats["misses"] += 1
                results.append(value)
            return results
    
    async def batch_set(self, items: List[Tuple[str, Any]], ttl: Optional[float] = None):
        """Asynchronously set multiple values in the cache."""
        async with self.lock:
            for key, value in items:
                self.cache.set(key, value, ttl)


# Global cache manager instance
cache_manager = CacheManager()


def get_global_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return cache_manager


def cache_function_result(ttl: Optional[float] = None, cache_name: str = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live for cached results
        cache_name: Name of the cache to use
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create a key based on function name and arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            key = hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()
            
            # Try to get from cache
            cache = cache_manager.get_cache(cache_name)
            result = cache.get(key)
            
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_model_predict(model_name: str, ttl: Optional[float] = 300):  # 5 minutes default
    """
    Decorator to cache model predictions.
    
    Args:
        model_name: Name of the model
        ttl: Time-to-live for cached predictions
    """
    def decorator(predict_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Use the input data as cache key
            input_data = args[0] if args else kwargs.get('input_data', {})
            
            # Try to get from model cache
            result = cache_manager.get_model_prediction(model_name, input_data)
            
            if result is not None:
                return result
            
            # Execute prediction and cache result
            result = predict_func(*args, **kwargs)
            cache_manager.cache_model_prediction(model_name, input_data, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Initialize default caches
def initialize_caches():
    """Initialize default caches."""
    cache_manager.create_cache("default_lru", CacheType.LRU, max_size=1000)
    cache_manager.create_cache("default_ttl", CacheType.TTL, default_ttl=3600)


# Initialize caches when module is loaded
initialize_caches()