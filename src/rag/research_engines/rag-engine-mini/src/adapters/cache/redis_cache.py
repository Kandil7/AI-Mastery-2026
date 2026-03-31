"""
Redis Cache Adapter
====================
Implementation of CachePort for Redis.

محول التخزين المؤقت Redis
"""

import json
from typing import Any

import redis


class RedisCache:
    """
    Redis adapter implementing CachePort.
    
    Stores JSON-serializable data with TTL.
    
    محول Redis للتخزين المؤقت JSON
    """
    
    def __init__(self, redis_url: str) -> None:
        """
        Initialize Redis client.
        
        Args:
            redis_url: Redis connection URL (redis://host:port/db)
        """
        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )
    
    def get_json(self, key: str) -> dict[str, Any] | None:
        """Get cached JSON value."""
        try:
            value = self._client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except (json.JSONDecodeError, redis.RedisError):
            return None
    
    def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        """Set JSON value with TTL."""
        try:
            self._client.setex(
                key,
                ttl_seconds,
                json.dumps(value),
            )
        except (TypeError, redis.RedisError):
            pass  # Silently fail on cache write errors
    
    def delete(self, key: str) -> bool:
        """Delete a cached value."""
        try:
            return bool(self._client.delete(key))
        except redis.RedisError:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self._client.exists(key))
        except redis.RedisError:
            return False
