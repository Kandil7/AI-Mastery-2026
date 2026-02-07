"""
Cache Port
===========
Interface for caching operations.

منفذ التخزين المؤقت
"""

from typing import Any, Protocol


class CachePort(Protocol):
    """
    Port for caching operations (embeddings, answers, etc.).
    
    Implementation: Redis
    
    Design Decision: JSON-based cache for flexibility.
    Embeddings are expensive, so caching is critical for cost control.
    
    قرار التصميم: تخزين مؤقت JSON للمرونة
    """
    
    def get_json(self, key: str) -> dict[str, Any] | None:
        """
        Get cached JSON value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        ...
    
    def set_json(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: int,
    ) -> None:
        """
        Set JSON value with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl_seconds: Time-to-live in seconds
        """
        ...
    
    def delete(self, key: str) -> bool:
        """
        Delete a cached value.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists and not expired
        """
        ...
