"""
Cached Embeddings Service
==========================
Wrapper that adds Redis caching to embeddings provider.

خدمة التضمينات المخزنة مؤقتاً
"""

import hashlib
from typing import Any

from src.application.ports.cache import CachePort
from src.application.ports.embeddings import EmbeddingsPort


class CachedEmbeddings:
    """
    Wraps an embeddings provider with Redis caching.
    
    Design Decision: Cache-first approach for embeddings because:
    - Embeddings are deterministic (same input = same output)
    - API calls are expensive ($)
    - Latency reduction for repeated queries
    - 7-day TTL balances freshness and cache hits
    
    قرار التصميم: نهج التخزين المؤقت أولاً للتضمينات لتقليل التكلفة
    
    Example:
        >>> cached = CachedEmbeddings(embeddings=openai_emb, cache=redis_cache)
        >>> vector = cached.embed_one("Hello world")  # Cache miss, calls API
        >>> vector = cached.embed_one("Hello world")  # Cache hit, instant
    """
    
    def __init__(
        self,
        embeddings: EmbeddingsPort,
        cache: CachePort,
        ttl_seconds: int = 604800,  # 7 days default
        key_prefix: str = "emb:",
    ) -> None:
        """
        Initialize cached embeddings.
        
        Args:
            embeddings: Underlying embeddings provider
            cache: Cache implementation (Redis)
            ttl_seconds: Cache TTL (default 7 days)
            key_prefix: Cache key prefix
        """
        self._embeddings = embeddings
        self._cache = cache
        self._ttl = ttl_seconds
        self._prefix = key_prefix
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key from text hash."""
        # Use MD5 for speed (not security-critical here)
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"{self._prefix}{text_hash}"
    
    def embed_one(self, text: str) -> list[float]:
        """
        Get embedding for single text with caching.
        
        The cache stores {"v": [vector]} format.
        """
        key = self._cache_key(text)
        
        # Check cache
        cached: dict[str, Any] | None = self._cache.get_json(key)
        if cached and "v" in cached:
            return cached["v"]
        
        # Cache miss - call provider
        vector = self._embeddings.embed_one(text)
        
        # Store in cache
        self._cache.set_json(key, {"v": vector}, ttl_seconds=self._ttl)
        
        return vector
    
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts with caching.
        
        Optimized to:
        1. Check cache for all texts first
        2. Batch embed only missing texts
        3. Update cache for newly embedded texts
        4. Return in original order
        
        محسّن للتحقق من التخزين المؤقت أولاً ثم تضمين النصوص المفقودة فقط
        """
        n = len(texts)
        if n == 0:
            return []
        
        # Step 1: Check cache for all
        keys = [self._cache_key(t) for t in texts]
        cached_values = [self._cache.get_json(k) for k in keys]
        
        # Step 2: Find missing indices
        missing_indices: list[int] = []
        for i, cached in enumerate(cached_values):
            if not (cached and "v" in cached):
                missing_indices.append(i)
        
        # If all cached, return immediately
        if not missing_indices:
            return [cv["v"] for cv in cached_values]  # type: ignore
        
        # Step 3: Batch embed missing
        missing_texts = [texts[i] for i in missing_indices]
        new_vectors = self._embeddings.embed_many(missing_texts)
        
        # Step 4: Store new vectors in cache
        for idx, vec in zip(missing_indices, new_vectors):
            self._cache.set_json(keys[idx], {"v": vec}, ttl_seconds=self._ttl)
        
        # Step 5: Build final result in order
        result: list[list[float]] = []
        new_vec_map = {idx: vec for idx, vec in zip(missing_indices, new_vectors)}
        
        for i, cached in enumerate(cached_values):
            if cached and "v" in cached:
                result.append(cached["v"])
            else:
                result.append(new_vec_map[i])
        
        return result
