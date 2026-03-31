"""
Embedding Cache Unit Tests
===========================
Tests for cached embeddings wrapper.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.application.services.embedding_cache import CachedEmbeddings


class MockCache:
    """Mock cache for testing."""
    
    def __init__(self):
        self._store: dict = {}
    
    def get_json(self, key: str) -> dict | None:
        return self._store.get(key)
    
    def set_json(self, key: str, value: dict, ttl_seconds: int) -> None:
        self._store[key] = value
    
    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False
    
    def exists(self, key: str) -> bool:
        return key in self._store


class MockEmbeddings:
    """Mock embeddings provider for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    def embed_one(self, text: str) -> list[float]:
        self.call_count += 1
        return [0.1, 0.2, 0.3]
    
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestCachedEmbeddings:
    """Tests for CachedEmbeddings class."""
    
    def test_cache_miss_calls_provider(self):
        """Cache miss should call underlying provider."""
        cache = MockCache()
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache)
        
        result = cached.embed_one("Hello")
        
        assert result == [0.1, 0.2, 0.3]
        assert embeddings.call_count == 1
    
    def test_cache_hit_skips_provider(self):
        """Cache hit should not call underlying provider."""
        cache = MockCache()
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache)
        
        # First call - cache miss
        cached.embed_one("Hello")
        assert embeddings.call_count == 1
        
        # Second call - cache hit
        result = cached.embed_one("Hello")
        assert result == [0.1, 0.2, 0.3]
        assert embeddings.call_count == 1  # Still 1, not 2
    
    def test_different_texts_different_keys(self):
        """Different texts should have different cache keys."""
        cache = MockCache()
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache)
        
        cached.embed_one("Hello")
        cached.embed_one("World")
        
        assert embeddings.call_count == 2
    
    def test_embed_many_partial_cache(self):
        """embed_many should only fetch missing texts."""
        cache = MockCache()
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache)
        
        # Pre-cache one text
        cached.embed_one("Hello")
        assert embeddings.call_count == 1
        
        # Request batch with one cached, one new
        results = cached.embed_many(["Hello", "World"])
        
        assert len(results) == 2
        assert embeddings.call_count == 2  # Only called for "World"
    
    def test_embed_many_all_cached(self):
        """embed_many with all cached should not call provider."""
        cache = MockCache()
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache)
        
        # Pre-cache all texts
        cached.embed_one("Hello")
        cached.embed_one("World")
        assert embeddings.call_count == 2
        
        # Request batch - all cached
        results = cached.embed_many(["Hello", "World"])
        
        assert len(results) == 2
        assert embeddings.call_count == 2  # No new calls
    
    def test_embed_many_empty_list(self):
        """embed_many with empty list should return empty list."""
        cache = MockCache()
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache)
        
        results = cached.embed_many([])
        
        assert results == []
        assert embeddings.call_count == 0
    
    def test_custom_ttl(self):
        """Custom TTL should be passed to cache."""
        cache = Mock()
        cache.get_json.return_value = None
        
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache, ttl_seconds=3600)
        
        cached.embed_one("Hello")
        
        cache.set_json.assert_called_once()
        call_args = cache.set_json.call_args
        assert call_args[1]["ttl_seconds"] == 3600
    
    def test_custom_key_prefix(self):
        """Custom key prefix should be used."""
        cache = Mock()
        cache.get_json.return_value = None
        
        embeddings = MockEmbeddings()
        cached = CachedEmbeddings(embeddings, cache, key_prefix="custom:")
        
        cached.embed_one("Hello")
        
        cache.get_json.assert_called_once()
        key = cache.get_json.call_args[0][0]
        assert key.startswith("custom:")
