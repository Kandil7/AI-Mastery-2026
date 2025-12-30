"""
Unit tests for Production Caching Module
=========================================
Tests for LRU cache, embedding cache, and cache decorators.
"""

import pytest
import time
from src.production.caching import (
    LRUCache, EmbeddingCache, PredictionCache, TieredCache,
    cached, CacheStats
)


class TestLRUCache:
    """Tests for LRU Cache implementation."""
    
    def test_basic_set_get(self):
        """Basic set and get operations."""
        cache = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_miss_returns_none(self):
        """Cache miss should return None."""
        cache = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None
    
    def test_eviction_on_capacity(self):
        """LRU item should be evicted when at capacity."""
        cache = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a"
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4
    
    def test_access_updates_recency(self):
        """Accessing item should move it to most recently used."""
        cache = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        # Access "a" to make it most recent
        cache.get("a")
        
        # Add new item, should evict "b" (now LRU)
        cache.set("d", 4)
        
        assert cache.get("a") == 1
        assert cache.get("b") is None
    
    def test_size(self):
        """Size should reflect number of items."""
        cache = LRUCache(max_size=10)
        assert cache.size() == 0
        
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size() == 2
    
    def test_delete(self):
        """Delete should remove item."""
        cache = LRUCache(max_size=10)
        cache.set("key", "value")
        cache.delete("key")
        assert cache.get("key") is None
    
    def test_clear(self):
        """Clear should remove all items."""
        cache = LRUCache(max_size=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size() == 0
    
    def test_exists(self):
        """Exists should check key presence."""
        cache = LRUCache(max_size=10)
        cache.set("key", "value")
        assert cache.exists("key") is True
        assert cache.exists("other") is False
    
    def test_ttl_expiration(self):
        """Items should expire after TTL."""
        cache = LRUCache(max_size=10, default_ttl=1)
        cache.set("key", "value")
        
        assert cache.get("key") == "value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        assert cache.get("key") is None
    
    def test_stats_tracking(self):
        """Stats should track hits and misses."""
        cache = LRUCache(max_size=10)
        cache.set("key", "value")
        
        cache.get("key")  # Hit
        cache.get("key")  # Hit
        cache.get("miss")  # Miss
        
        stats = cache.stats
        assert stats.hits == 2
        assert stats.misses == 1


class TestEmbeddingCache:
    """Tests for Embedding Cache."""
    
    def test_content_hashing(self):
        """Same content should produce same key."""
        cache = EmbeddingCache()
        cache.set("hello world", [0.1, 0.2, 0.3])
        assert cache.get("hello world") == [0.1, 0.2, 0.3]
    
    def test_different_content_different_key(self):
        """Different content should not collide."""
        cache = EmbeddingCache()
        cache.set("text1", [0.1, 0.2])
        cache.set("text2", [0.3, 0.4])
        
        assert cache.get("text1") == [0.1, 0.2]
        assert cache.get("text2") == [0.3, 0.4]
    
    def test_batch_operations(self):
        """Batch get and set should work."""
        cache = EmbeddingCache()
        texts = ["a", "b", "c"]
        embeddings = [[0.1], [0.2], [0.3]]
        
        count = cache.set_batch(texts, embeddings)
        assert count == 3
        
        results = cache.get_batch(texts)
        assert results["a"] == [0.1]
        assert results["b"] == [0.2]
    
    def test_get_or_compute(self):
        """get_or_compute should cache computed embeddings."""
        cache = EmbeddingCache()
        compute_calls = []
        
        def embed_fn(texts):
            compute_calls.extend(texts)
            return [[0.1] for _ in texts]
        
        # First call should compute
        result1 = cache.get_or_compute(["a", "b"], embed_fn)
        assert len(compute_calls) == 2
        
        # Second call should use cache
        result2 = cache.get_or_compute(["a", "c"], embed_fn)
        assert len(compute_calls) == 3  # Only "c" was computed
        assert "a" in compute_calls
        assert compute_calls.count("a") == 1  # "a" computed only once


class TestPredictionCache:
    """Tests for Prediction Cache."""
    
    def test_feature_hashing(self):
        """Same features should produce same cache key."""
        cache = PredictionCache(model_version="v1")
        features = {"x1": 1.0, "x2": 2.0}
        
        cache.set(features, 0.5)
        assert cache.get(features) == 0.5
    
    def test_different_features_different_key(self):
        """Different features should not collide."""
        cache = PredictionCache(model_version="v1")
        
        cache.set({"x": 1}, "pred1")
        cache.set({"x": 2}, "pred2")
        
        assert cache.get({"x": 1}) == "pred1"
        assert cache.get({"x": 2}) == "pred2"
    
    def test_model_version_namespace(self):
        """Different model versions should not share cache."""
        cache1 = PredictionCache(model_version="v1")
        cache2 = PredictionCache(model_version="v2")
        
        cache1.set({"x": 1}, "v1_result")
        
        # Different version shouldn't find it
        # (each cache has separate backend by default)
        assert cache2.get({"x": 1}) is None
    
    def test_get_or_predict(self):
        """get_or_predict should cache predictions."""
        cache = PredictionCache()
        predict_calls = []
        
        def predict_fn(features):
            predict_calls.append(features)
            return features["x"] * 2
        
        result1 = cache.get_or_predict({"x": 5}, predict_fn)
        assert result1 == 10
        assert len(predict_calls) == 1
        
        # Should use cache
        result2 = cache.get_or_predict({"x": 5}, predict_fn)
        assert result2 == 10
        assert len(predict_calls) == 1  # No additional call


class TestCachedDecorator:
    """Tests for @cached decorator."""
    
    def test_function_caching(self):
        """Decorated function should cache results."""
        cache = LRUCache(max_size=10)
        call_count = 0
        
        @cached(cache)
        def expensive_fn(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_fn(5)
        assert result1 == 10
        assert call_count == 1
        
        result2 = expensive_fn(5)
        assert result2 == 10
        assert call_count == 1  # Cached
        
        result3 = expensive_fn(10)
        assert result3 == 20
        assert call_count == 2  # New argument
    
    def test_custom_key_function(self):
        """Custom key function should be used."""
        cache = LRUCache(max_size=10)
        
        @cached(cache, key_fn=lambda x, y: f"{x}")
        def fn(x, y):
            return x + y
        
        result1 = fn(1, 100)
        result2 = fn(1, 200)  # Same key due to custom key_fn
        
        assert result2 == result1  # Returns cached value


class TestTieredCache:
    """Tests for Tiered (L1+L2) Cache."""
    
    def test_l1_hit(self):
        """L1 hit should not query L2."""
        l1 = LRUCache(max_size=10)
        l2 = LRUCache(max_size=100)
        tiered = TieredCache(l1, l2)
        
        l1.set("key", "l1_value")
        l2.set("key", "l2_value")
        
        # Should get L1 value
        assert tiered.get("key") == "l1_value"
    
    def test_l1_miss_l2_hit(self):
        """L1 miss should check L2 and populate L1."""
        l1 = LRUCache(max_size=10)
        l2 = LRUCache(max_size=100)
        tiered = TieredCache(l1, l2)
        
        l2.set("key", "l2_value")
        
        result = tiered.get("key")
        assert result == "l2_value"
        
        # L1 should now be populated
        assert l1.get("key") == "l2_value"
    
    def test_set_populates_both(self):
        """Set should populate both L1 and L2."""
        l1 = LRUCache(max_size=10)
        l2 = LRUCache(max_size=100)
        tiered = TieredCache(l1, l2)
        
        tiered.set("key", "value")
        
        assert l1.get("key") == "value"
        assert l2.get("key") == "value"


class TestCacheStats:
    """Tests for CacheStats."""
    
    def test_hit_rate_calculation(self):
        """Hit rate should be correctly calculated."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75
    
    def test_zero_operations(self):
        """Hit rate should be 0 with no operations."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
    
    def test_to_dict(self):
        """to_dict should include all fields."""
        stats = CacheStats(hits=10, misses=5, evictions=2)
        d = stats.to_dict()
        
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert "hit_rate" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
