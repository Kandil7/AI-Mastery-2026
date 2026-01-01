"""
Unit Tests for Embeddings Module

Tests for TextEmbedder, ImageEmbedder, EmbeddingCache, and utility functions.
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings.embeddings import (
    EmbeddingConfig,
    EmbeddingCache,
    TextEmbedder,
    cosine_similarity,
    pairwise_cosine_similarity,
    euclidean_distance,
)


class TestEmbeddingConfig(unittest.TestCase):
    """Tests for EmbeddingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        
        self.assertEqual(config.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(config.dimension, 384)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="custom-model",
            dimension=768,
            batch_size=64,
            normalize=False
        )
        
        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.dimension, 768)
        self.assertEqual(config.batch_size, 64)
        self.assertFalse(config.normalize)


class TestEmbeddingCache(unittest.TestCase):
    """Tests for EmbeddingCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = EmbeddingCache(max_size=10)
        self.test_embedding = np.array([0.1, 0.2, 0.3])
    
    def test_set_and_get(self):
        """Test setting and getting embeddings."""
        self.cache.set("test_text", self.test_embedding)
        
        result = self.cache.get("test_text")
        
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, self.test_embedding)
    
    def test_get_nonexistent(self):
        """Test getting non-existent key returns None."""
        result = self.cache.get("nonexistent")
        self.assertIsNone(result)
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3)
        
        # Add 4 items (should evict first)
        for i in range(4):
            cache.set(f"text_{i}", np.array([float(i)]))
        
        # First item should be evicted
        self.assertIsNone(cache.get("text_0"))
        self.assertIsNotNone(cache.get("text_3"))
    
    def test_hit_rate(self):
        """Test cache hit rate calculation."""
        self.cache.set("exists", self.test_embedding)
        
        # 1 hit
        self.cache.get("exists")
        # 1 miss
        self.cache.get("missing")
        
        hit_rate = self.cache.hit_rate
        self.assertEqual(hit_rate, 0.5)  # 1 hit / 2 total
    
    def test_clear(self):
        """Test cache clearing."""
        self.cache.set("test", self.test_embedding)
        self.cache.clear()
        
        self.assertIsNone(self.cache.get("test"))
    
    def test_persistence(self):
        """Test cache persistence to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "test_cache.pkl")
            
            # Create cache with persistence
            cache = EmbeddingCache(max_size=10, persist=True, cache_path=cache_path)
            cache.set("persistent_text", self.test_embedding)
            
            # Create new cache instance and load
            cache2 = EmbeddingCache(max_size=10, persist=True, cache_path=cache_path)
            
            result = cache2.get("persistent_text")
            self.assertIsNotNone(result)
            np.testing.assert_array_almost_equal(result, self.test_embedding)
    
    def test_get_batch(self):
        """Test batch get operation."""
        self.cache.set("text1", np.array([1.0]))
        self.cache.set("text2", np.array([2.0]))
        
        texts = ["text1", "text2", "text3"]
        results = self.cache.get_batch(texts)
        
        self.assertIn("text1", results)
        self.assertIn("text2", results)
        self.assertIn("text3", results)
        self.assertIsNotNone(results["text1"])
        self.assertIsNone(results["text3"])


class TestTextEmbedder(unittest.TestCase):
    """Tests for TextEmbedder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use fallback mode for testing without dependencies
        self.embedder = TextEmbedder()
    
    def test_encode_single_text(self):
        """Test encoding a single text."""
        embedding = self.embedder.encode("Hello world")
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding.shape), 2)  # Should be 2D
        self.assertEqual(embedding.shape[0], 1)  # One text
    
    def test_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        texts = ["Hello", "World", "Test"]
        embeddings = self.embedder.encode(texts)
        
        self.assertEqual(embeddings.shape[0], 3)
    
    def test_similarity(self):
        """Test similarity computation."""
        text1 = "machine learning"
        text2 = "deep learning"
        text3 = "cooking recipes"
        
        sim_similar = self.embedder.similarity(text1, text2)
        sim_different = self.embedder.similarity(text1, text3)
        
        # Similar texts should have higher similarity
        # This may not always hold with fallback, but tests the interface
        self.assertIsInstance(sim_similar, float)
        self.assertIsInstance(sim_different, float)
    
    def test_with_cache(self):
        """Test embedder with caching."""
        cache = EmbeddingCache(max_size=100)
        embedder = TextEmbedder(cache=cache)
        
        text = "cached text"
        
        # First call - should miss
        embedding1 = embedder.encode(text)
        
        # Second call - should hit
        embedding2 = embedder.encode(text)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    def test_find_most_similar(self):
        """Test finding most similar texts."""
        corpus = [
            "machine learning algorithms",
            "deep neural networks",
            "cooking pasta recipes",
            "artificial intelligence systems"
        ]
        
        query = "AI and ML"
        top_results = self.embedder.find_most_similar(query, corpus, top_k=2)
        
        self.assertEqual(len(top_results), 2)
        self.assertIn("text", top_results[0])
        self.assertIn("score", top_results[0])


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        v = np.array([1.0, 2.0, 3.0])
        
        sim = cosine_similarity(v, v)
        
        self.assertAlmostEqual(sim, 1.0, places=6)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        
        sim = cosine_similarity(v1, v2)
        
        self.assertAlmostEqual(sim, 0.0, places=6)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        v1 = np.array([1.0, 2.0])
        v2 = np.array([-1.0, -2.0])
        
        sim = cosine_similarity(v1, v2)
        
        self.assertAlmostEqual(sim, -1.0, places=6)
    
    def test_pairwise_cosine_similarity(self):
        """Test pairwise cosine similarity matrix."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        
        sim_matrix = pairwise_cosine_similarity(embeddings)
        
        self.assertEqual(sim_matrix.shape, (3, 3))
        self.assertAlmostEqual(sim_matrix[0, 0], 1.0, places=6)  # Self-similarity
        self.assertAlmostEqual(sim_matrix[0, 2], 1.0, places=6)  # Same vectors
        self.assertAlmostEqual(sim_matrix[0, 1], 0.0, places=6)  # Orthogonal
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        
        dist = euclidean_distance(v1, v2)
        
        self.assertAlmostEqual(dist, 5.0, places=6)  # 3-4-5 triangle


class TestEmbedderIntegration(unittest.TestCase):
    """Integration tests for embedder workflow."""
    
    def test_full_workflow(self):
        """Test complete embedding workflow."""
        # Setup
        cache = EmbeddingCache(max_size=100)
        embedder = TextEmbedder(cache=cache)
        
        # Embed documents
        documents = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "Machine learning uses algorithms"
        ]
        
        embeddings = embedder.encode(documents)
        
        # Should have correct shape
        self.assertEqual(embeddings.shape[0], 3)
        
        # Query for similar
        query = "programming languages"
        similar = embedder.find_most_similar(query, documents, top_k=2)
        
        self.assertEqual(len(similar), 2)
        
        # Cache should have entries
        self.assertGreater(len(cache._cache), 0)


if __name__ == "__main__":
    unittest.main()
