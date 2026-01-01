"""
Unit Tests for Retrieval Module

Tests for BM25Retriever, DenseRetriever, HybridRetriever, and utility functions.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.retrieval import (
    Document,
    RetrievalResult,
    RetrievalConfig,
    TextPreprocessor,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalPipeline,
)


class TestDocument(unittest.TestCase):
    """Tests for Document dataclass."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="doc1",
            content="Test content",
            metadata={"source": "test"}
        )
        
        self.assertEqual(doc.id, "doc1")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.metadata["source"], "test")
    
    def test_document_auto_id(self):
        """Test automatic ID generation when not provided."""
        doc = Document(id="", content="Some content")
        
        self.assertIsNotNone(doc.id)
        self.assertNotEqual(doc.id, "")


class TestTextPreprocessor(unittest.TestCase):
    """Tests for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokens = self.preprocessor.tokenize("Hello World")
        
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
    
    def test_stopword_removal(self):
        """Test stopword removal."""
        tokens = self.preprocessor.tokenize("The quick brown fox")
        
        self.assertNotIn("the", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
    
    def test_lowercase(self):
        """Test lowercasing."""
        tokens = self.preprocessor.tokenize("UPPERCASE Words")
        
        self.assertIn("uppercase", tokens)
        self.assertIn("words", tokens)
    
    def test_min_token_length(self):
        """Test minimum token length filtering."""
        preprocessor = TextPreprocessor(min_token_length=3)
        tokens = preprocessor.tokenize("I am a test")
        
        self.assertNotIn("i", tokens)
        self.assertNotIn("am", tokens)
        self.assertIn("test", tokens)


class TestBM25Retriever(unittest.TestCase):
    """Tests for BM25Retriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.documents = [
            Document(id="1", content="Machine learning is a subset of artificial intelligence."),
            Document(id="2", content="Deep learning uses neural networks."),
            Document(id="3", content="Natural language processing for text."),
            Document(id="4", content="Computer vision for image analysis."),
        ]
        
        self.retriever = BM25Retriever()
        self.retriever.index(self.documents)
    
    def test_index_creates_vocabulary(self):
        """Test that indexing creates a vocabulary."""
        self.assertTrue(self.retriever.is_indexed)
        self.assertGreater(len(self.retriever.doc_freqs), 0)
    
    def test_retrieve_returns_results(self):
        """Test that retrieval returns results."""
        results = self.retriever.retrieve("machine learning", top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)
    
    def test_retrieve_relevant_first(self):
        """Test that relevant documents rank higher."""
        results = self.retriever.retrieve("neural networks", top_k=4)
        
        # Document about neural networks should be in top results
        top_ids = [r.document.id for r in results[:2]]
        self.assertIn("2", top_ids)
    
    def test_explain_score(self):
        """Test score explanation."""
        contributions = self.retriever.explain_score("machine learning", 0)
        
        self.assertIn("machine", contributions)
        self.assertIn("learning", contributions)
    
    def test_empty_query(self):
        """Test handling of empty query."""
        results = self.retriever.retrieve("", top_k=2)
        
        self.assertEqual(len(results), 0)
    
    def test_query_with_no_matches(self):
        """Test query with no matching terms."""
        results = self.retriever.retrieve("xyz123 unknown terms", top_k=2)
        
        # Should return empty or low-scored results
        self.assertIsInstance(results, list)


class TestDenseRetriever(unittest.TestCase):
    """Tests for DenseRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.documents = [
            Document(id="1", content="Machine learning algorithms"),
            Document(id="2", content="Deep neural networks"),
            Document(id="3", content="Cooking recipes"),
        ]
        
        self.retriever = DenseRetriever(use_faiss=False)
        self.retriever.index(self.documents)
    
    def test_index_creates_embeddings(self):
        """Test that indexing creates embeddings."""
        self.assertTrue(self.retriever.is_indexed)
        self.assertIsNotNone(self.retriever.embeddings)
        self.assertEqual(self.retriever.embeddings.shape[0], 3)
    
    def test_retrieve_returns_results(self):
        """Test retrieval returns results."""
        results = self.retriever.retrieve("AI and machine learning", top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievalResult)
    
    def test_retrieve_scores_are_valid(self):
        """Test that scores are valid floats."""
        results = self.retriever.retrieve("neural networks", top_k=2)
        
        for r in results:
            self.assertIsInstance(r.score, float)


class TestHybridRetriever(unittest.TestCase):
    """Tests for HybridRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.documents = [
            Document(id="1", content="Machine learning is AI"),
            Document(id="2", content="Deep learning neural networks"),
            Document(id="3", content="NLP for text processing"),
        ]
        
        self.retriever = HybridRetriever(alpha=0.5, fusion="rrf")
        self.retriever.index(self.documents)
    
    def test_index_both_retrievers(self):
        """Test that both retrievers are indexed."""
        self.assertTrue(self.retriever.sparse_retriever.is_indexed)
        self.assertTrue(self.retriever.dense_retriever.is_indexed)
    
    def test_retrieve_hybrid_results(self):
        """Test hybrid retrieval."""
        results = self.retriever.retrieve("machine learning AI", top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].retriever, "hybrid")
    
    def test_rrf_fusion(self):
        """Test RRF fusion produces valid scores."""
        results = self.retriever.retrieve("neural networks", top_k=3)
        
        for r in results:
            self.assertGreater(r.score, 0)


class TestRetrievalPipeline(unittest.TestCase):
    """Tests for RetrievalPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.documents = [
            Document(id="1", content="Machine learning algorithms"),
            Document(id="2", content="Neural network architectures"),
        ]
        
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(use_faiss=False)
    
    def test_pipeline_single_stage(self):
        """Test pipeline with single stage."""
        pipeline = RetrievalPipeline()
        pipeline.add_retriever(self.bm25, stage="initial")
        pipeline.index(self.documents)
        
        results = pipeline.retrieve("machine learning", top_k=2)
        
        self.assertEqual(len(results), 2)
    
    def test_pipeline_multiple_stages(self):
        """Test pipeline with multiple stages."""
        pipeline = RetrievalPipeline()
        pipeline.add_retriever(self.bm25, stage="initial")
        pipeline.add_retriever(self.dense, stage="rerank")
        pipeline.index(self.documents)
        
        results = pipeline.retrieve("neural networks", top_k=2)
        
        self.assertLessEqual(len(results), 2)


class TestRetrievalConfig(unittest.TestCase):
    """Tests for RetrievalConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RetrievalConfig()
        
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.min_score, 0.0)
        self.assertFalse(config.use_reranking)
        self.assertEqual(config.hybrid_alpha, 0.5)


if __name__ == "__main__":
    unittest.main()
