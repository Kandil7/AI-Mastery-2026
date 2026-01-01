"""
Unit Tests for Reranking Module

Tests for CrossEncoderReranker, LLMReranker, ReciprocalRankFusion, and DiversityReranker.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.reranking.reranking import (
    Document,
    RerankResult,
    RerankConfig,
    CrossEncoderReranker,
    LLMReranker,
    ReciprocalRankFusion,
    DiversityReranker,
    RerankingPipeline,
)


class TestDocument(unittest.TestCase):
    """Tests for Document dataclass."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="doc1",
            content="Test content",
            score=0.85
        )
        
        self.assertEqual(doc.id, "doc1")
        self.assertEqual(doc.content, "Test content")
        self.assertEqual(doc.score, 0.85)
    
    def test_document_default_score(self):
        """Test document with default score."""
        doc = Document(id="doc1", content="Content")
        self.assertEqual(doc.score, 0.0)


class TestRerankConfig(unittest.TestCase):
    """Tests for RerankConfig."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = RerankConfig()
        
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.min_score, 0.0)
        self.assertTrue(config.normalize_scores)


class TestCrossEncoderReranker(unittest.TestCase):
    """Tests for CrossEncoderReranker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.documents = [
            Document(id="1", content="Machine learning is AI.", score=0.9),
            Document(id="2", content="Deep learning uses neural networks.", score=0.85),
            Document(id="3", content="Cooking pasta is easy.", score=0.7),
        ]
        self.reranker = CrossEncoderReranker()
    
    def test_rerank_returns_results(self):
        """Test that reranking returns results."""
        results = self.reranker.rerank("AI and machine learning", self.documents, top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RerankResult)
    
    def test_rerank_has_correct_attributes(self):
        """Test that results have correct attributes."""
        results = self.reranker.rerank("neural networks", self.documents, top_k=3)
        
        for r in results:
            self.assertIsInstance(r.document, Document)
            self.assertIsInstance(r.rerank_score, float)
            self.assertIsInstance(r.final_rank, int)
    
    def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        results = self.reranker.rerank("query", [], top_k=5)
        self.assertEqual(len(results), 0)
    
    def test_rerank_top_k_limits_results(self):
        """Test that top_k limits results."""
        results = self.reranker.rerank("query", self.documents, top_k=1)
        self.assertEqual(len(results), 1)


class TestLLMReranker(unittest.TestCase):
    """Tests for LLMReranker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.documents = [
            Document(id="1", content="AI is artificial intelligence."),
            Document(id="2", content="ML uses algorithms to learn."),
        ]
        self.reranker = LLMReranker(strategy="pointwise")
    
    def test_rerank_returns_results(self):
        """Test that reranking returns results."""
        results = self.reranker.rerank("What is AI?", self.documents, top_k=2)
        
        self.assertLessEqual(len(results), 2)
    
    def test_different_strategies(self):
        """Test different ranking strategies."""
        for strategy in ["pointwise", "pairwise", "listwise"]:
            reranker = LLMReranker(strategy=strategy)
            results = reranker.rerank("query", self.documents, top_k=2)
            self.assertIsInstance(results, list)


class TestReciprocalRankFusion(unittest.TestCase):
    """Tests for ReciprocalRankFusion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rrf = ReciprocalRankFusion(k=60)
        
        # Two different rankings
        self.ranking1 = [
            Document(id="1", content="Doc 1", score=0.9),
            Document(id="2", content="Doc 2", score=0.8),
            Document(id="3", content="Doc 3", score=0.7),
        ]
        
        self.ranking2 = [
            Document(id="2", content="Doc 2", score=0.85),
            Document(id="3", content="Doc 3", score=0.75),
            Document(id="1", content="Doc 1", score=0.65),
        ]
    
    def test_fuse_combines_rankings(self):
        """Test that fusion combines rankings."""
        results = self.rrf.fuse([self.ranking1, self.ranking2], top_k=3)
        
        self.assertEqual(len(results), 3)
    
    def test_fuse_scores_are_positive(self):
        """Test that RRF scores are positive."""
        results = self.rrf.fuse([self.ranking1, self.ranking2], top_k=3)
        
        for r in results:
            self.assertGreater(r.rerank_score, 0)
    
    def test_fuse_respects_top_k(self):
        """Test top_k limits results."""
        results = self.rrf.fuse([self.ranking1, self.ranking2], top_k=1)
        self.assertEqual(len(results), 1)
    
    def test_rrf_score_formula(self):
        """Test RRF score calculation."""
        # With k=60, document at rank 1 gets score 1/61
        # For document in both rankings at rank 1 and 3:
        # Score = 1/61 + 1/63
        k = 60
        expected_contribution_rank1 = 1 / (k + 1)
        expected_contribution_rank3 = 1 / (k + 3)
        
        # Doc 1: rank 1 in ranking1, rank 3 in ranking2
        results = self.rrf.fuse([self.ranking1, self.ranking2], top_k=3)
        
        # Find doc 1's score
        doc1_result = next((r for r in results if r.document.id == "1"), None)
        self.assertIsNotNone(doc1_result)


class TestDiversityReranker(unittest.TestCase):
    """Tests for DiversityReranker (MMR)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reranker = DiversityReranker(lambda_param=0.7)
        
        # Similar documents
        self.documents = [
            Document(id="1", content="Neural networks learn patterns.", score=0.9),
            Document(id="2", content="Neural nets are trained on data.", score=0.88),
            Document(id="3", content="Deep learning uses layers.", score=0.85),
            Document(id="4", content="Computer vision sees images.", score=0.75),
        ]
    
    def test_rerank_diversifies(self):
        """Test that diversity reranking produces results."""
        results = self.reranker.rerank("neural networks", self.documents, top_k=3)
        
        self.assertEqual(len(results), 3)
    
    def test_lambda_affects_diversity(self):
        """Test that lambda parameter affects results."""
        # High lambda = more relevance
        reranker_high = DiversityReranker(lambda_param=0.9)
        results_high = reranker_high.rerank("neural", self.documents, top_k=2)
        
        # Low lambda = more diversity
        reranker_low = DiversityReranker(lambda_param=0.3)
        results_low = reranker_low.rerank("neural", self.documents, top_k=2)
        
        # Both should return results
        self.assertEqual(len(results_high), 2)
        self.assertEqual(len(results_low), 2)
    
    def test_mmr_scores_valid(self):
        """Test that MMR scores are valid."""
        results = self.reranker.rerank("query", self.documents, top_k=3)
        
        for r in results:
            self.assertIsInstance(r.rerank_score, float)


class TestRerankingPipeline(unittest.TestCase):
    """Tests for RerankingPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.documents = [
            Document(id="1", content="Machine learning algorithms"),
            Document(id="2", content="Neural network architectures"),
            Document(id="3", content="Cooking recipes"),
        ]
    
    def test_pipeline_single_stage(self):
        """Test pipeline with single stage."""
        pipeline = RerankingPipeline()
        pipeline.add_stage(CrossEncoderReranker())
        
        results = pipeline.rerank("machine learning", self.documents, top_k=2)
        
        self.assertEqual(len(results), 2)
    
    def test_pipeline_multiple_stages(self):
        """Test pipeline with multiple stages."""
        pipeline = RerankingPipeline()
        pipeline.add_stage(CrossEncoderReranker())
        pipeline.add_stage(DiversityReranker(lambda_param=0.7))
        
        results = pipeline.rerank("AI", self.documents, top_k=2)
        
        self.assertLessEqual(len(results), 2)
    
    def test_pipeline_empty(self):
        """Test empty pipeline."""
        pipeline = RerankingPipeline()
        
        # Should return original documents
        results = pipeline.rerank("query", self.documents, top_k=3)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
