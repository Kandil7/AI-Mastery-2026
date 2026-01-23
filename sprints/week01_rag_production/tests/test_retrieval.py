"""
Comprehensive tests for the retrieval components of the RAG system.

This module contains unit and integration tests for all retrieval components:
- Document class
- DenseRetriever
- SparseRetriever (BM25)
- HybridRetriever
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.retrieval import Document, DenseRetriever, SparseRetriever, HybridRetriever, RetrievalResult


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Mock sentence transformer to avoid model downloads."""
    with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
        mock_transformer_instance = Mock()
        mock_transformer_instance.encode.return_value = np.random.rand(1, 384)
        mock_transformer_class.return_value = mock_transformer_instance
        yield mock_transformer_instance


@pytest.fixture(autouse=True)
def mock_chromadb():
    """Mock ChromaDB client to avoid persistence in tests."""
    with patch('chromadb.PersistentClient') as mock_chromadb:
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value = mock_client
        yield mock_collection

class TestDocument:
    """Test cases for the Document class."""
    
    def test_document_creation_valid(self):
        """Test creating a valid document."""
        doc = Document(
            id="test_id",
            content="Test content",
            source="test_source",
            doc_type="test_type"
        )
        assert doc.id == "test_id"
        assert doc.content == "Test content"
        assert doc.source == "test_source"
        assert doc.doc_type == "test_type"
        
    def test_document_validation(self):
        """Test document validation on creation."""
        # Test invalid ID
        with pytest.raises(ValueError):
            Document(id="", content="Test content")
        
        # Test invalid content
        with pytest.raises(ValueError):
            Document(id="test_id", content="")
        
        # Test whitespace-only content
        with pytest.raises(ValueError):
            Document(id="test_id", content="   ")
        
        # Test negative page number
        with pytest.raises(ValueError):
            Document(id="test_id", content="Test", page_number=-1)
    
    def test_document_access_control(self):
        """Test document access control validation."""
        doc = Document(
            id="test_id",
            content="Test content",
            access_control={"level": "confidential"}
        )
        
        # Test access allowed
        user_perms_public = {"level": "public"}
        user_perms_internal = {"level": "internal"}
        user_perms_confidential = {"level": "confidential"}
        user_perms_restricted = {"level": "restricted"}
        
        # Public user should not have access to confidential doc
        assert not doc.validate_access(user_perms_public)
        # Internal user should not have access to confidential doc
        assert not doc.validate_access(user_perms_internal)
        # Confidential user should have access
        assert doc.validate_access(user_perms_confidential)
        # Restricted user should have access
        assert doc.validate_access(user_perms_restricted)
    
    def test_document_metadata_summary(self):
        """Test document metadata summary generation."""
        doc = Document(
            id="test_id",
            content="Test content",
            source="test_source",
            doc_type="test_type",
            page_number=5,
            section_title="Introduction"
        )
        
        summary = doc.get_metadata_summary()
        assert summary["id"] == "test_id"
        assert summary["source"] == "test_source"
        assert summary["doc_type"] == "test_type"
        assert summary["page_number"] == "5"
        assert summary["section_title"] == "Introduction"


class TestDenseRetriever:
    """Test cases for the DenseRetriever class."""
    
    def test_initialization(self):
        """Test DenseRetriever initialization."""
        retriever = DenseRetriever(dense_model="all-MiniLM-L6-v2")
        assert retriever.encoder is not None
        assert retriever.documents == []
        assert retriever.batch_size == 32
    
    def test_add_documents(self):
        """Test adding documents to the dense retriever."""
        retriever = DenseRetriever(dense_model="all-MiniLM-L6-v2")
        docs = [
            Document(id="1", content="First document"),
            Document(id="2", content="Second document")
        ]
        
        retriever.add_documents(docs)
        
        assert len(retriever.documents) == 2
        assert retriever.documents[0].id == "1"
        assert retriever.documents[1].id == "2"
    
    def test_add_empty_documents(self):
        """Test adding empty documents list."""
        retriever = DenseRetriever(dense_model="all-MiniLM-L6-v2")
        retriever.add_documents([])
        
        assert len(retriever.documents) == 0
    
    def test_retrieve_no_documents(self):
        """Test retrieval when no documents are indexed."""
        retriever = DenseRetriever(dense_model="all-MiniLM-L6-v2")
        results = retriever.retrieve("test query")
        
        assert results == []
    
    def test_retrieve_with_documents(self):
        """Test retrieval with indexed documents."""
        retriever = DenseRetriever(dense_model="all-MiniLM-L6-v2")
        docs = [
            Document(id="1", content="Machine learning is a subset of AI"),
            Document(id="2", content="Deep learning involves neural networks"),
            Document(id="3", content="Natural language processing deals with text")
        ]
        retriever.add_documents(docs)
        
        results = retriever.retrieve("artificial intelligence", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].rank == 1
        assert results[1].rank == 2
    
    def test_clear_index(self):
        """Test clearing the index."""
        retriever = DenseRetriever(dense_model="all-MiniLM-L6-v2")
        docs = [Document(id="1", content="Test document")]
        retriever.add_documents(docs)
        
        assert retriever.get_document_count() == 1
        
        retriever.clear_index()
        
        assert retriever.get_document_count() == 0
        assert retriever.documents == []


class TestSparseRetriever:
    """Test cases for the SparseRetriever class."""
    
    def test_initialization(self):
        """Test SparseRetriever initialization."""
        retriever = SparseRetriever(k1=1.2, b=0.75)
        assert retriever.k1 == 1.2
        assert retriever.b == 0.75
        assert retriever.documents == []
        assert retriever.idf == {}
    
    def test_add_documents(self):
        """Test adding documents to the sparse retriever."""
        retriever = SparseRetriever()
        docs = [
            Document(id="1", content="First document about AI"),
            Document(id="2", content="Second document about ML")
        ]
        
        retriever.add_documents(docs)
        
        assert len(retriever.documents) == 2
        assert len(retriever.doc_freqs) == 2
        assert len(retriever.doc_lens) == 2
        assert retriever.avg_doc_len > 0
    
    def test_retrieve_no_documents(self):
        """Test retrieval when no documents are indexed."""
        retriever = SparseRetriever()
        results = retriever.retrieve("test query")
        
        assert results == []
    
    def test_retrieve_with_documents(self):
        """Test retrieval with indexed documents."""
        retriever = SparseRetriever()
        docs = [
            Document(id="1", content="Machine learning is a subset of AI"),
            Document(id="2", content="Deep learning involves neural networks"),
            Document(id="3", content="Natural language processing deals with text")
        ]
        retriever.add_documents(docs)
        
        results = retriever.retrieve("machine learning", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].rank == 1
        assert results[1].rank == 2


class TestHybridRetriever:
    """Test cases for the HybridRetriever class."""
    
    def test_initialization(self):
        """Test HybridRetriever initialization."""
        retriever = HybridRetriever(alpha=0.6, fusion="rrf", dense_model="all-MiniLM-L6-v2")
        assert retriever.alpha == 0.6
        assert retriever.fusion == "rrf"
        assert isinstance(retriever.dense_retriever, DenseRetriever)
        assert isinstance(retriever.sparse_retriever, SparseRetriever)
    
    def test_index_documents(self):
        """Test indexing documents in hybrid retriever."""
        retriever = HybridRetriever()
        docs = [
            Document(id="1", content="First document about AI"),
            Document(id="2", content="Second document about ML")
        ]
        
        retriever.index(docs)
        
        assert len(retriever.dense_retriever.documents) == 2
        assert len(retriever.sparse_retriever.documents) == 2
    
    def test_retrieve_rrf_fusion(self):
        """Test retrieval with RRF fusion."""
        retriever = HybridRetriever(fusion="rrf")
        docs = [
            Document(id="1", content="Machine learning is a subset of AI"),
            Document(id="2", content="Deep learning involves neural networks"),
            Document(id="3", content="Natural language processing deals with text")
        ]
        retriever.index(docs)
        
        results = retriever.retrieve("artificial intelligence", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].rank == 1
        assert results[1].rank == 2
    
    def test_retrieve_weighted_fusion(self):
        """Test retrieval with weighted fusion."""
        retriever = HybridRetriever(fusion="weighted", alpha=0.7)
        docs = [
            Document(id="1", content="Machine learning is a subset of AI"),
            Document(id="2", content="Deep learning involves neural networks"),
            Document(id="3", content="Natural language processing deals with text")
        ]
        retriever.index(docs)
        
        results = retriever.retrieve("machine learning", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_retrieve_densite_fusion(self):
        """Test retrieval with densite fusion."""
        retriever = HybridRetriever(fusion="densite", alpha=0.5)
        docs = [
            Document(id="1", content="Machine learning is a subset of AI"),
            Document(id="2", content="Deep learning involves neural networks"),
            Document(id="3", content="Natural language processing deals with text")
        ]
        retriever.index(docs)
        
        results = retriever.retrieve("neural networks", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_retrieve_combsum_fusion(self):
        """Test retrieval with combsum fusion."""
        retriever = HybridRetriever(fusion="combsum", alpha=0.5)
        docs = [
            Document(id="1", content="Machine learning is a subset of AI"),
            Document(id="2", content="Deep learning involves neural networks"),
            Document(id="3", content="Natural language processing deals with text")
        ]
        retriever.index(docs)
        
        results = retriever.retrieve("deep learning", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_retrieve_combmnz_fusion(self):
        """Test retrieval with combmnz fusion."""
        retriever = HybridRetriever(fusion="combmnz", alpha=0.5)
        docs = [
            Document(id="1", content="Machine learning is a subset of AI"),
            Document(id="2", content="Deep learning involves neural networks"),
            Document(id="3", content="Natural language processing deals with text")
        ]
        retriever.index(docs)
        
        results = retriever.retrieve("learning", top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
    
    def test_get_fusion_strategies(self):
        """Test getting available fusion strategies."""
        retriever = HybridRetriever()
        strategies = retriever.get_fusion_strategies()
        
        expected = ["rrf", "weighted", "densite", "combsum", "combmnz"]
        assert strategies == expected


class TestIntegration:
    """Integration tests for the retrieval system."""
    
    def test_full_retrieval_pipeline(self):
        """Test the complete retrieval pipeline with different fusion strategies."""
        retriever = HybridRetriever(fusion="rrf", alpha=0.6)
        
        # Add diverse documents
        docs = [
            Document(id="1", content="Artificial Intelligence (AI) is intelligence demonstrated by machines"),
            Document(id="2", content="Machine learning is a method of data analysis that automates analytical model building"),
            Document(id="3", content="Deep learning is part of a broader family of machine learning methods"),
            Document(id="4", content="Neural networks are a series of algorithms that mimic the operations of a human brain"),
            Document(id="5", content="Natural language processing helps computers understand, interpret and manipulate human language")
        ]
        
        retriever.index(docs)
        
        # Test different queries
        queries = [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "natural language processing"
        ]
        
        for query in queries:
            results = retriever.retrieve(query, top_k=3)
            
            # Verify results structure
            assert len(results) <= 3
            assert all(isinstance(r, RetrievalResult) for r in results)
            assert all(r.rank == i+1 for i, r in enumerate(results))  # Ranks should be sequential
            assert all(r.score >= 0 for r in results)  # Scores should be non-negative (for RRF/BM25)
    
    def test_different_fusion_strategies_consistency(self):
        """Test that different fusion strategies return properly structured results."""
        docs = [
            Document(id="1", content="Machine learning is a subset of artificial intelligence"),
            Document(id="2", content="Deep learning uses neural networks with multiple layers"),
            Document(id="3", content="Natural language processing analyzes human language computationally")
        ]
        
        strategies = ["rrf", "weighted", "densite", "combsum", "combmnz"]
        
        for strategy in strategies:
            retriever = HybridRetriever(fusion=strategy, alpha=0.5)
            retriever.index(docs)
            
            results = retriever.retrieve("machine learning", top_k=2)
            
            assert len(results) == 2
            assert all(isinstance(r, RetrievalResult) for r in results)
            assert results[0].rank == 1
            assert results[1].rank == 2
            assert all(r.document.id in ["1", "2", "3"] for r in results)
