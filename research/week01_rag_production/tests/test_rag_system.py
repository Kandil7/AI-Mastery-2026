"""
Comprehensive Unit Tests for Production RAG System

This module contains comprehensive unit tests for the production RAG system,
covering all major components and functionality. The tests follow best practices
for testing RAG systems including mocking external dependencies, testing edge cases,
and validating both functionality and performance.

The test suite includes:
- Unit tests for individual components
- Integration tests for component interactions
- Performance tests for critical paths
- Error handling tests
- Validation tests for inputs and outputs

Testing Strategy:
- Mock external dependencies (LLMs, databases, etc.)
- Use fixtures for common test objects
- Parameterized tests for multiple scenarios
- Property-based testing where appropriate
- Performance benchmarks
- Quality metrics validation

Test Categories:
- Document processing and ingestion
- Retrieval functionality
- Generation functionality
- API endpoints
- Configuration management
- Error handling
- Performance validation
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List
import numpy as np
from datetime import datetime

from src.pipeline import RAGPipeline, RAGConfig
from src.retrieval import Document, RetrievalResult, HybridRetriever
from src.chunking import chunk_document, ChunkingConfig, ChunkingStrategy
from src.config import RAGConfig as AppConfig, ModelConfig, RetrievalConfig
from src.ingestion import IngestionRequest, IngestionResult
from src.ingestion.file_processor import FileManager, FileProcessingResult
from src.retrieval.query_processing import QueryClassificationResult, QueryProcessingResult, QueryType
from src.retrieval.vector_store import VectorRecord, VectorConfig, VectorDBType


# Fixtures for common test objects
@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="test_doc_1",
        content="This is a test document for the RAG system. It contains sample content that can be used for testing retrieval and generation functionality.",
        source="unittest",
        doc_type="test",
        metadata={"category": "unittest", "test_field": "value"}
    )


@pytest.fixture
def sample_documents():
    """Create a list of sample documents for testing."""
    docs = []
    for i in range(5):
        doc = Document(
            id=f"test_doc_{i}",
            content=f"This is test document {i}. It contains sample content for testing purposes. The content is varied to simulate real documents.",
            source="unittest",
            doc_type="test",
            metadata={"category": "unittest", "index": i}
        )
        docs.append(doc)
    return docs


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return RAGConfig(
        generator_model="gpt2",  # Using gpt2 for testing
        dense_model="all-MiniLM-L6-v2",
        alpha=0.5,
        fusion="rrf",
        top_k=3,
        max_new_tokens=100
    )


@pytest.fixture
def mock_app_config():
    """Create a mock application configuration for testing."""
    return AppConfig(
        app_name="Test RAG API",
        app_version="1.0.0",
        environment="testing",
        debug=True,
        models=ModelConfig(
            generator_model="gpt2",
            dense_model="all-MiniLM-L6-v2",
            top_k=3,
            max_new_tokens=100,
        ),
        retrieval=RetrievalConfig(
            alpha=0.5,
            fusion_method="rrf",
        ),
    )


class TestDocumentClass:
    """Test cases for the Document class."""

    def test_document_creation(self, sample_document):
        """Test creating a document with valid parameters."""
        assert sample_document.id == "test_doc_1"
        assert "RAG system" in sample_document.content
        assert sample_document.source == "unittest"
        assert sample_document.doc_type == "test"
        assert sample_document.metadata["category"] == "unittest"

    def test_document_validation(self):
        """Test document validation with invalid parameters."""
        # Test with empty content
        with pytest.raises(ValueError):
            Document(id="test", content="")

        # Test with empty ID
        with pytest.raises(ValueError):
            Document(id="", content="test content")

        # Test with whitespace-only content
        with pytest.raises(ValueError):
            Document(id="test", content="   ")

    def test_validate_access_public(self, sample_document):
        """Test document access validation for public documents."""
        sample_document.access_control = {"level": "public"}
        user_perms = {"level": "public"}
        assert sample_document.validate_access(user_perms) is True

    def test_validate_access_private(self, sample_document):
        """Test document access validation for private documents."""
        sample_document.access_control = {"level": "confidential"}
        user_perms = {"level": "public"}
        assert sample_document.validate_access(user_perms) is False

        user_perms = {"level": "confidential"}
        assert sample_document.validate_access(user_perms) is True

    def test_get_content_length(self, sample_document):
        """Test getting document content length."""
        expected_length = len(sample_document.content)
        assert sample_document.get_content_length() == expected_length

    def test_get_metadata_summary(self, sample_document):
        """Test getting document metadata summary."""
        summary = sample_document.get_metadata_summary()
        assert "id" in summary
        assert "source" in summary
        assert "doc_type" in summary
        assert summary["id"] == sample_document.id


class TestRAGPipeline:
    """Test cases for the RAGPipeline class."""

    @pytest.fixture(autouse=True)
    def setup_mock_transformer(self):
        """Setup mock for sentence transformer."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            mock_transformer_instance = Mock()
            mock_transformer_instance.encode.return_value = np.random.rand(1, 384)  # Mock embedding
            mock_transformer_class.return_value = mock_transformer_instance
            yield mock_transformer_instance

    @pytest.fixture(autouse=True)
    def setup_mock_chromadb(self):
        """Setup mock for ChromaDB."""
        with patch('chromadb.PersistentClient') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.return_value = mock_client
            yield mock_collection

    @pytest.fixture(autouse=True)
    def setup_mock_generator(self):
        """Setup mock for transformers pipeline."""
        with patch('src.pipeline.pipeline') as mock_pipeline:
            mock_generator = Mock()
            mock_generator.return_value = [{"generated_text": "Answer: mocked"}]
            mock_pipeline.return_value = mock_generator
            yield mock_generator

    def test_pipeline_initialization(self, mock_config):
        """Test initializing the RAG pipeline."""
        pipeline = RAGPipeline(mock_config)
        assert pipeline.config == mock_config
        assert pipeline.retriever is not None
        assert pipeline.generator is not None

    def test_pipeline_index(self, sample_documents, mock_config):
        """Test indexing documents in the pipeline."""
        pipeline = RAGPipeline(mock_config)
        
        # Mock the add_documents method to avoid actual processing
        with patch.object(pipeline.retriever.dense_retriever, 'add_documents') as mock_add:
            pipeline.index(sample_documents)
            mock_add.assert_called_once_with(sample_documents)

    def test_pipeline_retrieve(self, sample_documents, mock_config):
        """Test retrieving documents from the pipeline."""
        pipeline = RAGPipeline(mock_config)
        
        # Add sample documents
        pipeline.index(sample_documents)
        
        # Mock the retrieve method to return known results
        mock_results = [
            RetrievalResult(
                document=sample_documents[0],
                score=0.9,
                rank=1
            )
        ]
        
        with patch.object(pipeline.retriever, 'retrieve', return_value=mock_results):
            results = pipeline.retrieve("test query", top_k=1)
            assert len(results) == 1
            assert results[0].document.id == sample_documents[0].id

    def test_pipeline_generate(self, sample_documents, mock_config):
        """Test generating responses from the pipeline."""
        pipeline = RAGPipeline(mock_config)
        
        # Create mock retrieval results
        mock_results = [
            RetrievalResult(
                document=sample_documents[0],
                score=0.9,
                rank=1
            )
        ]
        
        # Mock the generator to return a known response
        mock_generator_response = [{"generated_text": "Context:\nTest content\n\nQuestion: What is this?\n\nAnswer: This is a test response."}]
        
        with patch.object(pipeline.generator, '__call__', return_value=mock_generator_response):
            response = pipeline.generate("What is this?", mock_results)
            assert "test response" in response.lower()

    def test_pipeline_query(self, sample_documents, mock_config):
        """Test the complete query pipeline."""
        pipeline = RAGPipeline(mock_config)
        
        # Add sample documents
        pipeline.index(sample_documents)
        
        # Mock both retrieve and generate methods
        mock_retrieve_results = [
            RetrievalResult(
                document=sample_documents[0],
                score=0.9,
                rank=1
            )
        ]
        
        mock_generated_response = "This is a test response based on the context."
        
        with patch.object(pipeline, 'retrieve', return_value=mock_retrieve_results):
            with patch.object(pipeline, 'generate', return_value=mock_generated_response):
                result = pipeline.query("What is this?", top_k=1)
                
                assert result["query"] == "What is this?"
                assert result["response"] == mock_generated_response
                assert len(result["retrieved_documents"]) == 1
                assert result["retrieved_documents"][0]["id"] == sample_documents[0].id


class TestHybridRetriever:
    """Test cases for the HybridRetriever class."""

    def test_hybrid_retriever_initialization(self):
        """Test initializing the hybrid retriever."""
        retriever = HybridRetriever(
            alpha=0.7,
            fusion="rrf",
            dense_model="all-MiniLM-L6-v2",
            sparse_k1=1.2,
            sparse_b=0.75
        )
        
        assert retriever.alpha == 0.7
        assert retriever.fusion == "rrf"
        assert retriever.dense_retriever is not None
        assert retriever.sparse_retriever is not None

    def test_index_documents(self, sample_documents):
        """Test indexing documents in the hybrid retriever."""
        retriever = HybridRetriever()
        
        # Mock the add_documents methods
        with patch.object(retriever.dense_retriever, 'add_documents') as mock_dense_add, \
             patch.object(retriever.sparse_retriever, 'add_documents') as mock_sparse_add:
            
            retriever.index(sample_documents)
            
            mock_dense_add.assert_called_once_with(sample_documents)
            mock_sparse_add.assert_called_once_with(sample_documents)

    def test_retrieve_rrf_fusion(self, sample_documents):
        """Test retrieval with RRF fusion."""
        retriever = HybridRetriever(fusion="rrf")
        retriever.index(sample_documents)
        
        # Mock the individual retrievers to return different results
        dense_results = [
            RetrievalResult(document=sample_documents[0], score=0.8, rank=1),
            RetrievalResult(document=sample_documents[1], score=0.6, rank=2)
        ]
        
        sparse_results = [
            RetrievalResult(document=sample_documents[1], score=0.9, rank=1),
            RetrievalResult(document=sample_documents[2], score=0.7, rank=2)
        ]
        
        with patch.object(retriever.dense_retriever, 'retrieve', return_value=dense_results), \
             patch.object(retriever.sparse_retriever, 'retrieve', return_value=sparse_results):
            
            results = retriever.retrieve("test query", top_k=2)
            
            # Should have results from both systems, fused appropriately
            assert len(results) <= 2
            # Check that results are properly ranked
            for i, result in enumerate(results, 1):
                assert result.rank == i

    def test_retrieve_weighted_fusion(self, sample_documents):
        """Test retrieval with weighted fusion."""
        retriever = HybridRetriever(fusion="weighted", alpha=0.5)
        retriever.index(sample_documents)
        
        # Mock the individual retrievers to return different results
        dense_results = [
            RetrievalResult(document=sample_documents[0], score=0.8, rank=1),
            RetrievalResult(document=sample_documents[1], score=0.6, rank=2)
        ]
        
        sparse_results = [
            RetrievalResult(document=sample_documents[1], score=0.9, rank=1),
            RetrievalResult(document=sample_documents[2], score=0.7, rank=2)
        ]
        
        with patch.object(retriever.dense_retriever, 'retrieve', return_value=dense_results), \
             patch.object(retriever.sparse_retriever, 'retrieve', return_value=sparse_results):
            
            results = retriever.retrieve("test query", top_k=2)
            
            # Should have results from both systems, fused appropriately
            assert len(results) <= 2


class TestChunking:
    """Test cases for chunking functionality."""

    def test_recursive_chunking(self, sample_document):
        """Test recursive character chunking."""
        config = ChunkingConfig(
            chunk_size=50,
            chunk_overlap=10,
            strategy=ChunkingStrategy.RECURSIVE
        )
        
        chunks = chunk_document(sample_document, config)
        
        # Should have multiple chunks since the document is longer than 50 chars
        assert len(chunks) > 1
        
        # Check that chunks are properly sized
        for chunk in chunks[:-1]:  # All but the last chunk
            assert len(chunk.content) <= config.chunk_size
        
        # Check that metadata is preserved
        for chunk in chunks:
            assert "original_id" in chunk.metadata
            assert chunk.metadata["original_id"] == sample_document.id

    def test_semantic_chunking(self, sample_document):
        """Test semantic chunking."""
        # Add sentence structure to the document
        sample_document.content = "This is the first sentence. This is the second sentence. And this is the third sentence."
        
        config = ChunkingConfig(
            chunk_size=60,
            chunk_overlap=5,
            strategy=ChunkingStrategy.SEMANTIC
        )
        
        chunks = chunk_document(sample_document, config)
        
        # Should have multiple chunks
        assert len(chunks) >= 1
        
        # Check that metadata is preserved
        for chunk in chunks:
            assert "original_id" in chunk.metadata
            assert chunk.metadata["original_id"] == sample_document.id

    def test_chunking_config_validation(self):
        """Test validation of chunking configuration."""
        # Test with invalid chunk size
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=0, chunk_overlap=10)
        
        # Test with overlap larger than chunk size
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=100, chunk_overlap=150)


class TestFileProcessor:
    """Test cases for file processing functionality."""

    @pytest.fixture
    def file_manager(self):
        """Create a file manager for testing."""
        return FileManager(upload_dir="test_uploads", max_file_size=10 * 1024 * 1024)  # 10MB

    def test_get_file_type_pdf(self, file_manager):
        """Test detecting PDF file type."""
        file_type = file_manager.get_file_type("document.pdf")
        assert file_type.value == "pdf"

    def test_get_file_type_txt(self, file_manager):
        """Test detecting TXT file type."""
        file_type = file_manager.get_file_type("notes.txt")
        assert file_type.value == "txt"

    def test_get_file_type_invalid(self, file_manager):
        """Test detecting invalid file type."""
        with pytest.raises(ValueError):
            file_manager.get_file_type("script.exe")

    def test_validate_file_upload_valid(self, file_manager):
        """Test validating a valid file upload."""
        from src.ingestion.file_processor import FileUploadRequest
        
        req = FileUploadRequest(
            filename="test.pdf",
            content_type="application/pdf",
            file_size=1024
        )
        
        errors = file_manager.validate_file_upload(req)
        assert len(errors) == 0

    def test_validate_file_upload_invalid_type(self, file_manager):
        """Test validating a file with invalid type."""
        from src.ingestion.file_processor import FileUploadRequest
        
        req = FileUploadRequest(
            filename="script.exe",
            content_type="application/x-executable",
            file_size=1024
        )
        
        errors = file_manager.validate_file_upload(req)
        assert len(errors) > 0
        assert "Unsupported file type" in errors[0]

    def test_validate_file_upload_too_large(self, file_manager):
        """Test validating a file that's too large."""
        from src.ingestion.file_processor import FileUploadRequest
        
        req = FileUploadRequest(
            filename="large.pdf",
            content_type="application/pdf",
            file_size=100 * 1024 * 1024  # 100MB, exceeding limit
        )
        
        errors = file_manager.validate_file_upload(req)
        assert len(errors) > 0
        assert "exceeds maximum allowed size" in errors[0]


class TestIngestion:
    """Test cases for ingestion functionality."""

    def test_ingestion_request_validation(self):
        """Test validation of ingestion requests."""
        req = IngestionRequest(
            metadata={"source": "test"},
            chunk_size=1000,
            chunk_overlap=200
        )
        
        assert req.chunk_size == 1000
        assert req.chunk_overlap == 200
        assert req.metadata["source"] == "test"

    def test_ingestion_result_creation(self):
        """Test creation of ingestion results."""
        result = IngestionResult(
            success=True,
            message="Test message",
            processed_documents=5,
            indexed_documents=5,
            processing_time_ms=100.0,
            errors=[],
            warnings=[],
            metadata={"test": "value"}
        )
        
        assert result.success is True
        assert result.processed_documents == 5
        assert result.indexed_documents == 5
        assert result.processing_time_ms == 100.0


class TestQueryProcessing:
    """Test cases for query processing functionality."""

    def test_query_classification_result(self):
        """Test creation of query classification results."""
        result = QueryClassificationResult(
            QueryType.SIMPLE_FACT,
            0.8,
            ["test", "query"],
            ["TestEntity"],
            "factual inquiry"
        )
        
        assert result.query_type == QueryType.SIMPLE_FACT
        assert result.confidence == 0.8
        assert "test" in result.keywords
        assert "TestEntity" in result.entities
        assert result.intent == "factual inquiry"

    def test_query_processing_result(self, sample_documents):
        """Test creation of query processing results."""
        mock_results = [
            RetrievalResult(
                document=sample_documents[0],
                score=0.9,
                rank=1
            )
        ]
        
        result = QueryProcessingResult(
            query="Test query?",
            response="Test response",
            sources=mock_results,
            query_type=QueryType.SIMPLE_FACT,
            processing_time_ms=50.0,
            confidence_score=0.85,
            citations=[{"source_id": "doc1", "rank": 1}],
            metadata={"model": "test-model"}
        )
        
        assert result.query == "Test query?"
        assert result.response == "Test response"
        assert result.query_type == QueryType.SIMPLE_FACT
        assert result.processing_time_ms == 50.0
        assert result.confidence_score == 0.85
        assert len(result.citations) == 1


class TestVectorStore:
    """Test cases for vector storage functionality."""

    def test_vector_record_creation(self):
        """Test creation of vector records."""
        record = VectorRecord(
            id="test_vec_1",
            vector=[0.1, 0.2, 0.3],
            metadata={"category": "test"},
            document_id="doc_1",
            text_content="Test content"
        )
        
        assert record.id == "test_vec_1"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.metadata["category"] == "test"
        assert record.document_id == "doc_1"
        assert record.text_content == "Test content"

    def test_vector_config_creation(self):
        """Test creation of vector configurations."""
        config = VectorConfig(
            db_type=VectorDBType.IN_MEMORY,
            collection_name="test_collection",
            persist_directory="./test_data",
            dimension=384,
            metric="cosine",
            batch_size=32
        )
        
        assert config.db_type == VectorDBType.IN_MEMORY
        assert config.collection_name == "test_collection"
        assert config.dimension == 384
        assert config.metric == "cosine"
        assert config.batch_size == 32


class TestConfiguration:
    """Test cases for configuration management."""

    def test_app_config_creation(self, mock_app_config):
        """Test creation of application configuration."""
        assert mock_app_config.app_name == "Test RAG API"
        assert mock_app_config.environment == "testing"
        assert mock_app_config.debug is True

    def test_config_is_development(self, mock_app_config):
        """Test environment detection."""
        assert mock_app_config.is_development() is True
        assert mock_app_config.is_production() is False

    def test_config_database_url(self, mock_app_config):
        """Test database URL construction."""
        url = mock_app_config.get_database_url()
        # Should return the default URL since no credentials are set
        assert url == mock_app_config.database.url


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_pipeline_performance_large_documents(self, mock_config):
        """Test pipeline performance with larger documents."""
        pipeline = RAGPipeline(mock_config)
        
        # Create a larger document
        large_content = "This is a test sentence. " * 100  # 100 sentences
        large_doc = Document(
            id="large_doc",
            content=large_content,
            source="unittest",
            doc_type="test"
        )
        
        # Time the indexing operation
        import time
        start_time = time.time()
        pipeline.index([large_doc])
        indexing_time = time.time() - start_time
        
        # Indexing should complete in a reasonable time (under 5 seconds for this test)
        assert indexing_time < 5.0

    @pytest.mark.asyncio
    async def test_retrieval_performance_multiple_queries(self, sample_documents, mock_config):
        """Test retrieval performance with multiple queries."""
        pipeline = RAGPipeline(mock_config)
        pipeline.index(sample_documents)
        
        # Perform multiple queries and measure performance
        import time
        start_time = time.time()
        
        for i in range(10):  # 10 queries
            pipeline.retrieve(f"test query {i}", top_k=2)
        
        total_time = time.time() - start_time
        avg_time_per_query = total_time / 10
        
        # Each query should complete reasonably quickly (under 1 second for this test)
        assert avg_time_per_query < 1.0


# Integration tests
class TestIntegration:
    """Integration tests for multiple components."""

    def test_full_pipeline_integration(self, sample_documents, mock_config):
        """Test the full pipeline from indexing to querying."""
        pipeline = RAGPipeline(mock_config)
        
        # Index documents
        pipeline.index(sample_documents)
        
        # Query the system
        result = pipeline.query("What is this system about?", top_k=2)
        
        # Verify the result structure
        assert "query" in result
        assert "response" in result
        assert "retrieved_documents" in result
        assert result["query"] == "What is this system about?"
        assert isinstance(result["retrieved_documents"], list)
        
        # Should have retrieved some documents
        assert len(result["retrieved_documents"]) <= 2  # As specified by top_k


# Error handling tests
class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_pipeline_with_empty_documents(self, mock_config):
        """Test pipeline behavior with empty documents list."""
        pipeline = RAGPipeline(mock_config)
        
        # Indexing empty list should not raise an error
        pipeline.index([])
        
        # Querying with no documents should return empty results
        result = pipeline.query("Any question?", top_k=3)
        assert len(result["retrieved_documents"]) == 0

    def test_retrieval_with_invalid_query(self, sample_documents, mock_config):
        """Test retrieval with very short query."""
        pipeline = RAGPipeline(mock_config)
        pipeline.index(sample_documents)
        
        # Very short query should still return results (though possibly empty)
        result = pipeline.retrieve("", top_k=3)
        # Result could be empty but shouldn't crash

    def test_document_with_special_characters(self, mock_config):
        """Test handling of documents with special characters."""
        special_doc = Document(
            id="special_doc",
            content="This document has special characters: <script>alert('xss')</script>",
            source="unittest",
            doc_type="test"
        )
        
        pipeline = RAGPipeline(mock_config)
        pipeline.index([special_doc])
        
        # Should be able to retrieve and process without issues
        result = pipeline.query("special characters", top_k=1)
        assert "retrieved_documents" in result


if __name__ == "__main__":
    pytest.main([__file__])
