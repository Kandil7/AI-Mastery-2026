"""
Unit tests for RAG pipeline and LLM components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, '..')

from src.llm.rag import (
    Document, RetrievalResult, TextChunker, 
    EmbeddingModel, Retriever, Reranker, 
    ContextAssembler, RAGPipeline
)
from src.llm.attention import (
    MultiHeadAttention, TransformerBlock,
    FeedForwardNetwork, LayerNorm
)


class TestDocument:
    """Tests for Document class."""

    def test_document_creation(self):
        """Test Document creation with content and metadata."""
        doc = Document(content="Test content", metadata={"author": "test"})
        
        assert doc.content == "Test content"
        assert doc.metadata["author"] == "test"
        assert len(doc.id) == 12  # Default ID is 12 chars from hash

    def test_document_custom_id(self):
        """Test Document creation with custom ID."""
        doc = Document(content="Test content", id="custom_id")
        
        assert doc.id == "custom_id"

    def test_document_embedding(self):
        """Test Document with embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        doc = Document(content="Test", embedding=embedding)
        
        np.testing.assert_array_equal(doc.embedding, embedding)


class TestRetrievalResult:
    """Tests for RetrievalResult class."""

    def test_retrieval_result_creation(self):
        """Test RetrievalResult creation."""
        doc = Document(content="Test document")
        result = RetrievalResult(document=doc, score=0.8, rank=1)
        
        assert result.document.content == "Test document"
        assert result.score == 0.8
        assert result.rank == 1


class TestTextChunker:
    """Tests for TextChunker."""

    def test_fixed_chunking(self):
        """Test fixed-size chunking."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunker = TextChunker(chunk_size=4, overlap=1, strategy='fixed')
        
        chunks = chunker.chunk(text)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        # First chunk should have 4 words
        assert len(chunks[0].content.split()) == 4
        # Check overlap between first and second chunks
        first_chunk_words = set(chunks[0].content.split())
        second_chunk_words = set(chunks[1].content.split())
        # Should have 1 overlapping word
        overlap = first_chunk_words.intersection(second_chunk_words)
        assert len(overlap) == 1

    def test_semantic_chunking(self):
        """Test semantic chunking."""
        text = "First sentence. Second sentence. Third sentence! Fourth sentence? Fifth sentence."
        chunker = TextChunker(chunk_size=10, strategy='semantic')
        
        chunks = chunker.chunk(text)
        
        # Should create chunks based on sentence boundaries
        assert len(chunks) > 0
        # All chunks should end with sentence terminators
        for chunk in chunks:
            content = chunk.content.strip()
            assert content.endswith(('.', '!', '?'))

    def test_recursive_chunking(self):
        """Test recursive chunking."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunker = TextChunker(chunk_size=5, strategy='recursive')
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""

    def test_embedding_model_creation(self):
        """Test EmbeddingModel creation."""
        model = EmbeddingModel(dim=128)
        
        assert model.dim == 128
        assert model.model_fn is None

    def test_embed_function(self):
        """Test embedding of texts."""
        model = EmbeddingModel(dim=64)
        texts = ["Hello world", "Test embedding"]
        
        embeddings = model.embed(texts)
        
        assert embeddings.shape == (2, 64)
        # Should be random vectors (since no model_fn provided)
        assert not np.allclose(embeddings[0], embeddings[1])

    def test_embed_query(self):
        """Test embedding of a single query."""
        model = EmbeddingModel(dim=32)
        
        query_embedding = model.embed_query("Test query")
        
        assert query_embedding.shape == (32,)

    def test_embed_documents(self):
        """Test embedding of documents."""
        model = EmbeddingModel(dim=32)
        docs = [Document(content="Doc 1"), Document(content="Doc 2")]
        
        embedded_docs = model.embed_documents(docs)
        
        for doc in embedded_docs:
            assert doc.embedding is not None
            assert doc.embedding.shape == (32,)


class TestRetriever:
    """Tests for Retriever."""

    def test_retriever_add_documents(self):
        """Test adding documents to retriever."""
        embedding_model = EmbeddingModel(dim=64)
        retriever = Retriever(embedding_model)
        
        docs = [
            Document(content="Document 1 about cats"),
            Document(content="Document 2 about dogs"),
            Document(content="Document 3 about birds")
        ]
        
        retriever.add_documents(docs)
        
        assert len(retriever.documents) == 3
        assert retriever.index is not None
        assert retriever.index.shape == (3, 64)

    def test_retriever_retrieve(self):
        """Test retrieval functionality."""
        embedding_model = EmbeddingModel(dim=32)
        retriever = Retriever(embedding_model)
        
        # Add documents
        docs = [
            Document(content="Machine learning is great"),
            Document(content="Deep learning with neural networks"),
            Document(content="Natural language processing")
        ]
        retriever.add_documents(docs)
        
        # Query
        results = retriever.retrieve("learning algorithms", k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Scores should be between -1 and 1 (cosine similarity)
        assert all(-1 <= r.score <= 1 for r in results)

    def test_retriever_empty_retrieve(self):
        """Test retrieval with no documents."""
        embedding_model = EmbeddingModel(dim=32)
        retriever = Retriever(embedding_model)
        
        results = retriever.retrieve("test query")
        
        assert len(results) == 0


class TestReranker:
    """Tests for Reranker."""

    def test_reranker_rerank(self):
        """Test reranking functionality."""
        # Create mock rerank function
        def mock_rerank(pairs):
            # Return fixed scores for testing
            return [0.9, 0.7, 0.3]  # Scores for 3 input pairs
        
        reranker = Reranker(rerank_fn=mock_rerank)
        
        # Create results to rerank
        docs = [
            Document(content="Document 1"),
            Document(content="Document 2"), 
            Document(content="Document 3")
        ]
        initial_results = [
            RetrievalResult(docs[0], 0.8, 0),
            RetrievalResult(docs[1], 0.6, 1),
            RetrievalResult(docs[2], 0.4, 2)
        ]
        
        reranked_results = reranker.rerank("query", initial_results, top_k=2)
        
        assert len(reranked_results) == 2
        # First result should have the highest score after reranking
        assert reranked_results[0].score >= reranked_results[1].score
        # Ranks should be updated
        assert reranked_results[0].rank == 0
        assert reranked_results[1].rank == 1

    def test_reranker_no_function(self):
        """Test reranker works without rerank function."""
        reranker = Reranker()  # No rerank function
        
        docs = [Document(content="Doc 1"), Document(content="Doc 2")]
        results = [
            RetrievalResult(docs[0], 0.8, 0),
            RetrievalResult(docs[1], 0.6, 1)
        ]
        
        # Should just sort by original scores
        reranked_results = reranker.rerank("query", results, top_k=2)
        
        assert len(reranked_results) == 2
        assert reranked_results[0].score == 0.8


class TestContextAssembler:
    """Tests for ContextAssembler."""

    def test_context_assembler_stuffing(self):
        """Test stuffing strategy."""
        assembler = ContextAssembler(max_tokens=20, strategy='stuffing')
        
        docs = [
            Document(content="First document with some content"),
            Document(content="Second document with more content"),
            Document(content="Third document with additional content")
        ]
        results = [
            RetrievalResult(docs[0], 0.9, 0),
            RetrievalResult(docs[1], 0.8, 1),
            RetrievalResult(docs[2], 0.7, 2)
        ]
        
        context = assembler.assemble("test query", results)
        
        # Context should contain content from documents
        assert "First document" in context
        assert "Second document" in context
        # Third might not be included if it exceeds token limit

    def test_context_assembler_simple(self):
        """Test simple strategy."""
        assembler = ContextAssembler(strategy='simple')
        
        docs = [Document(content="Content 1"), Document(content="Content 2")]
        results = [
            RetrievalResult(docs[0], 0.9, 0),
            RetrievalResult(docs[1], 0.8, 1)
        ]
        
        context = assembler.assemble("test query", results)
        
        assert "Content 1" in context
        assert "Content 2" in context


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def test_rag_pipeline_basic(self):
        """Test basic RAG pipeline functionality."""
        rag = RAGPipeline()
        
        # Add documents
        docs = [
            Document(content="AI is artificial intelligence"),
            Document(content="Machine learning is part of AI"),
            Document(content="Deep learning uses neural networks")
        ]
        rag.add_documents(docs)
        
        # Query
        response = rag.query("What is AI?", k=2, return_sources=True)
        
        assert 'answer' in response
        assert 'context' in response
        assert 'sources' in response
        assert len(response['sources']) <= 2  # k=2

    def test_rag_pipeline_with_llm(self):
        """Test RAG pipeline with LLM function."""
        def mock_llm(prompt):
            return "This is a mock response based on the context."
        
        rag = RAGPipeline(llm_fn=mock_llm)
        
        docs = [Document(content="Test document")]
        rag.add_documents(docs)
        
        response = rag.query("What is this about?")
        
        assert response['answer'] == "This is a mock response based on the context."

    def test_rag_pipeline_no_sources(self):
        """Test RAG pipeline without returning sources."""
        rag = RAGPipeline()
        
        docs = [Document(content="Test content")]
        rag.add_documents(docs)
        
        response = rag.query("What is this?", return_sources=False)
        
        assert 'answer' in response
        assert 'context' in response
        assert 'sources' not in response


class TestAttentionMechanisms:
    """Tests for attention mechanisms."""

    def test_scaled_dot_product_attention_shape(self):
        """Test scaled dot product attention shape."""
        from src.llm.attention import scaled_dot_product_attention
        
        Q = np.random.randn(2, 10, 64)  # batch=2, seq=10, d_k=64
        K = np.random.randn(2, 15, 64)  # batch=2, seq=15, d_k=64
        V = np.random.randn(2, 15, 64)  # batch=2, seq=15, d_v=64
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (2, 10, 64)
        assert attention_weights.shape == (2, 10, 15)

    def test_multi_head_attention(self):
        """Test MultiHeadAttention."""
        mha = MultiHeadAttention(d_model=512, num_heads=8)
        
        # Self-attention input
        X = np.random.randn(4, 20, 512)  # batch=4, seq=20, d_model=512
        
        output = mha.forward(X, X, X)
        
        assert output.shape == (4, 20, 512)

    def test_multi_head_attention_num_heads_divisibility(self):
        """Test that d_model must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=511, num_heads=8)  # 511 not divisible by 8

    def test_feed_forward_network(self):
        """Test FeedForwardNetwork."""
        ffn = FeedForwardNetwork(d_model=256, d_ff=512, activation='relu')
        
        X = np.random.randn(3, 10, 256)  # batch=3, seq=10, d_model=256
        
        output = ffn.forward(X)
        
        assert output.shape == (3, 10, 256)

    def test_layer_norm(self):
        """Test LayerNorm."""
        ln = LayerNorm(d_model=128)
        
        X = np.random.randn(5, 20, 128)  # batch=5, seq=20, d_model=128
        
        output = ln.forward(X)
        
        assert output.shape == (5, 20, 128)

    def test_transformer_block(self):
        """Test TransformerBlock."""
        block = TransformerBlock(d_model=256, num_heads=8, d_ff=512)
        
        X = np.random.randn(2, 15, 256)  # batch=2, seq=15, d_model=256
        
        output = block.forward(X)
        
        assert output.shape == (2, 15, 256)


def test_rag_integration():
    """Integration test for RAG pipeline."""
    # Create a full RAG pipeline with mock LLM
    def mock_llm(prompt):
        if "AI" in prompt:
            return "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines."
        else:
            return "I don't have information about this topic in the provided context."
    
    # Create pipeline
    embedding_model = EmbeddingModel(dim=128)
    retriever = Retriever(embedding_model)
    assembler = ContextAssembler(max_tokens=500)
    
    rag = RAGPipeline(
        embedding_model=embedding_model,
        retriever=retriever,
        context_assembler=assembler,
        llm_fn=mock_llm
    )
    
    # Add knowledge base
    documents = [
        Document(content="Artificial Intelligence (AI) is intelligence demonstrated by machines."),
        Document(content="Machine learning is a subset of AI that focuses on algorithms."),
        Document(content="Deep learning uses neural networks with multiple layers."),
        Document(content="Natural Language Processing (NLP) enables machines to understand human language.")
    ]
    
    rag.add_documents(documents)
    
    # Query the system
    response = rag.query("What is Artificial Intelligence?", k=3, return_sources=True)
    
    # Validate response
    assert 'answer' in response
    assert 'context' in response
    assert 'sources' in response
    assert len(response['sources']) <= 3
    assert "Artificial Intelligence" in response['answer'] or "AI" in response['answer']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])