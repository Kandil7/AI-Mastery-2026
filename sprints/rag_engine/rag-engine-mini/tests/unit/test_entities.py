"""
Domain Entities Unit Tests
===========================
Tests for core domain objects.
"""

import pytest

from src.domain.entities import (
    TenantId,
    DocumentId,
    ChunkId,
    Chunk,
    Answer,
    StoredFile,
    DocumentStatus,
    ChunkSpec,
)


class TestTenantId:
    """Tests for TenantId value object."""
    
    def test_creation(self):
        """Should create TenantId with value."""
        tenant = TenantId("user_123")
        assert tenant.value == "user_123"
    
    def test_str_representation(self):
        """String should return the value."""
        tenant = TenantId("user_123")
        assert str(tenant) == "user_123"
    
    def test_immutability(self):
        """TenantId should be frozen (immutable)."""
        tenant = TenantId("user_123")
        with pytest.raises(AttributeError):
            tenant.value = "other"  # type: ignore
    
    def test_equality(self):
        """Same values should be equal."""
        t1 = TenantId("user_123")
        t2 = TenantId("user_123")
        assert t1 == t2
    
    def test_hashable(self):
        """Should be usable in sets/dicts."""
        t1 = TenantId("user_123")
        t2 = TenantId("user_456")
        s = {t1, t2}
        assert len(s) == 2


class TestChunk:
    """Tests for Chunk entity."""
    
    def test_creation(self):
        """Should create Chunk with all fields."""
        chunk = Chunk(
            id="chunk_1",
            tenant_id=TenantId("user_1"),
            document_id=DocumentId("doc_1"),
            text="Sample text content",
        )
        
        assert chunk.id == "chunk_1"
        assert chunk.tenant_id.value == "user_1"
        assert chunk.text == "Sample text content"
    
    def test_equality_by_id(self):
        """Chunks with same ID should be equal."""
        c1 = Chunk(
            id="chunk_1",
            tenant_id=TenantId("user_1"),
            document_id=DocumentId("doc_1"),
            text="Text A",
        )
        c2 = Chunk(
            id="chunk_1",
            tenant_id=TenantId("user_1"),
            document_id=DocumentId("doc_1"),
            text="Text B",  # Different text, same ID
        )
        
        assert c1 == c2
    
    def test_hashable(self):
        """Should be usable in sets."""
        c1 = Chunk(
            id="chunk_1",
            tenant_id=TenantId("user_1"),
            document_id=DocumentId("doc_1"),
            text="Text",
        )
        c2 = Chunk(
            id="chunk_2",
            tenant_id=TenantId("user_1"),
            document_id=DocumentId("doc_1"),
            text="Text",
        )
        
        s = {c1, c2}
        assert len(s) == 2


class TestAnswer:
    """Tests for Answer entity."""
    
    def test_creation(self):
        """Should create Answer with text and sources."""
        answer = Answer(
            text="The answer is 42.",
            sources=["chunk_1", "chunk_2"],
        )
        
        assert answer.text == "The answer is 42."
        assert len(answer.sources) == 2
    
    def test_sources_immutable(self):
        """Sources should be converted to tuple."""
        answer = Answer(
            text="Answer",
            sources=["a", "b"],
        )
        
        assert isinstance(answer.sources, tuple)


class TestChunkSpec:
    """Tests for ChunkSpec value object."""
    
    def test_defaults(self):
        """Should have sensible defaults."""
        spec = ChunkSpec()
        
        assert spec.max_tokens == 512
        assert spec.overlap_tokens == 50
        assert spec.encoding_name == "cl100k_base"
    
    def test_custom_values(self):
        """Should accept custom values."""
        spec = ChunkSpec(
            max_tokens=256,
            overlap_tokens=25,
            encoding_name="gpt2",
        )
        
        assert spec.max_tokens == 256
        assert spec.overlap_tokens == 25
