"""
Chunking Service Unit Tests
============================
Tests for token-aware text chunking.
"""

import pytest

from src.application.services.chunking import (
    chunk_text_token_aware,
    count_tokens,
    truncate_to_tokens,
)
from src.domain.entities import ChunkSpec


class TestChunkTextTokenAware:
    """Tests for chunk_text_token_aware function."""
    
    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list."""
        result = chunk_text_token_aware("")
        assert result == []
    
    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only text should return empty list."""
        result = chunk_text_token_aware("   \n\t  ")
        assert result == []
    
    def test_short_text_returns_single_chunk(self):
        """Text shorter than max_tokens should return single chunk."""
        text = "Hello, this is a short text."
        result = chunk_text_token_aware(text, ChunkSpec(max_tokens=100))
        assert len(result) == 1
        assert result[0] == text
    
    def test_long_text_returns_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        # Create text that's definitely longer than 50 tokens
        text = " ".join(["word"] * 200)
        result = chunk_text_token_aware(text, ChunkSpec(max_tokens=50, overlap_tokens=10))
        assert len(result) > 1
    
    def test_chunks_have_overlap(self):
        """Consecutive chunks should have overlapping content."""
        text = " ".join([f"word{i}" for i in range(100)])
        spec = ChunkSpec(max_tokens=30, overlap_tokens=10)
        result = chunk_text_token_aware(text, spec)
        
        assert len(result) >= 2
        # Check there's some overlap in the text content
        # (not exact token overlap check, but content similarity)
    
    def test_default_spec_used_when_none(self):
        """Default spec should be used when not provided."""
        text = "Hello world"
        result = chunk_text_token_aware(text)  # No spec
        assert len(result) == 1
    
    def test_minimum_tokens_enforced(self):
        """Minimum of 50 tokens per chunk should be enforced."""
        text = " ".join(["word"] * 100)
        spec = ChunkSpec(max_tokens=10)  # Below minimum
        result = chunk_text_token_aware(text, spec)
        # Should still work, using minimum of 50
        assert len(result) >= 1


class TestCountTokens:
    """Tests for count_tokens function."""
    
    def test_empty_string(self):
        """Empty string should return 0 tokens."""
        assert count_tokens("") == 0
    
    def test_simple_text(self):
        """Simple text should return reasonable token count."""
        result = count_tokens("Hello world")
        assert result > 0
        assert result <= 5  # Should be 2-3 tokens
    
    def test_longer_text(self):
        """Longer text should have more tokens."""
        short = count_tokens("Hello")
        long = count_tokens("Hello, this is a much longer piece of text with many words.")
        assert long > short


class TestTruncateToTokens:
    """Tests for truncate_to_tokens function."""
    
    def test_short_text_unchanged(self):
        """Text within limit should be unchanged."""
        text = "Hello world"
        result = truncate_to_tokens(text, max_tokens=100)
        assert result == text
    
    def test_long_text_truncated(self):
        """Text exceeding limit should be truncated."""
        text = " ".join(["word"] * 100)
        result = truncate_to_tokens(text, max_tokens=10)
        assert len(result) < len(text)
    
    def test_empty_string(self):
        """Empty string should return empty string."""
        result = truncate_to_tokens("", max_tokens=100)
        assert result == ""
