"""
RRF Fusion Unit Tests
======================
Tests for Reciprocal Rank Fusion merging.
"""

import pytest

from src.application.services.fusion import rrf_fusion
from src.application.services.scoring import ScoredChunk
from src.domain.entities import Chunk, TenantId, DocumentId


def make_chunk(chunk_id: str, text: str = "test") -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=chunk_id,
        tenant_id=TenantId("test_tenant"),
        document_id=DocumentId("test_doc"),
        text=text,
    )


def make_scored(chunk_id: str, score: float) -> ScoredChunk:
    """Create a test scored chunk."""
    return ScoredChunk(chunk=make_chunk(chunk_id), score=score)


class TestRRFFusion:
    """Tests for rrf_fusion function."""
    
    def test_empty_inputs(self):
        """Empty inputs should return empty result."""
        result = rrf_fusion(vector_hits=[], keyword_hits=[])
        assert result == []
    
    def test_single_list(self):
        """Single list should work correctly."""
        vector = [make_scored("a", 0.9), make_scored("b", 0.8)]
        result = rrf_fusion(vector_hits=vector, keyword_hits=[])
        assert len(result) == 2
        assert result[0].chunk.id == "a"  # Higher rank
    
    def test_merge_preserves_all(self):
        """Fusion should preserve all unique chunks."""
        vector = [make_scored("a", 0.9)]
        keyword = [make_scored("b", 0.8)]
        result = rrf_fusion(vector_hits=vector, keyword_hits=keyword)
        ids = {r.chunk.id for r in result}
        assert ids == {"a", "b"}
    
    def test_common_chunk_boosted(self):
        """Chunk in both lists should have higher RRF score."""
        # 'a' appears in both lists
        vector = [make_scored("a", 0.9), make_scored("b", 0.8)]
        keyword = [make_scored("a", 0.9), make_scored("c", 0.8)]
        
        result = rrf_fusion(vector_hits=vector, keyword_hits=keyword)
        
        # 'a' should be first due to appearing in both
        assert result[0].chunk.id == "a"
    
    def test_out_limit_respected(self):
        """Output limit should be respected."""
        chunks = [make_scored(f"chunk_{i}", 0.9 - i * 0.01) for i in range(10)]
        result = rrf_fusion(vector_hits=chunks, keyword_hits=[], out_limit=5)
        assert len(result) == 5
    
    def test_rrf_scores_are_calculated(self):
        """RRF scores should be calculated and stored."""
        vector = [make_scored("a", 0.9)]
        result = rrf_fusion(vector_hits=vector, keyword_hits=[], k=60)
        
        # Expected RRF score: 1/(60+1) = 0.01639...
        expected = 1.0 / (60 + 1)
        assert abs(result[0].score - expected) < 0.001
    
    def test_ordering_by_rrf_score(self):
        """Results should be ordered by RRF score descending."""
        vector = [make_scored("a", 0.9), make_scored("b", 0.8)]
        keyword = [make_scored("b", 0.9), make_scored("a", 0.8)]
        
        result = rrf_fusion(vector_hits=vector, keyword_hits=keyword)
        
        # Both appear in both lists, but at different ranks
        # Scores should be calculated properly
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
