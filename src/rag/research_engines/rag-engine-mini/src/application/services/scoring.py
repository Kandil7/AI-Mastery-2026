"""
Scored Chunk Dataclass
=======================
Data structure for chunks with relevance scores.

هيكل بيانات القطع مع درجات الصلة
"""

from dataclasses import dataclass

from src.domain.entities import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """
    A chunk with an associated relevance score.
    
    Used for:
    - Vector search results (cosine similarity)
    - Keyword search results (BM25-style rank)
    - Fusion and reranking operations
    
    قطعة مع درجة صلة مرتبطة
    """
    chunk: Chunk
    score: float
    
    def __hash__(self) -> int:
        return hash(self.chunk.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoredChunk):
            return NotImplemented
        return self.chunk.id == other.chunk.id
