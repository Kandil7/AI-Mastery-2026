"""
Reranker Port
==============
Interface for reranking retrieved chunks.

منفذ إعادة الترتيب
"""

from typing import Protocol, Sequence

from src.domain.entities import Chunk


class RerankerPort(Protocol):
    """
    Port for reranking retrieved chunks.
    
    Implementations: Cross-Encoder, LLM-based, heuristic
    
    Design Decision: Reranking after retrieval significantly improves
    precision by using a more sophisticated model to re-score results.
    
    قرار التصميم: إعادة الترتيب تحسن الدقة بشكل كبير
    
    Typical flow:
    1. Retrieve ~40 candidates (fast, coarse)
    2. Rerank to top ~8 (slow, precise)
    3. Use top results for prompt
    """
    
    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[Chunk],
        top_n: int,
    ) -> Sequence[Chunk]:
        """
        Rerank chunks by relevance to query.
        
        Args:
            query: User question
            chunks: Candidate chunks to rerank
            top_n: Number of top results to return
            
        Returns:
            Top N chunks sorted by relevance (best first)
            
        Note:
            Cross-Encoder processes (query, chunk) pairs and scores each.
            More accurate than embedding similarity but slower.
            
            Cross-Encoder يعالج أزواج (استعلام، قطعة) ويسجل كل منها
        """
        ...
