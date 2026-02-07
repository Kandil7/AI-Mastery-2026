"""
Cross-Encoder Reranker Adapter
===============================
Local reranking using SentenceTransformers Cross-Encoder.

محول إعادة الترتيب بـ Cross-Encoder المحلي
"""

from typing import Sequence

from sentence_transformers import CrossEncoder

from src.domain.entities import Chunk


class CrossEncoderReranker:
    """
    Cross-Encoder reranker using SentenceTransformers.
    
    Design Decision: Local Cross-Encoder over API-based because:
    - No per-request API cost
    - Works offline
    - Faster for small batches
    - More control over model choice
    
    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - BAAI/bge-reranker-base (better quality, slower)
    
    قرار التصميم: Cross-Encoder محلي بدلاً من API لتوفير التكلفة
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        """
        Initialize Cross-Encoder model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on (cpu/cuda)
        """
        self._model = CrossEncoder(model_name, device=device)
    
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
            chunks: Candidate chunks from retrieval
            top_n: Number of top results to return
            
        Returns:
            Top N chunks sorted by Cross-Encoder score (best first)
        """
        if not chunks:
            return []
        
        # Build (query, passage) pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        
        # Score all pairs
        scores = self._model.predict(pairs)
        
        # Sort by score descending
        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        
        # Return top N
        return [chunk for chunk, _ in scored[:top_n]]
