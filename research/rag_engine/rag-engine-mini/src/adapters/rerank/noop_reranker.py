"""
No-op Reranker Adapter
=======================
Passthrough reranker that doesn't modify ranking.

محول إعادة ترتيب بدون عملية
"""

from typing import Sequence

from src.domain.entities import Chunk


class NoopReranker:
    """
    Passthrough reranker - returns chunks unchanged.
    
    Used when:
    - Reranking is disabled via config
    - Testing without model overhead
    - Baseline comparison
    
    يُستخدم عند تعطيل إعادة الترتيب
    """
    
    def rerank(
        self,
        *,
        query: str,
        chunks: Sequence[Chunk],
        top_n: int,
    ) -> Sequence[Chunk]:
        """Return first top_n chunks without reranking."""
        return list(chunks)[:top_n]
