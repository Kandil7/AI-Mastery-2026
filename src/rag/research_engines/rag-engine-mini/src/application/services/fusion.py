"""
RRF Fusion Service
===================
Pure service for merging vector and keyword search results.

خدمة دمج نتائج البحث المتجه والكلمات المفتاحية
"""

from typing import Sequence

from src.application.services.scoring import ScoredChunk


def rrf_fusion(
    *,
    vector_hits: Sequence[ScoredChunk],
    keyword_hits: Sequence[ScoredChunk],
    k: int = 60,
    out_limit: int = 40,
) -> list[ScoredChunk]:
    """
    Merge results using Reciprocal Rank Fusion (RRF).
    
    Args:
        vector_hits: Results from vector search (sorted by score)
        keyword_hits: Results from keyword search (sorted by score)
        k: RRF constant (default 60, controls rank influence)
        out_limit: Maximum results to return
    
    Returns:
        Fused results sorted by combined RRF score
    
    Algorithm:
        RRF_score = Σ 1 / (k + rank)
        
        For each result list, calculate 1/(k+rank) and sum across lists.
        Higher score = better match across both methods.
    
    Design Decision: RRF over weighted combination because:
    - No score calibration needed (vector vs keyword scores are incomparable)
    - Robust to different score distributions
    - Simple and effective
    
    قرار التصميم: RRF بدلاً من المزيج المرجح لعدم الحاجة لمعايرة الدرجات
    
    Reference:
        Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
        individual Rank Learning Methods"
    
    Example:
        >>> fused = rrf_fusion(
        ...     vector_hits=vec_results,
        ...     keyword_hits=kw_results,
        ...     out_limit=40
        ... )
    """
    # Accumulator for RRF scores
    scores: dict[str, float] = {}
    
    # Chunk lookup for reconstruction
    chunk_by_id: dict[str, ScoredChunk] = {}
    
    def add_hits(hits: Sequence[ScoredChunk]) -> None:
        """Add RRF contribution from a result list."""
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.id
            # RRF formula: 1 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the chunk reference (prefer first occurrence)
            if chunk_id not in chunk_by_id:
                chunk_by_id[chunk_id] = hit
    
    # Process both result sets
    add_hits(vector_hits)
    add_hits(keyword_hits)
    
    # Sort by combined RRF score
    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build output with new RRF scores
    result: list[ScoredChunk] = []
    for chunk_id, rrf_score in sorted_ids[:out_limit]:
        if chunk_id in chunk_by_id:
            original = chunk_by_id[chunk_id]
            # Create new ScoredChunk with RRF score
            result.append(ScoredChunk(chunk=original.chunk, score=rrf_score))
    
    return result


def weighted_fusion(
    *,
    vector_hits: Sequence[ScoredChunk],
    keyword_hits: Sequence[ScoredChunk],
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    out_limit: int = 40,
) -> list[ScoredChunk]:
    """
    Alternative: Merge results using weighted score combination.
    
    Note: Requires score normalization for fair comparison.
    RRF is generally preferred for its robustness.
    
    ملاحظة: يتطلب تطبيع الدرجات للمقارنة العادلة
    """
    # Normalize scores to [0, 1] range
    def normalize(hits: Sequence[ScoredChunk]) -> dict[str, float]:
        if not hits:
            return {}
        scores = [h.score for h in hits]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1.0
        return {h.chunk.id: (h.score - min_s) / range_s for h in hits}
    
    vec_norm = normalize(vector_hits)
    kw_norm = normalize(keyword_hits)
    
    # Combine scores
    all_ids = set(vec_norm.keys()) | set(kw_norm.keys())
    combined: dict[str, float] = {}
    
    for chunk_id in all_ids:
        vec_score = vec_norm.get(chunk_id, 0.0) * vector_weight
        kw_score = kw_norm.get(chunk_id, 0.0) * keyword_weight
        combined[chunk_id] = vec_score + kw_score
    
    # Sort and build output
    chunk_by_id = {h.chunk.id: h for h in list(vector_hits) + list(keyword_hits)}
    sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    return [
        ScoredChunk(chunk=chunk_by_id[cid].chunk, score=score)
        for cid, score in sorted_ids[:out_limit]
        if cid in chunk_by_id
    ]
