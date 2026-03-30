"""
Diversity Reranker
==================

Reranks for diversity to reduce redundancy.

This approach ensures the top results cover diverse
aspects of the query, rather than returning many
similar documents.

Algorithms:
- MMR (Maximal Marginal Relevance)
- Diversity score based on embedding distance
- Topic-based diversity

Example:
    >>> reranker = DiversityReranker(
    ...     embedder=embedder,
    ...     diversity_factor=0.5,
    ... )
    >>> results = reranker.rerank(query, documents)
"""

import time
from typing import Any, Dict, List, Optional, Callable
import math

from .base import BaseReranker, RerankResult, RerankResults


class DiversityReranker(BaseReranker):
    """
    Diversity-focused reranker using MMR.

    Maximal Marginal Relevance (MMR) balances relevance
    and diversity to avoid redundant results.

    MMR = argmax [ λ * relevance(doc) - (1-λ) * max(similarity(doc, selected)) ]

    Example:
        >>> reranker = DiversityReranker(
        ...     embedder=embedder,
        ...     diversity_factor=0.5,  # Balance relevance/diversity
        ...     top_k=5,
        ... )
        >>> results = reranker.rerank(query, documents)
    """

    def __init__(
        self,
        embedder: Callable[[str], List[float]],
        top_k: int = 5,
        diversity_factor: float = 0.5,
    ):
        """
        Initialize diversity reranker.

        Args:
            embedder: Function to embed text to vectors
            top_k: Number of results to return
            diversity_factor: Balance between relevance (1) and diversity (0)
        """
        super().__init__(top_k)
        self.embedder = embedder
        self.diversity_factor = diversity_factor

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResults:
        """
        Rerank for diversity using MMR.

        Args:
            query: Query text
            results: Documents to rerank
            top_k: Number of results

        Returns:
            RerankResults with diverse ranking
        """
        top_k = top_k or self.top_k
        start_time = time.time()

        if not results:
            return RerankResults(results=[], original_count=0)

        # Embed query and documents
        query_embedding = self.embedder(query)
        doc_embeddings = [self.embedder(d["content"]) for d in results]

        # Calculate relevance scores (query-document similarity)
        relevance_scores = [
            self._cosine_similarity(query_embedding, emb)
            for emb in doc_embeddings
        ]

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(results)))

        while remaining_indices and len(selected_indices) < top_k:
            best_mmr = -float("inf")
            best_idx = None

            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (max similarity to selected)
                if selected_indices:
                    max_similarity = max(
                        self._cosine_similarity(
                            doc_embeddings[idx],
                            doc_embeddings[sel_idx],
                        )
                        for sel_idx in selected_indices
                    )
                else:
                    max_similarity = 0.0

                # MMR score
                mmr = (
                    self.diversity_factor * relevance
                    - (1 - self.diversity_factor) * max_similarity
                )

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Build reranked results
        reranked = []
        for rank, idx in enumerate(selected_indices):
            doc = results[idx]
            reranked.append(
                RerankResult(
                    id=doc["id"],
                    content=doc["content"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=relevance_scores[idx],
                    rank=rank + 1,
                    metadata={
                        **doc.get("metadata", {}),
                        "mmr_score": best_mmr if rank == len(selected_indices) - 1 else None,
                        "diversity_rank": rank + 1,
                    },
                )
            )

        rerank_time = (time.time() - start_time) * 1000

        return RerankResults(
            results=reranked,
            rerank_time_ms=rerank_time,
            original_count=len(results),
        )

    def set_diversity_factor(self, factor: float) -> None:
        """
        Update diversity factor.

        Args:
            factor: New diversity factor (0-1)
                - 0: Maximum diversity (ignore relevance)
                - 1: Maximum relevance (ignore diversity)
                - 0.5: Balanced (default)
        """
        if not 0 <= factor <= 1:
            raise ValueError("diversity_factor must be between 0 and 1")
        self.diversity_factor = factor

    def __repr__(self) -> str:
        return f"DiversityReranker(diversity={self.diversity_factor})"
