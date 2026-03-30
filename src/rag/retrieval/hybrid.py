"""
Hybrid Retrieval
================

Combines dense and sparse retrieval for better results.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import BaseRetriever, RetrievalResult, RetrievalResults
from src.vector_stores.base import VectorStore


@dataclass
class HybridConfig:
    """
    Configuration for hybrid retrieval.

    Attributes:
        dense_weight: Weight for dense retrieval (0-1)
        sparse_weight: Weight for sparse retrieval (0-1)
        normalization: Score normalization method
    """
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    normalization: str = "minmax"  # minmax, zscore, rank

    def __post_init__(self):
        if abs(self.dense_weight + self.sparse_weight - 1.0) > 0.01:
            raise ValueError("dense_weight + sparse_weight must equal 1.0")


class HybridRetrieval(BaseRetriever):
    """
    Hybrid retrieval combining dense and sparse methods.

    Combines vector similarity with keyword-based search (BM25)
    for improved retrieval quality.

    Example:
        >>> config = HybridConfig(dense_weight=0.7, sparse_weight=0.3)
        >>> retriever = HybridRetrieval(
        ...     dense_store=faiss_store,
        ...     sparse_store=bm25_store,
        ...     config=config,
        ... )
        >>> results = retriever.retrieve("query text")
    """

    def __init__(
        self,
        dense_store: VectorStore,
        sparse_store: Any,  # BM25 or similar
        config: Optional[HybridConfig] = None,
        top_k: int = 5,
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_store: Dense vector store
            sparse_store: Sparse vector store (BM25, etc.)
            config: Hybrid configuration
            top_k: Default number of results
        """
        super().__init__(top_k)
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.config = config or HybridConfig()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """
        Retrieve using hybrid approach.

        Args:
            query: Query text
            top_k: Number of results
            **kwargs: Additional parameters

        Returns:
            Combined RetrievalResults
        """
        top_k = top_k or self.top_k

        # Get dense results
        # Note: This assumes query is already embedded or store handles it
        dense_results = self.dense_store.search(
            query_vector=query if isinstance(query, list) else [],
            top_k=top_k * 2,  # Get more for fusion
        )

        # Get sparse results
        sparse_results = self.sparse_store.search(
            query=query,
            top_k=top_k * 2,
        )

        # Normalize and combine scores
        combined = self._reciprocal_rank_fusion(
            dense_results.results,
            sparse_results.results,
            k=60,  # RRF constant
        )

        # Take top_k
        top_results = combined[:top_k]

        # Convert to RetrievalResults
        results = [
            RetrievalResult(
                id=result.id,
                content=result.metadata.get("content", ""),
                score=result.score,
                metadata=result.metadata,
            )
            for result in top_results
        ]

        return RetrievalResults(
            results=results,
            query_time_ms=max(
                dense_results.query_time_ms,
                sparse_results.query_time_ms,
            ),
            total_count=len(results),
        )

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Any],
        sparse_results: List[Any],
        k: int = 60,
    ) -> List[Any]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each result
        """
        from src.vector_stores.base import SearchResult

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Any] = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + (1.0 / (k + rank))
            result_map[result.id] = result

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            rrf_scores[result.id] = rrf_scores.get(result.id, 0) + (1.0 / (k + rank))
            if result.id not in result_map:
                result_map[result.id] = result

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build combined results
        combined = []
        for id_ in sorted_ids:
            result = result_map[id_]
            combined.append(
                SearchResult(
                    id=result.id,
                    score=rrf_scores[id_],
                    distance=0.0,
                    metadata=result.metadata,
                )
            )

        return combined

    def __repr__(self) -> str:
        return f"HybridRetrieval(dense={self.dense_store}, sparse={self.sparse_store})"
