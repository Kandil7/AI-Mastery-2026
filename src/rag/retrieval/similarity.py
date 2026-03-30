"""
Similarity Retriever
====================

Basic similarity-based retrieval using vector search.
"""

from typing import Any, Dict, List, Optional

from .base import BaseRetriever, RetrievalResult, RetrievalResults
from src.vector_stores.base import VectorStore


class SimilarityRetriever(BaseRetriever):
    """
    Simple similarity-based retriever.

    Uses vector similarity search to find relevant documents.

    Example:
        >>> from src.vector_stores import FAISSStore, VectorStoreConfig
        >>>
        >>> store = FAISSStore(VectorStoreConfig(dim=384))
        >>> retriever = SimilarityRetriever(store, top_k=5)
        >>>
        >>> results = retriever.retrieve(query_vector)
        >>> for result in results:
        ...     print(f"{result.id}: {result.score:.3f}")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        """
        Initialize similarity retriever.

        Args:
            vector_store: Vector store to search
            top_k: Default number of results
            score_threshold: Minimum score threshold
        """
        super().__init__(top_k)
        self.vector_store = vector_store
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str | List[float],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """
        Retrieve documents by similarity.

        Args:
            query: Query vector (or string if store supports it)
            top_k: Number of results
            **kwargs: Additional search parameters

        Returns:
            RetrievalResults object
        """
        top_k = top_k or self.top_k

        # If query is string, assume vector store handles embedding
        if isinstance(query, str):
            # Some vector stores support direct text search
            search_results = self.vector_store.search(
                query_vector=[],  # Will be handled by store
                top_k=top_k,
            )
        else:
            search_results = self.vector_store.search(
                query_vector=query,
                top_k=top_k,
            )

        # Convert to RetrievalResults
        results = [
            RetrievalResult(
                id=result.id,
                content=result.metadata.get("content", ""),
                score=result.score,
                metadata=result.metadata,
            )
            for result in search_results.results
            if result.score >= self.score_threshold
        ]

        return RetrievalResults(
            results=results,
            query_time_ms=search_results.query_time_ms,
            total_count=len(results),
        )

    def set_score_threshold(self, threshold: float) -> None:
        """Update score threshold."""
        self.score_threshold = threshold

    def __repr__(self) -> str:
        return f"SimilarityRetriever(store={self.vector_store}, top_k={self.top_k})"
