"""
Base Reranker Interface
=======================

Abstract base class for reranking strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RerankResult:
    """
    Single reranking result.

    Attributes:
        id: Document/chunk ID
        content: Document content
        original_score: Original retrieval score
        rerank_score: New reranking score
        rank: New rank position
        metadata: Additional metadata
    """
    id: str
    content: str
    original_score: float
    rerank_score: float
    rank: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResults:
    """
    Collection of reranking results.

    Attributes:
        results: List of individual results
        rerank_time_ms: Reranking execution time
        original_count: Number of input results
    """
    results: List[RerankResult]
    rerank_time_ms: float = 0.0
    original_count: int = 0

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]

    @property
    def ids(self) -> List[str]:
        """Get result IDs."""
        return [r.id for r in self.results]

    @property
    def top_1(self) -> Optional[RerankResult]:
        """Get top result."""
        return self.results[0] if self.results else None

    @property
    def top_3(self) -> List[RerankResult]:
        """Get top 3 results."""
        return self.results[:3]

    @property
    def top_5(self) -> List[RerankResult]:
        """Get top 5 results."""
        return self.results[:5]


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.

    All reranking strategies should inherit from this class.

    Example:
        >>> class CustomReranker(BaseReranker):
        ...     def rerank(self, query, results, top_k=5):
        ...         # Custom reranking logic
        ...         pass
    """

    def __init__(self, top_k: int = 5):
        """
        Initialize reranker.

        Args:
            top_k: Default number of results to return
        """
        self.top_k = top_k

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResults:
        """
        Rerank retrieval results.

        Args:
            query: Original query
            results: List of retrieval results to rerank
            top_k: Number of results to return

        Returns:
            RerankResults object with new rankings
        """
        pass

    def rerank_from_retrieval(
        self,
        query: str,
        retrieval_results: Any,  # RetrievalResults
        top_k: Optional[int] = None,
    ) -> RerankResults:
        """
        Rerank directly from retrieval results.

        Convenience method that converts RetrievalResults
        to the expected format.

        Args:
            query: Original query
            retrieval_results: Results from retriever
            top_k: Number of results to return

        Returns:
            RerankResults object
        """
        # Convert retrieval results to dict format
        results = [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in retrieval_results.results
        ]

        return self.rerank(query, results, top_k)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"
