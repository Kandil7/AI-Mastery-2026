"""
Base Retriever Interface
========================

Abstract base class for retrieval strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class RetrievalResult:
    """
    Single retrieval result.

    Attributes:
        id: Document/chunk ID
        content: Document content
        score: Relevance score
        metadata: Additional metadata
    """
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResults:
    """
    Collection of retrieval results.

    Attributes:
        results: List of individual results
        query_time_ms: Query execution time
        total_count: Total results available
    """
    results: List[RetrievalResult]
    query_time_ms: float = 0.0
    total_count: int = 0

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
    def contents(self) -> List[str]:
        """Get result contents."""
        return [r.content for r in self.results]


class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.

    All retrieval strategies should inherit from this class.

    Example:
        >>> class CustomRetriever(BaseRetriever):
        ...     def retrieve(self, query, top_k=5):
        ...         # Custom retrieval logic
        ...         pass
    """

    def __init__(self, top_k: int = 5):
        """
        Initialize retriever.

        Args:
            top_k: Default number of results to return
        """
        self.top_k = top_k

    @abstractmethod
    def retrieve(
        self,
        query: str | List[float],
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        """
        Retrieve relevant documents.

        Args:
            query: Query string or embedding
            top_k: Number of results (overrides default)
            **kwargs: Additional retrieval parameters

        Returns:
            RetrievalResults object
        """
        pass

    def retrieve_with_scores(
        self,
        query: str | List[float],
        top_k: Optional[int] = None,
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> RetrievalResults:
        """
        Retrieve with score threshold filtering.

        Args:
            query: Query string or embedding
            top_k: Number of results
            threshold: Minimum score threshold
            **kwargs: Additional parameters

        Returns:
            Filtered RetrievalResults
        """
        results = self.retrieve(query, top_k, **kwargs)

        # Filter by threshold
        filtered = [r for r in results.results if r.score >= threshold]

        return RetrievalResults(
            results=filtered,
            query_time_ms=results.query_time_ms,
            total_count=len(filtered),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(top_k={self.top_k})"
