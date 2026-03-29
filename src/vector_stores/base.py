"""
Base Vector Store Interface
===========================

Abstract base class defining the vector store interface.

All vector store implementations must inherit from this class
and implement the required methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class MetricType(str, Enum):
    """Distance metric types."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorStoreConfig:
    """
    Configuration for vector store.

    Attributes:
        dim: Vector dimension
        metric: Distance metric to use
        index_type: Type of index (for stores that support it)
        ef_construct: Construction parameter for HNSW indexes
        ef_search: Search parameter for HNSW indexes
        **kwargs: Additional store-specific configuration
    """
    dim: int
    metric: MetricType = MetricType.COSINE
    index_type: str = "flat"
    ef_construct: int = 200
    ef_search: int = 50
    **kwargs: Any

    def __post_init__(self):
        if isinstance(self.metric, str):
            self.metric = MetricType(self.metric.lower())


@dataclass
class SearchResult:
    """
    Single search result.

    Attributes:
        id: Document/chunk ID
        score: Similarity score (higher = more similar)
        distance: Distance value (lower = more similar)
        metadata: Additional metadata
        vector: Optional vector data
    """
    id: str
    score: float
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


@dataclass
class SearchResults:
    """
    Collection of search results.

    Attributes:
        results: List of individual search results
        query_time_ms: Query execution time in milliseconds
        total_count: Total matching results (for pagination)
    """
    results: List[SearchResult]
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
        """Get list of result IDs."""
        return [r.id for r in self.results]

    @property
    def scores(self) -> List[float]:
        """Get list of similarity scores."""
        return [r.score for r in self.results]


class VectorStore(ABC):
    """
    Abstract base class for vector stores.

    Provides a unified interface for vector storage and retrieval
    across different backend implementations.

    Example:
        >>> config = VectorStoreConfig(dim=384, metric="cosine")
        >>> store = FAISSStore(config)
        >>>
        >>> # Add vectors
        >>> vectors = [[0.1] * 384, [0.2] * 384]
        >>> ids = ["doc1", "doc2"]
        >>> store.upsert(vectors, ids)
        >>>
        >>> # Search
        >>> results = store.search([0.15] * 384, top_k=2)
        >>> print(f"Found {len(results)} results")
    """

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store.

        Args:
            config: Store configuration
        """
        self.config = config
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if store is initialized."""
        return self._is_initialized

    @property
    @abstractmethod
    def count(self) -> int:
        """
        Get the number of vectors in the store.

        Returns:
            Number of vectors
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the vector store.

        Creates necessary indexes and resources.
        Should be called before first use.
        """
        pass

    @abstractmethod
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Insert or update vectors.

        Args:
            vectors: List of vectors to store
            ids: List of unique IDs for each vector
            metadata: Optional metadata for each vector

        Raises:
            ValueError: If vectors and ids have different lengths
            VectorStoreError: If insertion fails
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> SearchResults:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector to search for
            top_k: Number of results to return
            filter_fn: Optional filter function for results

        Returns:
            SearchResults object containing matches

        Raises:
            ValueError: If query_vector dimension doesn't match
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by ID.

        Args:
            ids: List of IDs to delete

        Returns:
            Number of vectors deleted
        """
        pass

    @abstractmethod
    def get(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve vectors by ID.

        Args:
            ids: List of IDs to retrieve

        Returns:
            Dictionary mapping IDs to vector data and metadata
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save to
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Path to load from
        """
        pass

    def close(self) -> None:
        """
        Close the vector store.

        Releases resources and connections.
        Should be called when done using the store.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __len__(self) -> int:
        """Get number of vectors."""
        return self.count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.config.dim}, count={self.count})"


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


class ConnectionError(VectorStoreError):
    """Raised when connection to vector store fails."""
    pass


class IndexError(VectorStoreError):
    """Raised when index operation fails."""
    pass


class SearchError(VectorStoreError):
    """Raised when search operation fails."""
    pass
