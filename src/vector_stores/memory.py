"""
In-Memory Vector Store
======================

Simple in-memory vector store for testing and development.

Not suitable for production use due to:
- No persistence (data lost on restart)
- Linear search (O(n) complexity)
- No scalability (all data in RAM)

Use for:
- Unit testing
- Prototyping
- Small datasets (<10k vectors)
"""

import time
from typing import Any, Dict, List, Optional, Callable
import math

from .base import (
    VectorStore,
    VectorStoreConfig,
    SearchResults,
    SearchResult,
    MetricType,
)


class MemoryVectorStore(VectorStore):
    """
    In-memory vector store implementation.

    Uses linear search with configurable distance metric.
    Suitable for testing and small-scale prototyping.

    Example:
        >>> config = VectorStoreConfig(dim=3, metric="cosine")
        >>> store = MemoryVectorStore(config)
        >>> store.initialize()
        >>>
        >>> vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        >>> ids = ["a", "b", "c"]
        >>> store.upsert(vectors, ids)
        >>>
        >>> results = store.search([1, 0, 0], top_k=2)
        >>> print(results[0].id)  # "a"
    """

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._distance_fn = self._get_distance_function(config.metric)

    def _get_distance_function(self, metric: MetricType) -> Callable:
        """Get distance function for metric."""
        if metric == MetricType.COSINE:
            return self._cosine_distance
        elif metric == MetricType.EUCLIDEAN:
            return self._euclidean_distance
        elif metric == MetricType.DOT_PRODUCT:
            return self._dot_product_distance
        elif metric == MetricType.MANHATTAN:
            return self._manhattan_distance
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float:
        """Calculate cosine distance."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return 1.0 - (dot_product / (norm_a * norm_b))

    @staticmethod
    def _euclidean_distance(a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _dot_product_distance(a: List[float], b: List[float]) -> float:
        """Calculate dot product distance."""
        return 1.0 - sum(x * y for x, y in zip(a, b))

    @staticmethod
    def _manhattan_distance(a: List[float], b: List[float]) -> float:
        """Calculate Manhattan distance."""
        return sum(abs(x - y) for x, y in zip(a, b))

    def _distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score."""
        if self.config.metric == MetricType.COSINE:
            return 1.0 - distance
        elif self.config.metric == MetricType.EUCLIDEAN:
            # Convert to similarity using exponential decay
            return math.exp(-distance)
        elif self.config.metric == MetricType.DOT_PRODUCT:
            return 1.0 - distance
        elif self.config.metric == MetricType.MANHATTAN:
            return math.exp(-distance)
        else:
            return 1.0 / (1.0 + distance)

    @property
    def count(self) -> int:
        """Get number of vectors."""
        return len(self._vectors)

    def initialize(self) -> None:
        """Initialize the store."""
        self._is_initialized = True

    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update vectors."""
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have same length")

        for i, (vec, id_) in enumerate(vectors, 0):
            if len(vec) != self.config.dim:
                raise ValueError(
                    f"Vector dimension {len(vec)} doesn't match config {self.config.dim}"
                )

            self._vectors[id_] = vec
            if metadata and i < len(metadata):
                self._metadata[id_] = metadata[i]
            else:
                self._metadata[id_] = {}

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_fn: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ) -> SearchResults:
        """Search for similar vectors."""
        if len(query_vector) != self.config.dim:
            raise ValueError(
                f"Query dimension {len(query_vector)} doesn't match config {self.config.dim}"
            )

        start_time = time.time()

        # Calculate distances for all vectors
        distances = []
        for id_, vec in self._vectors.items():
            # Apply filter if provided
            if filter_fn and not filter_fn(id_, self._metadata.get(id_, {})):
                continue

            distance = self._distance_fn(query_vector, vec)
            score = self._distance_to_score(distance)
            distances.append((id_, score, distance))

        # Sort by score (descending)
        distances.sort(key=lambda x: x[1], reverse=True)

        # Take top_k results
        top_results = distances[:top_k]

        # Build result objects
        results = [
            SearchResult(
                id=id_,
                score=score,
                distance=distance,
                metadata=self._metadata.get(id_, {}),
                vector=self._vectors.get(id_),
            )
            for id_, score, distance in top_results
        ]

        query_time = (time.time() - start_time) * 1000

        return SearchResults(
            results=results,
            query_time_ms=query_time,
            total_count=len(distances),
        )

    def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID."""
        count = 0
        for id_ in ids:
            if id_ in self._vectors:
                del self._vectors[id_]
                del self._metadata[id_]
                count += 1
        return count

    def get(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieve vectors by ID."""
        result = {}
        for id_ in ids:
            if id_ in self._vectors:
                result[id_] = {
                    "vector": self._vectors[id_],
                    "metadata": self._metadata.get(id_, {}),
                }
        return result

    def save(self, path: str) -> None:
        """Save to disk (JSON format)."""
        import json

        data = {
            "config": {
                "dim": self.config.dim,
                "metric": self.config.metric.value,
            },
            "vectors": self._vectors,
            "metadata": self._metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load from disk (JSON format)."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        self._vectors = data["vectors"]
        self._metadata = data["metadata"]
        self._is_initialized = True

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._vectors.clear()
        self._metadata.clear()

    def __repr__(self) -> str:
        return f"MemoryVectorStore(dim={self.config.dim}, count={self.count})"
