"""
Core Vector Index Module
========================

Base classes for vector indexing and search.

Classes:
    VectorItem: Represents an item with vector embedding and metadata
    VectorIndex: Base class for vector indexing
    LinearVectorIndex: Simple linear search index for small datasets

Author: AI-Mastery-2026
"""

import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VectorItem:
    """Represents an item with vector embedding and metadata."""

    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    created_at: float


class VectorIndex:
    """Base class for vector indexing."""

    def __init__(self, dim: int, metric: str = "cosine"):
        """
        Initialize vector index.

        Args:
            dim: Dimension of vectors
            metric: Distance metric ('cosine', 'euclidean', 'dot')
        """
        self.dim = dim
        self.metric = metric
        self.items: Dict[str, VectorItem] = {}

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a vector to the index."""
        if metadata is None:
            metadata = {}

        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} does not match index dimension {self.dim}"
            )

        # Normalize vector for cosine similarity
        if self.metric == "cosine":
            vector = vector / (np.linalg.norm(vector) + 1e-12)

        self.items[id] = VectorItem(
            id=id, vector=vector, metadata=metadata, created_at=time.time()
        )

    def remove(self, id: str):
        """Remove a vector from the index."""
        if id in self.items:
            del self.items[id]

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate similarity between two vectors."""
        if self.metric == "cosine":
            return np.dot(v1, v2)  # Already normalized
        elif self.metric == "euclidean":
            return -np.linalg.norm(
                v1 - v2
            )  # Negative for similarity (higher is more similar)
        elif self.metric == "dot":
            return np.dot(v1, v2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for k nearest neighbors."""
        raise NotImplementedError("Subclasses must implement search method")

    def save(self, path: str):
        """Save the index to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load the index from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class LinearVectorIndex(VectorIndex):
    """Simple linear search index for small datasets."""

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Linear search for k nearest neighbors."""
        if self.metric == "cosine":
            query = query / (np.linalg.norm(query) + 1e-12)

        # Calculate similarities with all vectors
        similarities = []
        for item_id, item in self.items.items():
            sim = self.similarity(query, item.vector)
            similarities.append((item_id, sim, item.metadata))

        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
