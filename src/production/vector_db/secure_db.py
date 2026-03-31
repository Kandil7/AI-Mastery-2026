"""
Secure Vector Database Module
==============================

Vector database with ACL filtering and drift detection.

Combines:
- Vector indexing (core, hnsw)
- ACL-based access control (acl)
- Embedding drift detection (drift)

Classes:
    SecureVectorDB: Vector database with security features

Author: AI-Mastery-2026
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .acl import ACLFilter
from .core import VectorIndex
from .drift import EmbeddingDriftDetector


class VectorDB:
    """Vector database with multiple indexes and advanced features."""

    def __init__(self):
        self.indexes: Dict[str, VectorIndex] = {}
        self.default_index: Optional[str] = None

    def create_index(
        self,
        name: str,
        dim: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
        **kwargs,
    ) -> VectorIndex:
        """Create a new vector index."""
        if index_type == "hnsw":
            from .hnsw import HNSWIndex

            index = HNSWIndex(dim, metric, **kwargs)
        elif index_type == "linear":
            from .core import LinearVectorIndex

            index = LinearVectorIndex(dim, metric)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.indexes[name] = index

        # Set as default if this is the first index
        if self.default_index is None:
            self.default_index = name

        return index

    def get_index(self, name: str = None) -> VectorIndex:
        """Get an index by name or return default."""
        if name is None:
            name = self.default_index

        if name not in self.indexes:
            raise ValueError(f"Index {name} does not exist")

        return self.indexes[name]

    def add(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any] = None,
        index_name: str = None,
    ):
        """Add a vector to the specified index."""
        index = self.get_index(index_name)
        index.add(id, vector, metadata)

    def search(
        self, query: np.ndarray, k: int = 10, index_name: str = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for nearest neighbors in the specified index."""
        index = self.get_index(index_name)
        return index.search(query, k)

    def remove(self, id: str, index_name: str = None):
        """Remove a vector from the specified index."""
        index = self.get_index(index_name)
        index.remove(id)

    def save(self, path: str):
        """Save the entire database to disk."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load the database from disk."""
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


class SecureVectorDB(VectorDB):
    """
    Vector database with ACL filtering and drift detection.

    Extends VectorDB with:
    - Object-level access control
    - Embedding drift monitoring
    - Security audit logging
    """

    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        self.acl_filter = ACLFilter()
        self.drift_detector = EmbeddingDriftDetector(embedding_dim)

    def add_with_acl(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any] = None,
        owner_id: str = None,
        read_users: List[str] = None,
        write_users: List[str] = None,
        public: bool = False,
        index_name: str = None,
    ) -> None:
        """Add vector with ACL metadata."""
        if metadata is None:
            metadata = {}

        # Add ACL to metadata
        if owner_id:
            acl_meta = ACLFilter.create_acl_metadata(
                owner_id, read_users, write_users, public
            )
            metadata.update(acl_meta)

        self.add(id, vector, metadata, index_name)

        # Track for drift detection
        self.drift_detector.add_embedding(vector)

    def search_with_acl(
        self, user_id: str, query: np.ndarray, k: int = 10, index_name: str = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search with ACL filtering."""
        # Get raw results (may contain vectors user can't access)
        raw_results = self.search(query, k * 2, index_name)  # Get extra for filtering

        # Filter by user permissions
        filtered = self.acl_filter.filter_search_results(user_id, raw_results)

        # Track query embedding for drift
        self.drift_detector.add_embedding(query)

        return filtered[:k]

    def get_security_stats(self) -> Dict[str, Any]:
        """Get combined security and drift statistics."""
        return {
            "acl": self.acl_filter.get_security_stats(),
            "drift": self.drift_detector.compute_drift_metrics(),
            "drift_trend": self.drift_detector.get_drift_trend(),
            "drift_alerts": len(self.drift_detector.get_alerts()),
        }
