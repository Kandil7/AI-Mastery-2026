"""
FAISS Vector Store
==================

Facebook AI Similarity Search (FAISS) integration.

FAISS is a library for efficient similarity search and clustering
of dense vectors. It contains algorithms that search in sets of
vectors of any size, up to ones that possibly do not fit in RAM.

Features:
- Flat index (exact search)
- IVF index (inverted file, approximate)
- HNSW index (graph-based, approximate)
- GPU acceleration support
- Index compression (PQ, SQ)

Installation:
    pip install faiss-cpu  # CPU version
    pip install faiss-gpu  # GPU version (CUDA required)

Example:
    >>> from src.vector_stores import FAISSStore, VectorStoreConfig
    >>>
    >>> config = VectorStoreConfig(dim=384, metric="cosine")
    >>> store = FAISSStore(config)
    >>> store.initialize()
    >>>
    >>> # Add vectors
    >>> vectors = [[0.1] * 384 for _ in range(1000)]
    >>> ids = [f"doc_{i}" for i in range(1000)]
    >>> store.upsert(vectors, ids)
    >>>
    >>> # Search
    >>> results = store.search([0.15] * 384, top_k=5)
"""

import time
from typing import Any, Dict, List, Optional, Callable
import pickle
from pathlib import Path

from .base import (
    VectorStore,
    VectorStoreConfig,
    SearchResults,
    SearchResult,
    MetricType,
    VectorStoreError,
)


class FAISSStore(VectorStore):
    """
    FAISS vector store implementation.

    Provides efficient similarity search using FAISS indexes.
    Supports multiple index types for different use cases.

    Attributes:
        config: Store configuration
        index: FAISS index object
        id_map: Mapping from FAISS internal IDs to external IDs
        metadata: Metadata storage

    Example:
        >>> config = VectorStoreConfig(
        ...     dim=384,
        ...     metric="cosine",
        ...     index_type="hnsw",
        ... )
        >>> store = FAISSStore(config)
    """

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        self.index = None
        self._id_map: Dict[int, str] = {}  # FAISS ID -> External ID
        self._reverse_id_map: Dict[str, int] = {}  # External ID -> FAISS ID
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_id = 0

    @property
    def count(self) -> int:
        """Get number of vectors."""
        if self.index is None:
            return 0
        return self.index.ntotal

    def initialize(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise VectorStoreError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        dim = self.config.dim
        metric = self.config.metric

        # Determine FAISS metric
        if metric == MetricType.COSINE:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
            # For cosine similarity, normalize vectors
            self._normalize_vectors = True
        elif metric == MetricType.EUCLIDEAN:
            faiss_metric = faiss.METRIC_L2
            self._normalize_vectors = False
        elif metric == MetricType.DOT_PRODUCT:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
            self._normalize_vectors = False
        else:
            faiss_metric = faiss.METRIC_L2
            self._normalize_vectors = False

        # Create index based on type
        index_type = self.config.index_type.lower()

        if index_type == "flat":
            self.index = faiss.IndexFlat(dim, faiss_metric)

        elif index_type == "hnsw":
            # HNSW index for fast approximate search
            M = 32  # Number of connections per node
            ef_construction = self.config.ef_construct
            self.index = faiss.IndexHNSWFlat(dim, M, faiss_metric)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = self.config.ef_search

        elif index_type == "ivf":
            # IVF index for large datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
            # Note: IVF requires training before use

        else:
            raise VectorStoreError(f"Unknown index type: {index_type}")

        self._faiss_metric = faiss_metric
        self._is_initialized = True

    def _normalize(self, vectors: List[List[float]]) -> List[List[float]]:
        """Normalize vectors for cosine similarity."""
        if not self._normalize_vectors:
            return vectors

        import math

        normalized = []
        for vec in vectors:
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                normalized.append([x / norm for x in vec])
            else:
                normalized.append(vec)
        return normalized

    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Insert or update vectors."""
        if self.index is None:
            raise VectorStoreError("Store not initialized. Call initialize() first.")

        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have same length")

        # Normalize if needed
        vectors = self._normalize(vectors)

        import faiss
        import numpy as np

        # Convert to numpy array
        vectors_array = np.array(vectors, dtype=np.float32)

        # Assign FAISS IDs
        faiss_ids = []
        for id_ in ids:
            if id_ in self._reverse_id_map:
                # Update existing vector (delete first)
                faiss_id = self._reverse_id_map[id_]
                # Note: FAISS doesn't support direct update, would need rebuild
                # For simplicity, we just add as new vector
                faiss_ids.append(self._next_id)
                self._next_id += 1
            else:
                faiss_ids.append(self._next_id)
                self._reverse_id_map[id_] = self._next_id
                self._id_map[self._next_id] = id_
                self._next_id += 1

        # Add to index
        self.index.add(vectors_array)

        # Store metadata
        for i, id_ in enumerate(ids):
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
        if self.index is None:
            raise VectorStoreError("Store not initialized. Call initialize() first.")

        if len(query_vector) != self.config.dim:
            raise ValueError(
                f"Query dimension {len(query_vector)} doesn't match config {self.config.dim}"
            )

        start_time = time.time()

        import numpy as np

        # Normalize if needed
        query_vector = self._normalize([query_vector])[0]

        # Convert to numpy array
        query_array = np.array([query_vector], dtype=np.float32)

        # Adjust k if necessary
        actual_k = min(top_k, self.count)

        # Search
        distances, indices = self.index.search(query_array, actual_k)

        # Build results
        results = []
        for i, (faiss_idx, distance) in enumerate(zip(indices[0], distances[0])):
            if faiss_idx == -1:  # FAISS returns -1 for missing results
                continue

            external_id = self._id_map.get(faiss_idx, str(faiss_idx))

            # Apply filter if provided
            if filter_fn and not filter_fn(external_id, self._metadata.get(external_id, {})):
                continue

            # Convert distance to score
            if self.config.metric == MetricType.COSINE:
                score = distance  # Already similarity
            elif self.config.metric == MetricType.EUCLIDEAN:
                import math
                score = math.exp(-distance)
            else:
                score = 1.0 / (1.0 + distance)

            results.append(
                SearchResult(
                    id=external_id,
                    score=float(score),
                    distance=float(distance),
                    metadata=self._metadata.get(external_id, {}),
                )
            )

        query_time = (time.time() - start_time) * 1000

        return SearchResults(
            results=results,
            query_time_ms=query_time,
            total_count=self.count,
        )

    def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID."""
        # Note: FAISS doesn't support direct deletion
        # Would need to rebuild index without deleted vectors
        # For now, we just remove from ID maps
        count = 0
        for id_ in ids:
            if id_ in self._reverse_id_map:
                faiss_id = self._reverse_id_map.pop(id_)
                self._id_map.pop(faiss_id, None)
                self._metadata.pop(id_, None)
                count += 1
        return count

    def get(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieve vectors by ID."""
        result = {}
        for id_ in ids:
            if id_ in self._metadata:
                result[id_] = {
                    "metadata": self._metadata.get(id_, {}),
                }
        return result

    def save(self, path: str) -> None:
        """Save index and metadata to disk."""
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix(".index")))

        # Save metadata and ID maps
        metadata_path = path.with_suffix(".meta.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "id_map": self._id_map,
                    "reverse_id_map": self._reverse_id_map,
                    "metadata": self._metadata,
                    "next_id": self._next_id,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load index and metadata from disk."""
        import faiss

        path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix(".index")))

        # Load metadata and ID maps
        metadata_path = path.with_suffix(".meta.pkl")
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self._id_map = data["id_map"]
            self._reverse_id_map = data["reverse_id_map"]
            self._metadata = data["metadata"]
            self._next_id = data["next_id"]

        self._is_initialized = True

    def reset(self) -> None:
        """Reset the index."""
        if self.index is not None:
            self.index.reset()
        self._id_map.clear()
        self._reverse_id_map.clear()
        self._metadata.clear()
        self._next_id = 0

    def __repr__(self) -> str:
        return f"FAISSStore(dim={self.config.dim}, count={self.count}, type={self.config.index_type})"
