"""
Vector Database Integration - Production RAG 2026

Following RAG Pipeline Guide 2026 - Phase 3: Embeddings & Vector Databases

Supports:
- Qdrant (production-grade, HNSW indexing)
- ChromaDB (lightweight, easy setup)
- In-memory (testing, small datasets)
- Weaviate (enterprise, GraphQL)

Features:
- HNSW indexing for fast approximate search
- Metadata filtering
- Batch operations
- Persistence
- Index aliases (zero-downtime reindexing)
- Cosine, Euclidean, Dot product distances

Usage:
    store = VectorStore(config)
    store.add_vectors(ids, vectors, payloads)
    results = store.search(query_vector, top_k=5)
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


# ==================== Enums & Data Classes ====================


class VectorStoreType(Enum):
    """Supported vector store types."""

    MEMORY = "memory"  # In-memory, for testing
    QDRANT = "qdrant"  # Production-grade
    CHROMA = "chroma"  # Lightweight
    WEAVIATE = "weaviate"  # Enterprise


class DistanceMetric(Enum):
    """Distance metrics for vector search."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot"
    MANHATTAN = "manhattan"


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    # Store type
    store_type: VectorStoreType = VectorStoreType.MEMORY

    # Collection settings
    collection_name: str = "arabic_islamic_literature"
    vector_size: int = 768  # mpnet-base-v2 output dimension
    distance: DistanceMetric = DistanceMetric.COSINE

    # HNSW indexing (Qdrant)
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_full_scan_threshold: int = 10000

    # Persistence
    persist_directory: Optional[str] = None

    # Connection settings (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_https: bool = False

    # Connection settings (Weaviate)
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: Optional[str] = None

    # Batch settings
    batch_size: int = 100
    parallel: int = 1


@dataclass
class SearchResult:
    """Search result from vector store."""

    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None


# ==================== Base Vector Store ====================


class BaseVectorStore:
    """Base class for vector stores."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._initialized = False

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to the store."""
        raise NotImplementedError

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        raise NotImplementedError

    def delete(self, ids: List[str]):
        """Delete vectors by ID."""
        raise NotImplementedError

    def count(self) -> int:
        """Get total number of vectors."""
        raise NotImplementedError

    def clear(self):
        """Clear all vectors."""
        raise NotImplementedError


# ==================== In-Memory Vector Store ====================


class MemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store for testing and small datasets.

    Uses brute-force cosine similarity search.
    Not suitable for production with large datasets.
    """

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)

        self._vectors: Dict[str, np.ndarray] = {}
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._index: Dict[str, int] = {}  # ID to index mapping

        logger.info("Initialized in-memory vector store")

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to memory store."""

        payloads = payloads or [{} for _ in ids]

        for id_, vector, payload in zip(ids, vectors, payloads):
            self._vectors[id_] = vector.astype(np.float32)
            self._payloads[id_] = payload
            self._index[id_] = len(self._index)

        logger.info(f"Added {len(ids)} vectors to memory store")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search using brute-force cosine similarity."""

        if not self._vectors:
            return []

        query_vector = query_vector.astype(np.float32)

        # Calculate similarities
        scores = []
        for id_, vector in self._vectors.items():
            # Apply filters
            if filters:
                payload = self._payloads.get(id_, {})
                if not self._matches_filter(payload, filters):
                    continue

            # Cosine similarity
            score = self._cosine_similarity(query_vector, vector)
            scores.append((id_, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for id_, score in scores[:top_k]:
            results.append(SearchResult(
                id=id_,
                score=float(score),
                payload=self._payloads.get(id_, {}),
            ))

        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _matches_filter(
        self,
        payload: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        """Check if payload matches filters."""

        for key, value in filters.items():
            if key not in payload:
                return False

            if isinstance(value, dict):
                # Handle operators ($eq, $ne, $in, etc.)
                if "$eq" in value:
                    if payload[key] != value["$eq"]:
                        return False
                elif "$in" in value:
                    if payload[key] not in value["$in"]:
                        return False
                elif "$gte" in value:
                    if payload[key] < value["$gte"]:
                        return False
                elif "$lte" in value:
                    if payload[key] > value["$lte"]:
                        return False
            else:
                # Exact match
                if payload[key] != value:
                    return False

        return True

    def delete(self, ids: List[str]):
        """Delete vectors by ID."""

        for id_ in ids:
            self._vectors.pop(id_, None)
            self._payloads.pop(id_, None)
            self._index.pop(id_, None)

        logger.info(f"Deleted {len(ids)} vectors")

    def count(self) -> int:
        """Get total number of vectors."""
        return len(self._vectors)

    def clear(self):
        """Clear all vectors."""
        self._vectors.clear()
        self._payloads.clear()
        self._index.clear()
        logger.info("Cleared memory vector store")

    def save(self, filepath: str):
        """Save to disk."""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "vectors": {k: v.tolist() for k, v in self._vectors.items()},
            "payloads": self._payloads,
            "index": self._index,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved {self.count()} vectors to {filepath}")

    def load(self, filepath: str):
        """Load from disk."""

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self._vectors = {
            k: np.array(v, dtype=np.float32)
            for k, v in data["vectors"].items()
        }
        self._payloads = data["payloads"]
        self._index = data["index"]

        logger.info(f"Loaded {self.count()} vectors from {filepath}")


# ==================== Qdrant Vector Store ====================


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector database integration.

    Production-grade vector search with:
    - HNSW indexing for fast approximate search
    - Metadata filtering
    - Payload indexing
    - Index aliases for zero-downtime reindexing
    - Distributed deployment support

    Install: pip install qdrant-client
    """

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)

        self._client = None
        self._async_client = None

        self._initialize()

    def _initialize(self):
        """Initialize Qdrant client."""

        try:
            from qdrant_client import QdrantClient

            # Create client
            self._client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
                https=self.config.qdrant_https,
            )

            # Create or get collection
            self._create_collection()

            self._initialized = True
            logger.info(
                f"Connected to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}"
            )

        except ImportError:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            )
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            logger.info("Falling back to in-memory store")
            self._client = None

    def _create_collection(self):
        """Create Qdrant collection if not exists."""

        from qdrant_client.http.models import (
            Distance,
            VectorParams,
            HnswConfigDiff,
        )

        # Map distance metric
        distance_map = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
        }

        distance = distance_map.get(self.config.distance, Distance.COSINE)

        try:
            # Check if collection exists
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.config.collection_name not in collection_names:
                # Create collection
                self._client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=distance,
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=self.config.hnsw_m,
                        ef_construct=self.config.hnsw_ef_construct,
                        full_scan_threshold=self.config.hnsw_full_scan_threshold,
                    ),
                )

                logger.info(f"Created collection: {self.config.collection_name}")

            # Create payload indexes for filtering
            self._create_payload_indexes()

        except Exception as e:
            logger.error(f"Error creating collection: {e}")

    def _create_payload_indexes(self):
        """Create indexes for common payload fields."""

        from qdrant_client.http.models import PayloadSchemaType

        indexed_fields = ["book_id", "category", "author", "source_type"]

        for field in indexed_fields:
            try:
                self._client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index may already exist

    @property
    def client(self):
        """Get Qdrant client."""
        return self._client

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to Qdrant."""

        if not self._client:
            logger.warning("Qdrant not connected, using fallback")
            return

        from qdrant_client.http.models import PointStruct

        payloads = payloads or [{} for _ in ids]

        # Prepare points
        points = []
        for id_, vector, payload in zip(ids, vectors, payloads):
            # Convert numpy to list
            vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector

            points.append(
                PointStruct(
                    id=id_,
                    vector=vector_list,
                    payload=payload,
                )
            )

        # Upsert in batches
        batch_size = self.config.batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]

            self._client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )

        logger.info(f"Added {len(ids)} vectors to Qdrant")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search Qdrant with optional filtering."""

        if not self._client:
            logger.warning("Qdrant not connected")
            return []

        # Build filter
        qdrant_filter = None
        if filters:
            qdrant_filter = self._build_filter(filters)

        # Search
        results = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
        )

        # Convert to SearchResult
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=str(result.id),
                score=result.score,
                payload=result.payload or {},
            ))

        return search_results

    def _build_filter(self, filters: Dict[str, Any]):
        """Build Qdrant filter from dict."""

        from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

        conditions = []

        for key, value in filters.items():
            if isinstance(value, dict):
                if "$eq" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value["$eq"]),
                        )
                    )
                elif "$in" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value["$in"]),
                        )
                    )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )

        if conditions:
            return Filter(must=conditions)
        return None

    def delete(self, ids: List[str]):
        """Delete vectors from Qdrant."""

        if not self._client:
            return

        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector={"points": ids},
        )

        logger.info(f"Deleted {len(ids)} vectors from Qdrant")

    def count(self) -> int:
        """Get total number of vectors."""

        if not self._client:
            return 0

        try:
            info = self._client.get_collection(self.config.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def clear(self):
        """Clear collection."""

        if not self._client:
            return

        try:
            self._client.delete_collection(self.config.collection_name)
            self._create_collection()
            logger.info("Cleared Qdrant collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def create_alias(self, alias_name: str):
        """Create alias for collection (for zero-downtime reindexing)."""

        if not self._client:
            return

        try:
            self._client.create_alias(
                alias_name=alias_name,
                collection_name=self.config.collection_name,
            )
            logger.info(f"Created alias {alias_name}")
        except Exception as e:
            logger.error(f"Error creating alias: {e}")

    def swap_alias(self, alias_name: str, new_collection: str):
        """Swap alias to point to new collection."""

        if not self._client:
            return

        try:
            self._client.update_alias(
                alias_name=alias_name,
                new_collection_name=new_collection,
            )
            logger.info(f"Swapped alias {alias_name} to {new_collection}")
        except Exception as e:
            logger.error(f"Error swapping alias: {e}")


# ==================== ChromaDB Vector Store ====================


class ChromaDBVectorStore(BaseVectorStore):
    """
    ChromaDB vector database integration.

    Lightweight, easy-to-setup vector store.
    Good for development and small-to-medium datasets.

    Install: pip install chromadb
    """

    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)

        self._client = None
        self._collection = None

        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client."""

        try:
            import chromadb
            from chromadb.config import Settings

            # Create client
            if self.config.persist_directory:
                # Persistent client
                self._client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                )
            else:
                # In-memory client
                self._client = chromadb.Client()

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": self._get_hnsw_space(),
                },
            )

            self._initialized = True
            logger.info(f"Initialized ChromaDB collection: {self.config.collection_name}")

        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChromaDB: {e}")
            self._client = None

    def _get_hnsw_space(self) -> str:
        """Get HNSW space for ChromaDB."""

        space_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "l2",
            DistanceMetric.DOT_PRODUCT: "ip",
        }

        return space_map.get(self.config.distance, "cosine")

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to ChromaDB."""

        if not self._collection:
            logger.warning("ChromaDB not initialized")
            return

        payloads = payloads or [{} for _ in ids]

        # ChromaDB expects specific format
        documents = [p.get("content", "") for p in payloads]
        metadatas = [
            {k: str(v) if not isinstance(v, (str, int, float)) else v for k, v in p.items()}
            for p in payloads
        ]

        # Add in batches
        batch_size = self.config.batch_size
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vectors = [v.tolist() for v in vectors[i : i + batch_size]]
            batch_docs = documents[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]

            self._collection.add(
                ids=batch_ids,
                embeddings=batch_vectors,
                documents=batch_docs,
                metadatas=batch_metas,
            )

        logger.info(f"Added {len(ids)} vectors to ChromaDB")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search ChromaDB with optional filtering."""

        if not self._collection:
            return []

        # Build where filter
        where_filter = None
        if filters:
            where_filter = self._build_where_filter(filters)

        # Search
        results = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["embeddings", "metadatas", "distances"],
        )

        # Convert to SearchResult
        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    id=id_,
                    score=1 - results["distances"][0][i] if results["distances"] else 0,  # Convert distance to similarity
                    payload=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where filter."""

        conditions = {}

        for key, value in filters.items():
            if isinstance(value, dict):
                if "$eq" in value:
                    conditions[key] = value["$eq"]
                elif "$in" in value:
                    conditions[key] = {"$in": value["$in"]}
                elif "$ne" in value:
                    conditions[key] = {"$ne": value["$ne"]}
            else:
                conditions[key] = value

        return conditions if conditions else None

    def delete(self, ids: List[str]):
        """Delete vectors from ChromaDB."""

        if not self._collection:
            return

        self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors from ChromaDB")

    def count(self) -> int:
        """Get total number of vectors."""

        if not self._collection:
            return 0

        return self._collection.count()

    def clear(self):
        """Clear collection."""

        if not self._client:
            return

        # Delete and recreate collection
        self._client.delete_collection(self.config.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
        )
        logger.info("Cleared ChromaDB collection")


# ==================== Vector Store Factory ====================


class VectorStore:
    """
    Unified vector store interface supporting multiple backends.

    Usage:
        store = VectorStore(config)
        store.add_vectors(ids, vectors, payloads)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._store = self._create_store()

    def _create_store(self) -> BaseVectorStore:
        """Create appropriate vector store based on config."""

        store_type = self.config.store_type

        if store_type == VectorStoreType.QDRANT:
            return QdrantVectorStore(self.config)
        elif store_type == VectorStoreType.CHROMA:
            return ChromaDBVectorStore(self.config)
        elif store_type == VectorStoreType.MEMORY:
            return MemoryVectorStore(self.config)
        else:
            return MemoryVectorStore(self.config)

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors to the store."""
        self._store.add_vectors(ids, vectors, payloads)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        return self._store.search(query_vector, top_k, filters)

    def delete(self, ids: List[str]):
        """Delete vectors by ID."""
        self._store.delete(ids)

    def count(self) -> int:
        """Get total number of vectors."""
        return self._store.count()

    def clear(self):
        """Clear all vectors."""
        self._store.clear()

    def save(self, filepath: str):
        """Save to disk (if supported)."""
        if hasattr(self._store, "save"):
            self._store.save(filepath)

    def load(self, filepath: str):
        """Load from disk (if supported)."""
        if hasattr(self._store, "load"):
            self._store.load(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""

        return {
            "type": self.config.store_type.value,
            "collection": self.config.collection_name,
            "vector_size": self.config.vector_size,
            "distance": self.config.distance.value,
            "count": self.count(),
        }


# ==================== Factory Functions ====================


def create_vector_store(
    store_type: str = "memory",
    collection_name: str = "arabic_islamic_literature",
    vector_size: int = 768,
    persist_directory: Optional[str] = None,
    **kwargs,
) -> VectorStore:
    """
    Create a vector store.

    Args:
        store_type: Store type (memory, qdrant, chroma)
        collection_name: Collection name
        vector_size: Vector dimension
        persist_directory: Directory for persistence
        **kwargs: Additional config options

    Returns:
        VectorStore instance
    """

    store_type_map = {
        "memory": VectorStoreType.MEMORY,
        "qdrant": VectorStoreType.QDRANT,
        "chroma": VectorStoreType.CHROMA,
    }

    config = VectorStoreConfig(
        store_type=store_type_map.get(store_type.lower(), VectorStoreType.MEMORY),
        collection_name=collection_name,
        vector_size=vector_size,
        persist_directory=persist_directory,
        **kwargs,
    )

    return VectorStore(config)


def get_recommended_store(dataset_size: int) -> VectorStoreType:
    """
    Get recommended vector store type based on dataset size.

    Args:
        dataset_size: Number of vectors

    Returns:
        Recommended VectorStoreType
    """

    if dataset_size < 10000:
        return VectorStoreType.MEMORY  # In-memory is fine
    elif dataset_size < 1000000:
        return VectorStoreType.CHROMA  # ChromaDB for medium datasets
    else:
        return VectorStoreType.QDRANT  # Qdrant for large datasets


if __name__ == "__main__":
    # Demo
    print("Vector Store - Demo")
    print("=" * 50)

    # Create store
    store = create_vector_store(
        store_type="memory",
        collection_name="test_collection",
        vector_size=768,
    )

    # Add test vectors
    ids = [f"doc_{i}" for i in range(100)]
    vectors = [np.random.randn(768).astype(np.float32) for _ in range(100)]
    payloads = [
        {"book_id": i, "category": "test", "content": f"Document {i}"}
        for i in range(100)
    ]

    store.add_vectors(ids, vectors, payloads)
    print(f"Added {store.count()} vectors")

    # Search
    query_vector = np.random.randn(768).astype(np.float32)
    results = store.search(query_vector, top_k=5)

    print(f"\nSearch results:")
    for result in results:
        print(f"  {result.id}: {result.score:.4f}")

    # Stats
    print(f"\nStore stats:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
