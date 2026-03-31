"""
Vector Database Module

Production-ready vector database integrations:
- Qdrant: Production vector database with filtering
- FAISS: Facebook AI Similarity Search
- Chroma: Lightweight embedded vector store
- Pinecone: Managed vector database

Features:
- Hybrid search (dense + sparse)
- Metadata filtering
- Index management
- Batch operations
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """A record in the vector database."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
            "payload": self.payload,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorRecord":
        return cls(
            id=data["id"],
            vector=data["vector"],
            metadata=data.get("metadata", {}),
            payload=data.get("payload"),
            score=data.get("score"),
        )


@dataclass
class SearchRequest:
    """Search request configuration."""

    query_vector: List[float]
    filter: Optional[Dict[str, Any]] = None
    top_k: int = 10
    include_vectors: bool = False
    include_metadata: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_vector": self.query_vector,
            "filter": self.filter,
            "top_k": self.top_k,
            "include_vectors": self.include_vectors,
            "include_metadata": self.include_metadata,
        }


@dataclass
class SearchResult:
    """Search result with scored vectors."""

    records: List[VectorRecord]
    total: int
    latency_ms: float
    query_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "total": self.total,
            "latency_ms": self.latency_ms,
        }


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    dense_weight: float = 0.7  # Weight for dense retrieval
    sparse_weight: float = 0.3  # Weight for sparse retrieval
    rrf_k: int = 60  # Reciprocal Rank Fusion parameter
    use_reranking: bool = True
    rerank_top_k: int = 50


class VectorDatabase(ABC):
    """Abstract base class for vector databases."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        pass

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any,
    ) -> None:
        """Create a new collection."""
        pass

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
    ) -> int:
        """Upsert records into collection."""
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> int:
        """Delete records by ID."""
        pass

    @abstractmethod
    async def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """Get collection information."""
        pass


class QdrantClient(VectorDatabase):
    """
    Qdrant vector database client.

    Features:
    - Full-text search with filtering
    - Payload-based filtering
    - HNSW index optimization
    - Distributed deployment support
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        timeout: int = 30,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout

        self._client = None
        self._async_client = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Qdrant."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qdrant_models

            self._qdrant_models = qdrant_models

            if self.url:
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            else:
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    api_key=self.api_key,
                    timeout=self.timeout,
                )

            self._async_client = self._client
            self._connected = True

            logger.info(f"Connected to Qdrant at {self.url or f'{self.host}:{self.port}'}")
        except ImportError:
            raise RuntimeError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Qdrant."""
        if self._client:
            self._client = None
            self._connected = False
            logger.info("Disconnected from Qdrant")

    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "cosine",
        hnsw_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new collection."""
        if not self._connected:
            await self.connect()

        distance_map = {
            "cosine": self._qdrant_models.Distance.COSINE,
            "euclidean": self._qdrant_models.Distance.EUCLID,
            "dot": self._qdrant_models.Distance.DOT,
        }

        vectors_config = self._qdrant_models.VectorParams(
            size=dimension,
            distance=distance_map.get(distance.lower(), self._qdrant_models.Distance.COSINE),
        )

        # Add HNSW config if provided
        if hnsw_config:
            vectors_config.hnsw_config = self._qdrant_models.HnswConfigDiff(**hnsw_config)

        try:
            self._client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                **kwargs,
            )
            logger.info(f"Created collection: {name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Collection {name} already exists")
            else:
                raise

    async def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
    ) -> int:
        """Upsert records into Qdrant."""
        if not self._connected:
            await self.connect()

        points = []
        for record in records:
            point = self._qdrant_models.PointStruct(
                id=record.id,
                vector=record.vector,
                payload={**record.metadata, **(record.payload or {})},
            )
            points.append(point)

        result = self._client.upsert(
            collection_name=collection,
            points=points,
        )

        logger.info(f"Upserted {len(records)} records to {collection}")
        return len(records)

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for similar vectors."""
        import time
        start_time = time.time()

        if not self._connected:
            await self.connect()

        # Convert filter to Qdrant format
        qdrant_filter = None
        if filter:
            qdrant_filter = self._build_filter(filter)

        results = self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vector=False,
        )

        records = []
        for hit in results:
            payload = hit.payload or {}
            record = VectorRecord(
                id=str(hit.id),
                vector=hit.vector if hasattr(hit, 'vector') else [],
                metadata=payload,
                score=hit.score,
            )
            records.append(record)

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            records=records,
            total=len(records),
            latency_ms=latency_ms,
            query_vector=query_vector,
        )

    def _build_filter(self, filter_dict: Dict[str, Any]) -> Any:
        """Build Qdrant filter from dict."""
        conditions = []

        for key, value in filter_dict.items():
            if isinstance(value, dict):
                # Handle operators
                if "$eq" in value:
                    conditions.append(
                        self._qdrant_models.FieldCondition(
                            key=key,
                            match=self._qdrant_models.MatchValue(value=value["$eq"]),
                        )
                    )
                elif "$in" in value:
                    conditions.append(
                        self._qdrant_models.FieldCondition(
                            key=key,
                            match=self._qdrant_models.MatchAny(any=value["$in"]),
                        )
                    )
                elif "$gte" in value:
                    conditions.append(
                        self._qdrant_models.FieldCondition(
                            key=key,
                            range=self._qdrant_models.Range(gte=value["$gte"]),
                        )
                    )
                elif "$lte" in value:
                    conditions.append(
                        self._qdrant_models.FieldCondition(
                            key=key,
                            range=self._qdrant_models.Range(lte=value["$lte"]),
                        )
                    )
            else:
                # Simple equality
                conditions.append(
                    self._qdrant_models.FieldCondition(
                        key=key,
                        match=self._qdrant_models.MatchValue(value=value),
                    )
                )

        if conditions:
            return self._qdrant_models.Filter(must=conditions)
        return None

    async def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> int:
        """Delete records by ID."""
        if not self._connected:
            await self.connect()

        self._client.delete(
            collection_name=collection,
            points_selector=self._qdrant_models.PointIdsList(points=ids),
        )

        logger.info(f"Deleted {len(ids)} records from {collection}")
        return len(ids)

    async def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """Get collection information."""
        if not self._connected:
            await self.connect()

        info = self._client.get_collection(collection_name=collection)

        return {
            "name": collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "config": info.config.dict() if hasattr(info.config, 'dict') else str(info.config),
        }

    async def hybrid_search(
        self,
        collection: str,
        query_vector: List[float],
        query_text: Optional[str] = None,
        top_k: int = 10,
        config: Optional[HybridSearchConfig] = None,
    ) -> SearchResult:
        """Perform hybrid search with dense and sparse retrieval."""
        import time
        start_time = time.time()

        config = config or HybridSearchConfig()

        # Dense search
        dense_results = await self.search(collection, query_vector, top_k=config.rerank_top_k)

        # If query text provided, do full-text search (if configured)
        sparse_results = []
        if query_text and self._has_full_text_index(collection):
            sparse_results = await self._full_text_search(collection, query_text, config.rerank_top_k)

        # Fuse results
        fused = self._reciprocal_rank_fusion(
            dense_results.records,
            sparse_results,
            k=config.rrf_k,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        )

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            records=fused[:top_k],
            total=len(fused),
            latency_ms=latency_ms,
        )

    def _has_full_text_index(self, collection: str) -> bool:
        """Check if collection has full-text index."""
        # Implementation depends on Qdrant configuration
        return False

    async def _full_text_search(
        self,
        collection: str,
        query: str,
        top_k: int,
    ) -> List[VectorRecord]:
        """Full-text search."""
        # Placeholder for full-text search
        return []

    def _reciprocal_rank_fusion(
        self,
        dense: List[VectorRecord],
        sparse: List[VectorRecord],
        k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> List[VectorRecord]:
        """Fuse results using Reciprocal Rank Fusion."""
        scores: Dict[str, Tuple[float, VectorRecord]] = {}

        # Score dense results
        for i, record in enumerate(dense):
            rank = i + 1
            score = dense_weight / (k + rank)
            scores[record.id] = (score, record)

        # Score sparse results
        for i, record in enumerate(sparse):
            rank = i + 1
            score = sparse_weight / (k + rank)

            if record.id in scores:
                existing_score, existing_record = scores[record.id]
                scores[record.id] = (existing_score + score, existing_record)
            else:
                scores[record.id] = (score, record)

        # Sort by combined score
        sorted_results = sorted(scores.values(), key=lambda x: x[0], reverse=True)

        # Update scores
        for score, record in sorted_results:
            record.score = score

        return [record for _, record in sorted_results]


class FAISSClient(VectorDatabase):
    """
    FAISS (Facebook AI Similarity Search) client.

    Features:
    - In-memory vector index
    - GPU acceleration support
    - Multiple index types
    - Quantization options
    """

    def __init__(
        self,
        index_type: str = "Flat",
        dimension: int = 384,
        use_gpu: bool = False,
        persist_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.index_type = index_type
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.persist_path = Path(persist_path) if persist_path else None

        self._index = None
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._id_mapping: Dict[int, str] = {}
        self._reverse_mapping: Dict[str, int] = {}
        self._next_id = 0

        self._load_index()

    def _load_index(self) -> None:
        """Load or create FAISS index."""
        try:
            import faiss

            self._faiss = faiss

            if self.use_gpu:
                try:
                    self._res = faiss.StandardGpuResources()
                except Exception:
                    logger.warning("GPU not available, using CPU")
                    self.use_gpu = False

            if self.persist_path and self.persist_path.exists():
                self._index = faiss.read_index(str(self.persist_path))
                logger.info(f"Loaded FAISS index from {self.persist_path}")
            else:
                self._create_index()
        except ImportError:
            raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")

    def _create_index(self) -> None:
        """Create new FAISS index."""
        if self.index_type == "Flat":
            self._index = self._faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IP":
            self._index = self._faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "HNSW":
            self._index = self._faiss.IndexHNSWFlat(self.dimension, 32)
        elif self.index_type == "IVF":
            quantizer = self._faiss.IndexFlatL2(self.dimension)
            self._index = self._faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            self._index = self._faiss.IndexFlatL2(self.dimension)

        if self.use_gpu and hasattr(self, '_res'):
            self._index = self._faiss.index_cpu_to_gpu(self._res, 0, self._index)

        logger.info(f"Created FAISS index: {self.index_type}")

    async def connect(self) -> None:
        """Connect (initialize) FAISS."""
        if not self._index:
            self._load_index()

    async def disconnect(self) -> None:
        """Disconnect (save) FAISS."""
        self._save_index()

    def _save_index(self) -> None:
        """Save index to disk."""
        if self.persist_path and self._index:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._faiss.write_index(self._index, str(self.persist_path))
            logger.info(f"Saved FAISS index to {self.persist_path}")

    async def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any,
    ) -> None:
        """Create a new collection (index)."""
        self.dimension = dimension
        self._create_index()

    async def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
    ) -> int:
        """Upsert records into FAISS."""
        if not self._index:
            await self.connect()

        vectors = []
        for record in records:
            # Add to mapping
            if record.id not in self._reverse_mapping:
                self._reverse_mapping[record.id] = self._next_id
                self._id_mapping[self._next_id] = record.id
                self._next_id += 1

            vectors.append(record.vector)
            self._metadata[record.id] = {**record.metadata, **(record.payload or {})}

        # Add to index
        import numpy as np
        vectors_array = np.array(vectors, dtype=np.float32)
        self._index.add(vectors_array)

        logger.info(f"Upserted {len(records)} records to FAISS")
        return len(records)

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for similar vectors."""
        import time
        import numpy as np
        start_time = time.time()

        if not self._index:
            await self.connect()

        # Adjust top_k if necessary
        actual_k = min(top_k, self._index.ntotal)

        if actual_k == 0:
            return SearchResult(records=[], total=0, latency_ms=0)

        # Search
        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = self._index.search(query_array, actual_k)

        records = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            record_id = self._id_mapping.get(idx)
            if record_id:
                metadata = self._metadata.get(record_id, {})

                # Apply filter if provided
                if filter and not self._matches_filter(metadata, filter):
                    continue

                # Convert distance to similarity score
                score = 1 / (1 + dist) if dist >= 0 else 0

                record = VectorRecord(
                    id=record_id,
                    vector=[],
                    metadata=metadata,
                    score=score,
                )
                records.append(record)

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            records=records,
            total=len(records),
            latency_ms=latency_ms,
        )

    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter_dict: Dict[str, Any],
    ) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            if isinstance(value, dict):
                if "$eq" in value and metadata[key] != value["$eq"]:
                    return False
                if "$in" in value and metadata[key] not in value["$in"]:
                    return False
                if "$gte" in value and metadata[key] < value["$gte"]:
                    return False
                if "$lte" in value and metadata[key] > value["$lte"]:
                    return False
            elif metadata[key] != value:
                return False

        return True

    async def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> int:
        """Delete records by ID."""
        # FAISS doesn't support direct deletion
        # We need to rebuild the index without deleted IDs
        deleted = 0

        for record_id in ids:
            if record_id in self._metadata:
                del self._metadata[record_id]
                deleted += 1

            if record_id in self._reverse_mapping:
                internal_id = self._reverse_mapping.pop(record_id)
                del self._id_mapping[internal_id]

        logger.info(f"Marked {deleted} records for deletion")
        return deleted

    async def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """Get collection information."""
        return {
            "name": collection,
            "vectors_count": self._index.ntotal if self._index else 0,
            "dimension": self.dimension,
            "index_type": self.index_type,
        }


class ChromaClient(VectorDatabase):
    """
    Chroma vector database client.

    Features:
    - Embedded or client-server mode
    - Full-text search
    - Metadata filtering
    - Simple API
    """

    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        self.persist_directory = str(persist_directory) if persist_directory else None
        self.host = host
        self.port = port

        self._client = None
        self._collections: Dict[str, Any] = {}

    async def connect(self) -> None:
        """Connect to Chroma."""
        try:
            import chromadb
            from chromadb.config import Settings

            if self.host and self.port:
                # Client-server mode
                self._client = chromadb.HttpClient(host=self.host, port=self.port)
            else:
                # Persistent or in-memory
                settings = Settings()
                if self.persist_directory:
                    settings.persist_directory = self.persist_directory
                    settings.is_persistent = True

                self._client = chromadb.Client(settings)

            logger.info("Connected to Chroma")
        except ImportError:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")

    async def disconnect(self) -> None:
        """Disconnect from Chroma."""
        self._client = None
        logger.info("Disconnected from Chroma")

    async def create_collection(
        self,
        name: str,
        dimension: int = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new collection."""
        if not self._client:
            await self.connect()

        # Chroma auto-detects dimension from embeddings
        collection_metadata = metadata or {}
        if dimension:
            collection_metadata["dimension"] = dimension

        self._client.create_collection(
            name=name,
            metadata=collection_metadata,
            **kwargs,
        )

        logger.info(f"Created Chroma collection: {name}")

    async def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
    ) -> int:
        """Upsert records into Chroma."""
        if not self._client:
            await self.connect()

        chroma_collection = self._client.get_or_create_collection(collection)

        ids = [record.id for record in records]
        embeddings = [record.vector for record in records]
        metadatas = [{**record.metadata, **(record.payload or {})} for record in records]

        chroma_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Upserted {len(records)} records to Chroma collection: {collection}")
        return len(records)

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for similar vectors."""
        import time
        start_time = time.time()

        if not self._client:
            await self.connect()

        chroma_collection = self._client.get_or_create_collection(collection)

        # Convert filter to Chroma format
        where = None
        if filter:
            where = self._build_chroma_filter(filter)

        results = chroma_collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )

        records = []
        if results["ids"] and results["ids"][0]:
            for i, record_id in enumerate(results["ids"][0]):
                record = VectorRecord(
                    id=record_id,
                    vector=[],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=1 / (1 + results["distances"][0][i]) if results["distances"] else 0,
                )
                records.append(record)

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            records=records,
            total=len(records),
            latency_ms=latency_ms,
        )

    def _build_chroma_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build Chroma where clause from filter dict."""
        where = {}

        for key, value in filter_dict.items():
            if isinstance(value, dict):
                if "$eq" in value:
                    where[key] = {"$eq": value["$eq"]}
                elif "$ne" in value:
                    where[key] = {"$ne": value["$ne"]}
                elif "$in" in value:
                    where[key] = {"$in": value["$in"]}
                elif "$nin" in value:
                    where[key] = {"$nin": value["$nin"]}
                elif "$gte" in value:
                    where[key] = {"$gte": value["$gte"]}
                elif "$gt" in value:
                    where[key] = {"$gt": value["$gt"]}
                elif "$lte" in value:
                    where[key] = {"$lte": value["$lte"]}
                elif "$lt" in value:
                    where[key] = {"$lt": value["$lt"]}
            else:
                where[key] = value

        return where

    async def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> int:
        """Delete records by ID."""
        if not self._client:
            await self.connect()

        chroma_collection = self._client.get_or_create_collection(collection)
        chroma_collection.delete(ids=ids)

        logger.info(f"Deleted {len(ids)} records from Chroma collection: {collection}")
        return len(ids)

    async def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """Get collection information."""
        if not self._client:
            await self.connect()

        chroma_collection = self._client.get_or_create_collection(collection)

        return {
            "name": collection,
            "count": chroma_collection.count(),
            "metadata": chroma_collection.metadata,
        }


class PineconeClient(VectorDatabase):
    """
    Pinecone managed vector database client.

    Features:
    - Fully managed service
    - Automatic scaling
    - Serverless and pod-based indexes
    - Built-in filtering
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment
        self.host = host

        self._client = None
        self._index = None

        if not self.api_key:
            raise ValueError("Pinecone API key not provided")

    async def connect(self) -> None:
        """Connect to Pinecone."""
        try:
            import pinecone

            if self.environment:
                pinecone.init(api_key=self.api_key, environment=self.environment)
            else:
                # New Pinecone API
                self._client = pinecone.Pinecone(api_key=self.api_key)

            logger.info("Connected to Pinecone")
        except ImportError:
            raise RuntimeError("pinecone not installed. Run: pip install pinecone-client")

    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self._index = None
        logger.info("Disconnected from Pinecone")

    async def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new index (Pinecone uses 'index' instead of 'collection')."""
        if not self._client:
            await self.connect()

        # Check if index already exists
        existing_indexes = self._client.list_indexes().names()
        if name in existing_indexes:
            logger.warning(f"Index {name} already exists")
            return

        # Create index
        self._client.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            cloud=cloud,
            region=region,
            **kwargs,
        )

        logger.info(f"Created Pinecone index: {name}")

    async def upsert(
        self,
        collection: str,
        records: List[VectorRecord],
    ) -> int:
        """Upsert records into Pinecone."""
        if not self._client:
            await self.connect()

        index = self._client.Index(collection)

        vectors = []
        for record in records:
            vector = {
                "id": record.id,
                "values": record.vector,
            }
            if record.metadata or record.payload:
                vector["metadata"] = {**record.metadata, **(record.payload or {})}
            vectors.append(vector)

        # Pinecone has batch size limits
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)

        logger.info(f"Upserted {len(records)} records to Pinecone index: {collection}")
        return len(records)

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Search for similar vectors."""
        import time
        start_time = time.time()

        if not self._client:
            await self.connect()

        index = self._client.Index(collection)

        response = index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            include_values=False,
        )

        records = []
        for match in response.matches:
            record = VectorRecord(
                id=match.id,
                vector=[],
                metadata=match.metadata or {},
                score=match.score,
            )
            records.append(record)

        latency_ms = (time.time() - start_time) * 1000

        return SearchResult(
            records=records,
            total=len(records),
            latency_ms=latency_ms,
        )

    async def delete(
        self,
        collection: str,
        ids: List[str],
    ) -> int:
        """Delete records by ID."""
        if not self._client:
            await self.connect()

        index = self._client.Index(collection)
        index.delete(ids=ids)

        logger.info(f"Deleted {len(ids)} records from Pinecone index: {collection}")
        return len(ids)

    async def get_collection_info(self, collection: str) -> Dict[str, Any]:
        """Get index information."""
        if not self._client:
            await self.connect()

        index = self._client.Index(collection)
        stats = index.describe_index_stats()

        return {
            "name": collection,
            "dimension": stats.dimension,
            "vectors_count": stats.total_vector_count,
            "metric": stats.get("metric", "cosine"),
        }


class VectorStoreManager:
    """
    Manager for multiple vector stores.

    Features:
    - Multi-store routing
    - Fallback handling
    - Statistics tracking
    """

    def __init__(self) -> None:
        self._stores: Dict[str, VectorDatabase] = {}
        self._default_store: Optional[str] = None

    def register_store(
        self,
        name: str,
        store: VectorDatabase,
        default: bool = False,
    ) -> None:
        """Register a vector store."""
        self._stores[name] = store
        if default or not self._default_store:
            self._default_store = name
        logger.info(f"Registered vector store: {name}")

    def get_store(self, name: Optional[str] = None) -> VectorDatabase:
        """Get a vector store by name."""
        name = name or self._default_store
        if not name or name not in self._stores:
            raise ValueError(f"Vector store not found: {name}")
        return self._stores[name]

    async def upsert_to_all(
        self,
        records: List[VectorRecord],
    ) -> Dict[str, int]:
        """Upsert records to all registered stores."""
        results = {}

        for name, store in self._stores.items():
            try:
                count = await store.upsert(name, records)
                results[name] = count
            except Exception as e:
                logger.error(f"Failed to upsert to {name}: {e}")
                results[name] = 0

        return results

    async def search_all(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> Dict[str, SearchResult]:
        """Search all registered stores."""
        results = {}

        for name, store in self._stores.items():
            try:
                result = await store.search(name, query_vector, top_k)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to search {name}: {e}")

        return results

    async def close_all(self) -> None:
        """Close all registered stores."""
        for name, store in self._stores.items():
            try:
                await store.disconnect()
            except Exception as e:
                logger.error(f"Failed to close {name}: {e}")

        self._stores.clear()
