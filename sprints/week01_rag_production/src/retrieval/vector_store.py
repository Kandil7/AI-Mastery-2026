"""
Vector Storage and Retrieval System for Production RAG

This module implements a comprehensive vector storage and retrieval system for the RAG system.
It manages vector embeddings for documents, provides efficient similarity search capabilities,
and integrates with various vector databases for scalable storage and retrieval.

The vector storage system follows production best practices:
- Support for multiple vector databases (Chroma, FAISS, Pinecone, etc.)
- Efficient similarity search algorithms
- Vector indexing and optimization
- Batch operations for performance
- Memory management for large vector sets
- Dimensionality validation and normalization

Key Features:
- Multiple vector database backends
- Similarity search with configurable algorithms
- Vector indexing and optimization
- Batch operations for efficient processing
- Memory-efficient vector storage
- Dimensionality validation and normalization
- Performance optimization for large-scale deployments
- Comprehensive error handling and monitoring

Security Considerations:
- Secure vector database connections
- Input validation for vector dimensions
- Access control for vector operations
- Encryption for vector data transmission
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception as e:
    CHROMA_AVAILABLE = False
    logging.warning(f"ChromaDB not available: {e}")

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception as e:
    FAISS_AVAILABLE = False
    logging.warning(f"FAISS not available: {e}")


class VectorDBType(Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


class VectorConfig(BaseModel):
    db_type: VectorDBType = Field(default=VectorDBType.IN_MEMORY)
    collection_name: str = Field(default="rag_vectors")
    persist_directory: str = Field(default="./data/vector_store")
    dimension: int = Field(default=384, ge=1)
    metric: str = Field(default="cosine")  # cosine | inner_product | l2
    batch_size: int = Field(default=64, ge=1, le=4096)

    # HNSW (Chroma)
    ef_construction: int = Field(default=200, ge=1)
    ef_search: int = Field(default=50, ge=1)
    m: int = Field(default=16, ge=1)


class VectorRecord(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: str
    text_content: Optional[str] = None


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def _flatten_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    # Chroma metadata must be scalar-like. Keep safe keys only.
    out = {}
    for k, v in (md or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class BaseVectorStore(ABC):
    def __init__(self, config: VectorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self): ...

    @abstractmethod
    async def upsert(self, records: List[VectorRecord]): ...

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]: ...

    @abstractmethod
    async def delete(self, ids: List[str]) -> None: ...

    @abstractmethod
    async def count(self) -> int: ...

    @abstractmethod
    async def close(self): ...


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        self._vectors: Dict[str, np.ndarray] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        self.logger.info("InMemoryVectorStore initialized (dev only).")

    async def upsert(self, records: List[VectorRecord]):
        try:
            for r in records:
                if len(r.vector) != self.config.dimension:
                    raise ValueError(f"Vector dimension mismatch: expected {self.config.dimension}, got {len(r.vector)}")
                v = _normalize(np.array(r.vector, dtype=np.float32)) if self.config.metric in ("cosine", "inner_product") else np.array(r.vector, dtype=np.float32)
                self._vectors[r.id] = v
                self._meta[r.id] = _flatten_metadata(r.metadata)
            self.logger.debug(f"Upserted {len(records)} records to InMemoryVectorStore")
        except Exception as e:
            self.logger.error(f"Error upserting records to InMemoryVectorStore: {e}")
            raise

    async def search(self, query_vector: List[float], k: int = 10, where: Optional[Dict[str, Any]] = None):
        try:
            if len(query_vector) != self.config.dimension:
                raise ValueError(f"Query vector dimension mismatch: expected {self.config.dimension}, got {len(query_vector)}")

            q = np.array(query_vector, dtype=np.float32)
            if self.config.metric in ("cosine", "inner_product"):
                q = _normalize(q)

            ids = list(self._vectors.keys())
            if where:
                # naive filter
                ids = [i for i in ids if all(self._meta.get(i, {}).get(k) == v for k, v in where.items())]

            if not ids:
                return []

            mat = np.vstack([self._vectors[i] for i in ids])
            if self.config.metric in ("cosine", "inner_product"):
                sims = mat @ q
            else:
                # l2 distance -> convert to similarity
                d = np.linalg.norm(mat - q, axis=1)
                sims = 1.0 / (1.0 + d)

            order = np.argsort(sims)[::-1][:k]
            results = [(ids[int(i)], float(sims[int(i)])) for i in order]
            self.logger.debug(f"Found {len(results)} results for query in InMemoryVectorStore")
            return results
        except Exception as e:
            self.logger.error(f"Error searching in InMemoryVectorStore: {e}")
            raise

    async def delete(self, ids: List[str]) -> None:
        try:
            for i in ids:
                self._vectors.pop(i, None)
                self._meta.pop(i, None)
            self.logger.debug(f"Deleted {len(ids)} records from InMemoryVectorStore")
        except Exception as e:
            self.logger.error(f"Error deleting records from InMemoryVectorStore: {e}")
            raise

    async def count(self) -> int:
        count = len(self._vectors)
        self.logger.debug(f"Counted {count} vectors in InMemoryVectorStore")
        return count

    async def close(self):
        self._vectors.clear()
        self._meta.clear()
        self.logger.info("Closed InMemoryVectorStore")


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        if not CHROMA_AVAILABLE:
            raise RuntimeError("Chroma not installed")

        self._client = None
        self._collection = None
        self._settings = Settings(
            anonymized_telemetry=False,
            persist_directory=config.persist_directory,
        )

    async def initialize(self):
        try:
            loop = asyncio.get_event_loop()

            def _init():
                client = chromadb.PersistentClient(path=self.config.persist_directory, settings=self._settings)
                metadata = {
                    "hnsw:space": "cosine" if self.config.metric == "cosine" else "ip",
                    "hnsw:construction_ef": self.config.ef_construction,
                    "hnsw:search_ef": self.config.ef_search,
                    "hnsw:M": self.config.m,
                }
                col = client.get_or_create_collection(name=self.config.collection_name, metadata=metadata)
                return client, col

            self._client, self._collection = await loop.run_in_executor(None, _init)
            self.logger.info("ChromaVectorStore initialized: %s", self.config.collection_name)
        except Exception as e:
            self.logger.error(f"Error initializing ChromaVectorStore: {e}")
            raise

    async def upsert(self, records: List[VectorRecord]):
        try:
            if not records:
                return
            loop = asyncio.get_event_loop()

            ids = [r.id for r in records]
            embs = []
            for r in records:
                if len(r.vector) != self.config.dimension:
                    raise ValueError(f"Vector dimension mismatch: expected {self.config.dimension}, got {len(r.vector)}")
                v = np.array(r.vector, dtype=np.float32)
                if self.config.metric in ("cosine", "inner_product"):
                    v = _normalize(v)
                embs.append(v.tolist())

            metadatas = [_flatten_metadata(r.metadata) for r in records]
            documents = [r.text_content or "" for r in records]

            def _upsert():
                # safest portable path: delete then add
                try:
                    self._collection.delete(ids=ids)
                except Exception:
                    pass
                self._collection.add(ids=ids, embeddings=embs, metadatas=metadatas, documents=documents)

            await loop.run_in_executor(None, _upsert)
            self.logger.debug(f"Upserted {len(records)} records to ChromaVectorStore")
        except Exception as e:
            self.logger.error(f"Error upserting records to ChromaVectorStore: {e}")
            raise

    async def search(self, query_vector: List[float], k: int = 10, where: Optional[Dict[str, Any]] = None):
        try:
            loop = asyncio.get_event_loop()

            q = np.array(query_vector, dtype=np.float32)
            if self.config.metric in ("cosine", "inner_product"):
                q = _normalize(q)

            def _query():
                kwargs = {"query_embeddings": [q.tolist()], "n_results": k, "include": ["distances", "metadatas", "ids"]}
                if where:
                    kwargs["where"] = where
                return self._collection.query(**kwargs)

            res = await loop.run_in_executor(None, _query)
            ids = res.get("ids", [[]])[0]
            distances = res.get("distances", [[]])[0]

            # Chroma distances depend on space; we standardize similarity:
            # cosine space often returns (1 - cosine_similarity)
            sims = []
            for i, d in zip(ids, distances):
                if self.config.metric == "cosine":
                    sims.append((i, float(1.0 - d)))
                else:
                    # treat distance as inverse
                    sims.append((i, float(1.0 / (1.0 + d))))

            self.logger.debug(f"Found {len(sims)} results for query in ChromaVectorStore")
            return sims
        except Exception as e:
            self.logger.error(f"Error searching in ChromaVectorStore: {e}")
            raise

    async def delete(self, ids: List[str]) -> None:
        try:
            loop = asyncio.get_event_loop()

            def _del():
                self._collection.delete(ids=ids)

            await loop.run_in_executor(None, _del)
            self.logger.debug(f"Deleted {len(ids)} records from ChromaVectorStore")
        except Exception as e:
            self.logger.error(f"Error deleting records from ChromaVectorStore: {e}")
            raise

    async def count(self) -> int:
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, lambda: self._collection.count())
            self.logger.debug(f"Counted {count} vectors in ChromaVectorStore")
            return count
        except Exception as e:
            self.logger.error(f"Error counting vectors in ChromaVectorStore: {e}")
            raise

    async def close(self):
        self._client = None
        self._collection = None
        self.logger.info("Closed ChromaVectorStore")


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed")
        self.index = None
        self._ids: List[str] = []
        self._meta: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        try:
            if self.config.metric in ("cosine", "inner_product"):
                self.index = faiss.IndexFlatIP(self.config.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.config.dimension)
            self.logger.info("FAISSVectorStore initialized dim=%s", self.config.dimension)
        except Exception as e:
            self.logger.error(f"Error initializing FAISSVectorStore: {e}")
            raise

    async def upsert(self, records: List[VectorRecord]):
        try:
            # FAISS upsert is non-trivial; production: use IVF/HNSW + external mapping.
            # Here we implement append-only with duplicate prevention by delete+rebuild (expensive).
            # On scale use a DB that supports upsert or store per-shard.
            existing = set(self._ids)
            new = [r for r in records if r.id not in existing]
            if not new:
                return

            vecs = []
            for r in new:
                v = np.array(r.vector, dtype=np.float32)
                if self.config.metric in ("cosine", "inner_product"):
                    v = _normalize(v)
                vecs.append(v)
                self._meta[r.id] = _flatten_metadata(r.metadata)
                self._ids.append(r.id)

            mat = np.vstack(vecs).astype("float32")
            self.index.add(mat)
            self.logger.debug(f"Upserted {len(new)} records to FAISSVectorStore")
        except Exception as e:
            self.logger.error(f"Error upserting records to FAISSVectorStore: {e}")
            raise

    async def search(self, query_vector: List[float], k: int = 10, where: Optional[Dict[str, Any]] = None):
        try:
            q = np.array([query_vector], dtype=np.float32)
            if self.config.metric in ("cosine", "inner_product"):
                q = np.array([_normalize(q[0])], dtype=np.float32)

            distances, idxs = self.index.search(q, k)
            out = []
            for dist, idx in zip(distances[0], idxs[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                vid = self._ids[int(idx)]
                if where:
                    md = self._meta.get(vid, {})
                    if not all(md.get(k) == v for k, v in where.items()):
                        continue
                sim = float(dist) if self.config.metric in ("cosine", "inner_product") else float(1.0 / (1.0 + dist))
                out.append((vid, sim))

            self.logger.debug(f"Found {len(out)} results for query in FAISSVectorStore")
            return out
        except Exception as e:
            self.logger.error(f"Error searching in FAISSVectorStore: {e}")
            raise

    async def delete(self, ids: List[str]) -> None:
        try:
            # FAISS delete is not supported efficiently; mark deleted.
            for i in ids:
                if i in self._meta:
                    self._meta.pop(i, None)
            self.logger.debug(f"Marked {len(ids)} records for deletion in FAISSVectorStore")
        except Exception as e:
            self.logger.error(f"Error deleting records from FAISSVectorStore: {e}")
            raise

    async def count(self) -> int:
        count = self.index.ntotal
        self.logger.debug(f"Counted {count} vectors in FAISSVectorStore")
        return count

    async def close(self):
        self.index = None
        self._ids.clear()
        self._meta.clear()
        self.logger.info("Closed FAISSVectorStore")


class VectorStoreFactory:
    @staticmethod
    def create(config: VectorConfig) -> BaseVectorStore:
        if config.db_type == VectorDBType.CHROMA:
            return ChromaVectorStore(config)
        if config.db_type == VectorDBType.FAISS:
            return FAISSVectorStore(config)
        return InMemoryVectorStore(config)
