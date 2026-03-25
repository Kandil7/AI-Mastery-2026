"""
Vector database implementation for RAG system.
Supports Qdrant, ChromaDB, and in-memory fallback.
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    store_type: str = "memory"  # 'qdrant', 'chroma', 'memory'
    collection_name: str = "arabic_islamic_literature"
    vector_size: int = 768
    distance: str = "cosine"

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ChromaDB settings
    chroma_persist_dir: str = "./data/chroma"


class VectorStore:
    """Vector database for storing and retrieving embeddings."""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize vector store.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self._client = None
        self._collection = None

    @property
    def client(self):
        """Lazy load the client."""
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_client(self):
        """Initialize the vector database client."""
        if self.config.store_type == "qdrant":
            return self._init_qdrant()
        elif self.config.store_type == "chroma":
            return self._init_chroma()
        else:
            return self._init_memory()

    def _init_qdrant(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import DistanceVector, VectorParams

            client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
            )

            # Create collection if not exists
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.config.collection_name not in collection_names:
                client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=DistanceVector.COSINE,
                    ),
                )

            return client
        except ImportError:
            print("Qdrant not installed. Falling back to in-memory store.")
            self.config.store_type = "memory"
            return self._init_memory()
        except Exception as e:
            print(f"Could not connect to Qdrant: {e}. Falling back to in-memory store.")
            self.config.store_type = "memory"
            return self._init_memory()

    def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=self.config.chroma_persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create collection
            collection = client.get_or_create_collection(
                name=self.config.collection_name, metadata={"hnsw:space": "cosine"}
            )

            return client
        except ImportError:
            print("ChromaDB not installed. Falling back to in-memory store.")
            self.config.store_type = "memory"
            return self._init_memory()

    def _init_memory(self):
        """Initialize in-memory store."""
        return {"vectors": {}, "metadata": {}, "ids": []}

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: List[Dict[str, Any]],
    ):
        """
        Add vectors to the store.

        Args:
            ids: List of unique IDs
            vectors: List of embedding vectors
            payloads: List of metadata payloads
        """
        if self.config.store_type == "qdrant":
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(id=idx, vector=vector.tolist(), payload=payload)
                for idx, (vector, payload) in enumerate(zip(vectors, payloads))
            ]

            self.client.upsert(
                collection_name=self.config.collection_name, points=points
            )

        elif self.config.store_type == "chroma":
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name
            )

            # Convert numpy arrays to lists
            vectors_list = [vector.tolist() for vector in vectors]

            self._collection.add(ids=ids, embeddings=vectors_list, metadatas=payloads)

        else:  # memory
            for id_, vector, payload in zip(ids, vectors, payloads):
                self.client["ids"].append(id_)
                self.client["vectors"][id_] = vector
                self.client["metadata"][id_] = payload

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with scores and metadata
        """
        if self.config.store_type == "qdrant":
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                search_filter = Filter(must=conditions)

            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=search_filter,
            )

            return [
                {"id": str(r.id), "score": r.score, "payload": r.payload}
                for r in results
            ]

        elif self.config.store_type == "chroma":
            results = self._collection.query(
                query_embeddings=[query_vector.tolist()], n_results=top_k, where=filters
            )

            return [
                {
                    "id": results["ids"][0][i],
                    "score": 1
                    - results["distances"][0][i],  # Convert distance to similarity
                    "payload": results["metadatas"][0][i]
                    if results["metadatas"]
                    else {},
                }
                for i in range(len(results["ids"][0]))
            ]

        else:  # memory - simple cosine similarity
            scores = []

            for id_ in self.client["ids"]:
                vector = self.client["vectors"][id_]

                # Apply filters if provided
                if filters:
                    metadata = self.client["metadata"][id_]
                    if not all(metadata.get(k) == v for k, v in filters.items()):
                        continue

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, vector)
                scores.append((id_, similarity, self.client["metadata"][id_]))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            return [
                {"id": id_, "score": score, "payload": metadata}
                for id_, score, metadata in scores[:top_k]
            ]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def get_by_id(self, id_: str) -> Optional[Dict[str, Any]]:
        """Get vector and metadata by ID."""
        if self.config.store_type == "qdrant":
            results = self.client.retrieve(
                collection_name=self.config.collection_name, ids=[id_]
            )
            if results:
                return {
                    "id": str(results[0].id),
                    "vector": results[0].vector,
                    "payload": results[0].payload,
                }
            return None

        elif self.config.store_type == "chroma":
            results = self._collection.get(ids=[id_])
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "vector": results["embeddings"][0],
                    "payload": results["metadatas"][0],
                }
            return None

        else:  # memory
            if id_ in self.client["vectors"]:
                return {
                    "id": id_,
                    "vector": self.client["vectors"][id_],
                    "payload": self.client["metadata"][id_],
                }
            return None

    def delete_collection(self):
        """Delete the collection."""
        if self.config.store_type == "qdrant":
            self.client.delete_collection(self.config.collection_name)

        elif self.config.store_type == "chroma":
            self.client.delete_collection(self.config.collection_name)

        else:  # memory
            self.client["vectors"].clear()
            self.client["metadata"].clear()
            self.client["ids"].clear()

    def count(self) -> int:
        """Get the number of vectors in the store."""
        if self.config.store_type == "qdrant":
            return self.client.get_collection(self.config.collection_name).count()

        elif self.config.store_type == "chroma":
            return self._collection.count()

        else:  # memory
            return len(self.client["ids"])

    def exists(self, id_: str) -> bool:
        """Check if ID exists in store."""
        if self.config.store_type == "qdrant":
            results = self.client.retrieve(
                collection_name=self.config.collection_name, ids=[id_]
            )
            return len(results) > 0

        elif self.config.store_type == "chroma":
            results = self._collection.get(ids=[id_])
            return len(results["ids"]) > 0

        else:  # memory
            return id_ in self.client["vectors"]


def create_vector_store(
    store_type: str = "memory",
    collection_name: str = "arabic_islamic_literature",
    vector_size: int = 768,
    **kwargs,
) -> VectorStore:
    """
    Factory function to create a vector store.

    Args:
        store_type: Type of vector store ('qdrant', 'chroma', 'memory')
        collection_name: Name of the collection
        vector_size: Dimension of vectors
        **kwargs: Additional configuration

    Returns:
        VectorStore instance
    """
    config = VectorStoreConfig(
        store_type=store_type,
        collection_name=collection_name,
        vector_size=vector_size,
        **kwargs,
    )

    return VectorStore(config)
