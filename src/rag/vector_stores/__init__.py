"""
Vector Stores Module
====================

Unified interfaces for vector database operations.

Supports:
- FAISS (Facebook AI Similarity Search)
- Qdrant (Vector similarity search engine)
- ChromaDB (Embedding database)
- Weaviate (Vector search engine)
- pgvector (PostgreSQL extension)
- In-memory store (for testing)

Quick Start:
    >>> from src.rag.vector_stores import FAISSStore, QdrantStore
    >>> from src.rag.vector_stores import VectorStoreConfig
    >>>
    >>> # Create FAISS store
    >>> config = VectorStoreConfig(dim=384, metric="cosine")
    >>> store = FAISSStore(config)
    >>>
    >>> # Add vectors
    >>> vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    >>> ids = ["doc1", "doc2"]
    >>> store.upsert(vectors, ids)
    >>>
    >>> # Search
    >>> results = store.search(query_vector, top_k=5)
"""

from .base import VectorStore, VectorStoreConfig, SearchResults, SearchResult
from .memory import MemoryVectorStore
from .faiss_store import FAISSStore

__all__ = [
    # Base classes
    "VectorStore",
    "VectorStoreConfig",
    "SearchResults",
    "SearchResult",
    
    # Implementations
    "MemoryVectorStore",
    "FAISSStore",
    
    # Note: Additional stores available via submodule imports
    # from src.rag.vector_stores.qdrant_store import QdrantStore
    # from src.rag.vector_stores.chroma_store import ChromaStore
    # from src.rag.vector_stores.weaviate_store import WeaviateStore
    # from src.rag.vector_stores.pgvector_store import PGVectorStore
]

