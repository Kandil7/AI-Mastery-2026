"""
RAG Retrieval Module
====================

Retrieval strategies for RAG systems.

Provides:
- Similarity search
- Hybrid retrieval (dense + sparse)
- Multi-query retrieval
- HyDE (Hypothetical Document Embeddings)
- Ensemble retrieval

Quick Start:
    >>> from src.rag.retrieval import HybridRetrieval, SimilarityRetriever
    >>> from src.rag.vector_stores import FAISSStore, VectorStoreConfig
    >>>
    >>> # Create vector store
    >>> config = VectorStoreConfig(dim=384)
    >>> store = FAISSStore(config)
    >>>
    >>> # Create retriever
    >>> retriever = SimilarityRetriever(store, top_k=5)
    >>> results = retriever.retrieve(query_vector)
"""

from .similarity import SimilarityRetriever
from .hybrid import HybridRetrieval
from .multi_query import MultiQueryRetriever
from .hyde import HyDERetriever
from .base import BaseRetriever, RetrievalResults

__all__ = [
    # Base classes
    "BaseRetriever",
    "RetrievalResults",
    
    # Retrieval strategies
    "SimilarityRetriever",
    "HybridRetrieval",
    "MultiQueryRetriever",
    "HyDERetriever",
]

