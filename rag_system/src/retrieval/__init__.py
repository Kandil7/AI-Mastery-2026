"""
Retrieval Module

Handles document retrieval:
- Vector search (Qdrant, ChromaDB, Memory)
- BM25 keyword search
- Hybrid retrieval
- Query transformation
"""

from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorStoreType,
    create_vector_store,
    SearchResult,
    MemoryVectorStore,
    QdrantVectorStore,
    ChromaDBVectorStore,
)

from .hybrid_retriever import (
    HybridRetriever,
    BM25Index,
    Reranker,
    RetrievalResult,
    AdaptiveRetriever,
)

from .query_transformer import (
    QueryTransformer,
    create_query_transformer,
    TransformedQuery,
    QueryType,
)

from .bm25_retriever import (
    BM25Retriever,
)

__all__ = [
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreType",
    "create_vector_store",
    "SearchResult",
    "MemoryVectorStore",
    "QdrantVectorStore",
    "ChromaDBVectorStore",
    # Retrievers
    "HybridRetriever",
    "BM25Index",
    "BM25Retriever",
    "Reranker",
    "RetrievalResult",
    "AdaptiveRetriever",
    # Query
    "QueryTransformer",
    "create_query_transformer",
    "TransformedQuery",
    "QueryType",
]
