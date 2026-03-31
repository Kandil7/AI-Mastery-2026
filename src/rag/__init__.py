"""
RAG Module - Unified Retrieval-Augmented Generation
===================================================

Complete RAG implementation with chunking, retrieval, and reranking.

Components:
- **types**: Core type definitions (Document, Chunk, Query, RAGResult)
- **core**: Main RAGPipeline implementation
- **chunking**: Document splitting strategies (fixed, recursive, semantic, hierarchical)
- **retrieval**: Search strategies (similarity, hybrid, multi-query, HyDE)
- **reranking**: Result refinement (cross-encoder, LLM, diversity)
- **advanced**: Advanced techniques (query construction, tools, post-processing)
- **specialized**: Specialized RAGs (multimodal, temporal, graph-enhanced)

Quick Start:
------------
    >>> from src.rag import RAGPipeline, RAGConfig, Document
    >>> from src.rag.chunking import SemanticChunker
    >>> from src.rag.retrieval import HybridRetrieval
    >>> from src.rag.reranking import CrossEncoderReranker
    >>> from src.rag.vector_stores import FAISSStore, VectorStoreConfig
    >>> from src.rag.embeddings import TextEmbedder
    >>>
    >>> # Initialize components
    >>> embedder = TextEmbedder("all-MiniLM-L6-v2")
    >>> store = FAISSStore(VectorStoreConfig(dim=384))
    >>> chunker = SemanticChunker()
    >>> retriever = HybridRetrieval(dense_store=store, sparse_store=bm25)
    >>> reranker = CrossEncoderReranker()
    >>>
    >>> # Create pipeline
    >>> pipeline = RAGPipeline(
    ...     embedder=embedder,
    ...     vector_store=store,
    ...     chunker=chunker,
    ...     retriever=retriever,
    ...     reranker=reranker,
    ... )
    >>>
    >>> # Add documents
    >>> docs = [Document(id="1", content="AI is transforming industries.")]
    >>> pipeline.add_documents(docs)
    >>>
    >>> # Query
    >>> from src.rag import Query
    >>> results = pipeline.query(Query(text="How does AI work?", top_k=5))
    >>> for result in results:
    ...     print(f"{result.score:.3f}: {result.content[:100]}...")

Structure:
----------
    rag/
    ├── core.py              # Main RAGPipeline class
    ├── types.py             # Document, Chunk types
    ├── chunking/            # Chunking strategies
    ├── retrieval/           # Retrieval strategies
    ├── reranking/           # Reranking strategies
    ├── embeddings/          # Embedding models
    ├── vector_stores/       # Vector storage
    ├── advanced/            # Advanced techniques
    └── specialized/         # Specialized RAG variants
"""

# Core types and pipeline
from .types import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    Query,
    RAGResult,
    RetrievalMetrics,
    ChunkList,
    ResultList,
    FilterDict,
    Embedding,
    EmbeddingList,
)

from .core import (
    RAGConfig,
    RAGMetrics,
    RAGPipeline,
)

# Chunking strategies
from .chunking import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    HierarchicalChunker,
    TokenAwareChunker,
    CodeChunker,
    ChunkerFactory,
)

# Retrieval strategies
from .retrieval import (
    BaseRetriever,
    SimilarityRetriever,
    HybridRetrieval,
    MultiQueryRetriever,
    HyDERetriever,
    RetrievalResults,
)

# Reranking strategies
from .reranking import (
    BaseReranker,
    CrossEncoderReranker,
    LLMReranker,
    DiversityReranker,
    RerankResults,
)

__all__ = [
    # Core
    "RAGPipeline",
    "RAGConfig",
    "RAGMetrics",
    "Document",
    "DocumentChunk",
    "Query",
    "RAGResult",
    "RetrievalMetrics",
    "ChunkMetadata",
    # Type aliases
    "ChunkList",
    "ResultList",
    "FilterDict",
    "Embedding",
    "EmbeddingList",
    # Chunking
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "TokenAwareChunker",
    "CodeChunker",
    "ChunkerFactory",
    # Retrieval
    "BaseRetriever",
    "SimilarityRetriever",
    "HybridRetrieval",
    "MultiQueryRetriever",
    "HyDERetriever",
    "RetrievalResults",
    # Reranking
    "BaseReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "DiversityReranker",
    "RerankResults",
]

__version__ = "2.0.0"
