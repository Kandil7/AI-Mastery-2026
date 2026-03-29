"""
RAG Module - Unified Retrieval-Augmented Generation
====================================================

Complete RAG implementation with chunking, retrieval, and reranking.

Components:
- **chunking**: Document splitting strategies (fixed, recursive, semantic, hierarchical)
- **retrieval**: Search strategies (similarity, hybrid, multi-query, HyDE)
- **reranking**: Result refinement (cross-encoder, LLM, diversity)
- **advanced**: Advanced techniques (query construction, tools, post-processing)
- **specialized**: Specialized RAGs (multimodal, temporal, graph-enhanced)

Quick Start:
------------
    >>> from src.rag import RAGPipeline, Document, DocumentChunk
    >>> from src.rag.chunking import SemanticChunker
    >>> from src.rag.retrieval import HybridRetrieval
    >>> from src.rag.reranking import CrossEncoderReranker
    >>> from src.vector_stores import FAISSStore, VectorStoreConfig
    >>> from src.embeddings import TextEmbedder
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
    >>> docs = [
    ...     {"id": "1", "content": "AI is transforming industries."},
    ...     {"id": "2", "content": "Machine learning powers modern AI."},
    ... ]
    >>> pipeline.add_documents(docs)
    >>>
    >>> # Query
    >>> results = pipeline.query("How does AI work?")
    >>> for result in results:
    ...     print(f"{result.id}: {result.content}")

Structure:
----------
    rag/
    ├── core.py              # Main RAGPipeline class
    ├── types.py             # Document, Chunk types
    ├── chunking/            # Chunking strategies
    ├── retrieval/           # Retrieval strategies
    ├── reranking/           # Reranking strategies
    ├── advanced/            # Advanced techniques
    └── specialized/         # Specialized RAG variants
"""

# Core RAG pipeline
from .core import RAGPipeline, RAGConfig
from .types import Document, DocumentChunk, Query, RAGResult

# Chunking strategies
from .chunking import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    HierarchicalChunker,
    TokenAwareChunker,
    CodeChunker,
    ChunkingFactory,
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
    "Document",
    "DocumentChunk",
    "Query",
    "RAGResult",
    
    # Chunking
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "TokenAwareChunker",
    "CodeChunker",
    "ChunkingFactory",
    
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
