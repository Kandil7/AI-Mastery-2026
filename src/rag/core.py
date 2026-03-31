"""
RAG Pipeline Core
=================

Main RAG pipeline implementation.

Classes:
    RAGConfig: Configuration for RAG pipeline
    RAGPipeline: Main RAG pipeline for document retrieval and generation
    RAGMetrics: Metrics collection for RAG operations

Author: AI-Mastery-2026
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .types import (
    Document,
    DocumentChunk,
    Query,
    RAGResult,
    RetrievalMetrics,
    ChunkMetadata,
)

logger = logging.getLogger(__name__)


# Type imports for components (will be imported lazily)
ChunkingStrategy = None
RetrievalStrategy = None
RerankingStrategy = None
TextEmbedder = None
VectorStore = None


def _get_chunking_strategy() -> Type:
    """Lazy import of chunking strategy."""
    global ChunkingStrategy
    if ChunkingStrategy is None:
        from .chunking.base import BaseChunker, ChunkingStrategy

        ChunkingStrategy = ChunkingStrategy
    return ChunkingStrategy


def _get_retrieval_strategy() -> Type:
    """Lazy import of retrieval strategy."""
    global RetrievalStrategy
    if RetrievalStrategy is None:
        from .retrieval.base import BaseRetriever

        RetrievalStrategy = BaseRetriever
    return RetrievalStrategy


def _get_reranking_strategy() -> Type:
    """Lazy import of reranking strategy."""
    global RerankingStrategy
    if RerankingStrategy is None:
        from .reranking.base import BaseReranker

        RerankingStrategy = BaseReranker
    return RerankingStrategy


@dataclass
class RAGConfig:
    """
    Configuration for RAG pipeline.

    Attributes:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        top_k: Number of results to retrieve
        enable_reranking: Whether to enable reranking
        rerank_top_k: Number of results to keep after reranking
        embedding_model: Name of embedding model
        vector_store_type: Type of vector store (faiss, memory)
        retrieval_mode: Retrieval mode (dense, sparse, hybrid)
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 10
    enable_reranking: bool = True
    rerank_top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_type: str = "faiss"
    retrieval_mode: str = "dense"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_size < 10:
            raise ValueError("chunk_size must be at least 10")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.rerank_top_k > self.top_k:
            raise ValueError("rerank_top_k cannot be larger than top_k")


class RAGMetrics:
    """Metrics collection for RAG pipeline."""

    def __init__(self):
        self.chunking_time_ms: float = 0.0
        self.embedding_time_ms: float = 0.0
        self.retrieval_time_ms: float = 0.0
        self.reranking_time_ms: float = 0.0
        self.total_time_ms: float = 0.0
        self.chunks_created: int = 0
        self.results_returned: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "chunking_time_ms": self.chunking_time_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "reranking_time_ms": self.reranking_time_ms,
            "total_time_ms": self.total_time_ms,
            "chunks_created": self.chunks_created,
            "results_returned": self.results_returned,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            ),
        }


class RAGPipeline:
    """
    Main RAG pipeline for document retrieval and generation.

    This pipeline orchestrates:
    1. Document chunking
    2. Embedding generation
    3. Vector storage
    4. Retrieval
    5. Optional reranking

    Example:
        >>> from src.rag import RAGPipeline, RAGConfig
        >>> from src.rag.chunking import SemanticChunker
        >>> from src.rag.vector_stores import FAISSStore
        >>> from src.rag.embeddings import TextEmbedder
        >>>
        >>> config = RAGConfig(chunk_size=512)
        >>> pipeline = RAGPipeline(
        ...     embedder=TextEmbedder("all-MiniLM-L6-v2"),
        ...     vector_store=FAISSStore(dim=384),
        ...     config=config
        ... )
        >>>
        >>> # Add documents
        >>> docs = [Document(id="1", content="AI is transforming industries")]
        >>> pipeline.add_documents(docs)
        >>>
        >>> # Query
        >>> results = pipeline.query("How is AI impacting business?")
        >>> for r in results:
        ...     print(f"{r.score:.3f}: {r.content[:100]}...")
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        chunker: Optional[Any] = None,
        retriever: Optional[Any] = None,
        reranker: Optional[Any] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            embedder: Text embedder for generating embeddings
            vector_store: Vector store for similarity search
            chunker: Document chunker
            retriever: Retrieval strategy
            reranker: Reranking strategy
            config: RAG configuration
        """
        self.config = config or RAGConfig()
        self.config.validate()

        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker

        # Internal state
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, DocumentChunk] = {}
        self._chunk_to_doc: Dict[str, str] = {}
        self._metrics = RAGMetrics()

        logger.info(f"Initialized RAGPipeline with config: {self.config}")

    @property
    def document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self._chunks)

    def add_documents(
        self,
        documents: List[Document],
        auto_embed: bool = True,
    ) -> int:
        """
        Add documents to the RAG pipeline.

        Args:
            documents: List of documents to add
            auto_embed: Whether to automatically generate embeddings

        Returns:
            Number of chunks created
        """
        start_time = time.time()
        total_chunks = 0

        for doc in documents:
            # Store document
            self._documents[doc.id] = doc

            # Chunk the document
            chunks = self._chunk_document(doc)
            total_chunks += len(chunks)

            # Store chunks
            for chunk in chunks:
                self._chunks[chunk.id] = chunk
                self._chunk_to_doc[chunk.id] = doc.id

            # Generate embeddings if enabled
            if auto_embed and self.embedder and self.vector_store:
                self._index_chunks(chunks)

        self._metrics.chunks_created += total_chunks
        self._metrics.chunking_time_ms += (time.time() - start_time) * 1000

        logger.info(f"Added {len(documents)} documents, created {total_chunks} chunks")
        return total_chunks

    def _chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces."""
        if self.chunker is None:
            # Default simple chunking
            return self._simple_chunk(document)

        # Use configured chunker
        from .chunking.base import Chunk

        chunks = self.chunker.chunk(document.content)
        return [
            DocumentChunk(
                id=f"{document.id}_chunk_{i}",
                content=c.content,
                document_id=document.id,
                chunk_index=i,
                metadata=ChunkMetadata(
                    source=document.metadata.get("source"),
                    custom=document.metadata,
                ),
                start_char=c.start_index,
                end_char=c.end_index,
            )
            for i, c in enumerate(chunks)
        ]

    def _simple_chunk(self, document: Document) -> List[DocumentChunk]:
        """Simple fixed-size chunking as fallback."""
        content = document.content
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        chunks = []
        start = 0
        index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(
                DocumentChunk(
                    id=f"{document.id}_chunk_{index}",
                    content=content[start:end],
                    document_id=document.id,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                )
            )
            start = end - overlap
            index += 1

        return chunks

    def _index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks in the vector store."""
        if not self.vector_store or not self.embedder:
            return

        # Generate embeddings
        contents = [c.content for c in chunks]
        embeddings = self.embedder(contents)

        # Add to vector store
        for chunk, embedding in zip(chunks, embeddings):
            self.vector_store.add(
                id=chunk.id,
                vector=embedding,
                metadata={
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                },
            )

    def query(
        self,
        query: Query,
        return_metadata: bool = False,
    ) -> List[RAGResult]:
        """
        Query the RAG pipeline.

        Args:
            query: Query object or string
            return_metadata: Whether to return full metadata

        Returns:
            List of RAG results sorted by relevance
        """
        # Convert string to Query if needed
        if isinstance(query, str):
            query = Query(text=query, top_k=self.config.top_k)

        start_time = time.time()
        results = []

        # 1. Retrieve from vector store
        if self.vector_store and self.embedder:
            retrieval_results = self._retrieve_from_vector_store(query)
        else:
            retrieval_results = self._retrieve_fallback(query)

        # 2. Optionally rerank
        if self.config.enable_reranking and self.reranker and retrieval_results:
            retrieval_results = self._rerank(query, retrieval_results)

        # 3. Convert to RAG results
        for rank, (chunk_id, score) in enumerate(retrieval_results, 1):
            chunk = self._chunks.get(chunk_id)
            if chunk:
                results.append(
                    RAGResult(
                        chunk=chunk,
                        score=score,
                        query=query.text,
                        rank=rank,
                    )
                )

        self._metrics.results_returned = len(results)
        self._metrics.total_time_ms += (time.time() - start_time) * 1000

        return results[: query.top_k]

    def _retrieve_from_vector_store(self, query: Query) -> List[tuple]:
        """Retrieve from vector store."""
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embedder([query.text])[0]

        # Search vector store
        raw_results = self.vector_store.search(
            query=query_embedding,
            top_k=query.top_k,
            filters=query.filters,
        )

        self._metrics.retrieval_time_ms += (time.time() - start_time) * 1000

        # Extract IDs and scores
        return [(r.id, r.score) for r in raw_results]

    def _retrieve_fallback(self, query: Query) -> List[tuple]:
        """Fallback retrieval using simple text matching."""
        # Simple keyword matching as fallback
        query_terms = set(query.text.lower().split())
        scores = []

        for chunk_id, chunk in self._chunks.items():
            chunk_terms = set(chunk.content.lower().split())
            overlap = len(query_terms & chunk_terms)
            if overlap > 0:
                scores.append((chunk_id, overlap / len(query_terms)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[: query.top_k]

    def _rerank(self, query: Query, results: List[tuple]) -> List[tuple]:
        """Rerank results using reranker."""
        start_time = time.time()

        # Get chunks for results
        chunks = [
            self._chunks[chunk_id]
            for chunk_id, _ in results
            if chunk_id in self._chunks
        ]

        # Rerank
        reranked = self.reranker.rerank(query.text, [c.content for c in chunks])

        # Map back to (id, score) format
        reranked_results = []
        for chunk, score in zip(chunks, reranked):
            reranked_results.append((chunk.id, score))

        self._metrics.reranking_time_ms += (time.time() - start_time) * 1000

        return reranked_results

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by ID."""
        return self._chunks.get(chunk_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self._metrics.to_dict()

    def clear(self) -> None:
        """Clear all documents and chunks."""
        self._documents.clear()
        self._chunks.clear()
        self._chunk_to_doc.clear()
        if self.vector_store:
            self.vector_store.clear()
        logger.info("Cleared RAG pipeline")

    def __len__(self) -> int:
        """Return number of documents."""
        return len(self._documents)

    def __contains__(self, doc_id: str) -> bool:
        """Check if document exists."""
        return doc_id in self._documents
