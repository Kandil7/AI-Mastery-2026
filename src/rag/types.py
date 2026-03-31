"""
RAG Type Definitions
=====================

Type definitions for the RAG pipeline.

Dataclasses:
    Document: Represents a document in the RAG system
    DocumentChunk: Represents a chunk of a document
    Query: Represents a user query
    RAGResult: Represents a search result
    ChunkMetadata: Metadata for document chunks
    RetrievalMetrics: Metrics for retrieval operations

Author: AI-Mastery-2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""

    source: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    headings: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """
    Represents a document in the RAG system.

    Attributes:
        id: Unique identifier for the document
        content: The text content of the document
        metadata: Additional metadata about the document
        created_at: Timestamp when the document was created
        updated_at: Timestamp when the document was last updated
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate the document after initialization."""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata key-value pair."""
        self.metadata[key] = value
        self.updated_at = datetime.now()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class DocumentChunk:
    """
    Represents a chunk of a document.

    Attributes:
        id: Unique identifier for the chunk
        content: The text content of the chunk
        document_id: ID of the parent document
        chunk_index: Index of this chunk within the document
        embedding: Optional vector embedding for the chunk
        metadata: Metadata about this chunk
        start_char: Starting character position in original document
        end_char: Ending character position in original document
    """

    id: str
    content: str
    document_id: str
    chunk_index: int = 0
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    def __post_init__(self):
        """Validate the chunk after initialization."""
        if not self.id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        if not self.document_id:
            raise ValueError("Document ID cannot be empty")

    def get_text(self, context_chars: int = 0) -> str:
        """
        Get the chunk text with optional context.

        Args:
            context_chars: Number of characters to include as context

        Returns:
            The chunk content (with context if specified)
        """
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "embedding": self.embedding,
            "metadata": {
                "source": self.metadata.source,
                "page_number": self.metadata.page_number,
                "section": self.metadata.section,
                "headings": self.metadata.headings,
                **self.metadata.custom,
            },
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class Query:
    """
    Represents a user query for the RAG system.

    Attributes:
        text: The query text
        top_k: Number of results to return
        filters: Optional metadata filters
        metadata: Additional query metadata
        user_id: Optional user ID for ACL filtering
    """

    text: str
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None

    def __post_init__(self):
        """Validate the query after initialization."""
        if not self.text:
            raise ValueError("Query text cannot be empty")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

    def add_filter(self, key: str, value: Any) -> "Query":
        """Add a filter to the query (returns new query)."""
        new_filters = dict(self.filters or {})
        new_filters[key] = value
        return Query(
            text=self.text,
            top_k=self.top_k,
            filters=new_filters,
            metadata=self.metadata,
            user_id=self.user_id,
        )


@dataclass
class RAGResult:
    """
    Represents a search result from the RAG system.

    Attributes:
        chunk: The document chunk that matched
        score: Similarity score (higher is better)
        query: The original query
        rank: Position in result list (1-based)
    """

    chunk: DocumentChunk
    score: float
    query: str
    rank: int = 0

    def __post_init__(self):
        """Set rank based on position in results."""
        if self.rank == 0:
            self.rank = getattr(self, "_rank", 0)

    @property
    def content(self) -> str:
        """Get the chunk content."""
        return self.chunk.content

    @property
    def document_id(self) -> str:
        """Get the parent document ID."""
        return self.chunk.document_id

    @property
    def chunk_id(self) -> str:
        """Get the chunk ID."""
        return self.chunk.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "query": self.query,
            "rank": self.rank,
        }


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval operations."""

    total_results: int = 0
    query_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    reranking_time_ms: float = 0.0
    cache_hit: bool = False
    cache_key: Optional[str] = None
    embedding_time_ms: float = 0.0

    def total_time_ms(self) -> float:
        """Get total query time."""
        return (
            self.query_time_ms
            + self.retrieval_time_ms
            + self.reranking_time_ms
            + self.embedding_time_ms
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_results": self.total_results,
            "query_time_ms": self.query_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "reranking_time_ms": self.reranking_time_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "total_time_ms": self.total_time_ms(),
            "cache_hit": self.cache_hit,
        }


# Type aliases for common patterns
ChunkList = List[DocumentChunk]
ResultList = List[RAGResult]
FilterDict = Dict[str, Any]
Embedding = List[float]
EmbeddingList = List[Embedding]
