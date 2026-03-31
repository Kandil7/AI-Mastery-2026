"""
RAG Engine SDK - Models Module

Pydantic models for type-safe API interactions.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""

    CREATED = "created"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    ARCHIVED = "archived"


class QuerySortBy(str, Enum):
    """Query history sorting options."""

    CREATED = "created"
    UPDATED = "updated"
    FILENAME = "filename"
    SIZE = "size"
    RELEVANCE = "relevance"


class Answer(BaseModel):
    """RAG answer with metadata."""

    text: str = Field(..., description="Generated answer text")
    sources: List[str] = Field(default_factory=list, description="Source document IDs")
    retrieval_k: int = Field(default=5, description="Number of documents retrieved")
    embed_ms: Optional[int] = Field(None, description="Embedding time in milliseconds")
    search_ms: Optional[int] = Field(None, description="Search time in milliseconds")
    llm_ms: Optional[int] = Field(None, description="LLM generation time in milliseconds")
    reranked: bool = Field(default=False, description="Whether results were reranked")
    hybrid_search: bool = Field(default=False, description="Whether hybrid search was used")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "RAG (Retrieval-Augmented Generation) is a technique...",
                "sources": ["doc-123", "doc-456"],
                "retrieval_k": 5,
                "embed_ms": 150,
                "search_ms": 45,
                "llm_ms": 1200,
            }
        }


class Document(BaseModel):
    """Document metadata."""

    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Document title")
    content_type: str = Field(..., description="MIME type")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    status: DocumentStatus = Field(..., description="Processing status")
    created_at: str = Field(..., description="Creation timestamp (ISO 8601)")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc-abc123",
                "filename": "report.pdf",
                "title": "Annual Report 2024",
                "content_type": "application/pdf",
                "size_bytes": 1048576,
                "status": "indexed",
                "created_at": "2024-01-15T10:30:00Z",
            }
        }


class QueryHistoryItem(BaseModel):
    """Query history entry."""

    question: str = Field(..., description="User question")
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list, description="Source document IDs")
    timestamp: str = Field(..., description="Query timestamp (ISO 8601)")
    success: bool = Field(default=True, description="Whether query succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    latency_ms: Optional[int] = Field(None, description="Total query latency")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is RAG?",
                "answer": "RAG is a technique...",
                "sources": ["doc-123"],
                "timestamp": "2024-01-15T10:30:00Z",
                "success": True,
            }
        }


class QueryOptions(BaseModel):
    """Options for document search queries."""

    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    use_hybrid_search: bool = Field(
        default=True, description="Use hybrid (vector + keyword) search"
    )
    use_reranking: bool = Field(default=True, description="Use cross-encoder reranking")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    sort_by: QuerySortBy = Field(default=QuerySortBy.RELEVANCE, description="Sort order")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API requests."""
        return {
            "limit": self.limit,
            "offset": self.offset,
            "hybrid": self.use_hybrid_search,
            "rerank": self.use_reranking,
            "filters": self.filters,
            "sort": self.sort_by.value,
        }

    class Config:
        json_schema_extra = {
            "example": {
                "limit": 10,
                "offset": 0,
                "use_hybrid_search": True,
                "use_reranking": True,
            }
        }


class SearchResult(BaseModel):
    """Document search results."""

    documents: List[Document] = Field(..., description="List of matching documents")
    total: int = Field(..., ge=0, description="Total number of matches")
    page: int = Field(default=1, ge=1, description="Current page number")
    limit: int = Field(..., ge=1, description="Results per page")
    query_time_ms: Optional[int] = Field(None, description="Query execution time")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [],
                "total": 42,
                "page": 1,
                "limit": 10,
                "query_time_ms": 125,
            }
        }


class ChatMessage(BaseModel):
    """Chat message."""

    role: str = Field(..., pattern="^(user|assistant|system)$", description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is machine learning?",
            }
        }


class ChatSession(BaseModel):
    """Chat session metadata."""

    id: str = Field(..., description="Session identifier")
    title: Optional[str] = Field(None, description="Session title")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    message_count: int = Field(default=0, ge=0, description="Number of messages")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "chat-xyz789",
                "title": "ML Discussion",
                "created_at": "2024-01-15T10:30:00Z",
                "message_count": 5,
            }
        }


class BulkUploadResult(BaseModel):
    """Bulk document upload result."""

    uploaded: int = Field(..., ge=0, description="Number of successfully uploaded documents")
    failed: int = Field(..., ge=0, description="Number of failed uploads")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    documents: List[Document] = Field(
        default_factory=list, description="Uploaded document metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "uploaded": 5,
                "failed": 1,
                "errors": ["File too large: document.pdf"],
            }
        }


class APIHealth(BaseModel):
    """API health status."""

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health statuses")
    timestamp: str = Field(..., description="Health check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "database": "healthy",
                    "cache": "healthy",
                    "vector_store": "healthy",
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
