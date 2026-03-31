"""
API Module

FastAPI REST API for RAG system:
- Query endpoints
- Indexing endpoints
- Streaming support
"""

from .service import (
    app,
    QueryRequest,
    QueryResponse,
    IndexRequest,
    IndexStatus,
    HealthResponse,
)

__all__ = [
    # FastAPI App
    "app",
    # Models
    "QueryRequest",
    "QueryResponse",
    "IndexRequest",
    "IndexStatus",
    "HealthResponse",
]
