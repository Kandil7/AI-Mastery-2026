"""
RAG Engine Python SDK

Official Python SDK for RAG Engine Mini - A production-ready RAG system.

Basic Usage:
    >>> from rag_engine import RAGClient
    >>>
    >>> # Initialize client
    >>> client = RAGClient(api_key="your-api-key")
    >>>
    >>> # Ask a question
    >>> answer = await client.ask("What is RAG?")
    >>> print(answer.text)

Advanced Usage:
    >>> from rag_engine import RAGClient, QueryOptions
    >>>
    >>> client = RAGClient(api_key="your-api-key")
    >>>
    >>> # Upload document
    >>> doc = await client.upload_document(
    ...     file_path="./document.pdf",
    ...     title="My Document"
    ... )
    >>>
    >>> # Search with options
    >>> results = await client.search_documents(
    ...     query="machine learning",
    ...     options=QueryOptions(limit=10, use_hybrid_search=True)
    ... )

Async Context Manager:
    >>> async with RAGClient(api_key="your-api-key") as client:
    ...     answer = await client.ask("What is RAG?")
    ...     # Client automatically closes

For more information, visit: https://rag-engine.readthedocs.io/
"""

__version__ = "1.0.0"
__author__ = "RAG Engine Team"
__license__ = "MIT"

from .client import RAGClient
from .models import (
    Answer,
    Document,
    QueryHistoryItem,
    QueryOptions,
    SearchResult,
    DocumentStatus,
    QuerySortBy,
)
from .exceptions import (
    RAGEngineError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServerError,
)

__all__ = [
    "RAGClient",
    "Answer",
    "Document",
    "QueryHistoryItem",
    "QueryOptions",
    "SearchResult",
    "DocumentStatus",
    "QuerySortBy",
    "RAGEngineError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "ServerError",
]
