"""
RAG Engine SDK - Client Module

Main client implementation for interacting with RAG Engine API.
"""

from typing import Optional, List, Dict, Any, Union, AsyncGenerator
import httpx
from contextlib import asynccontextmanager

from .models import Answer, Document, QueryHistoryItem, QueryOptions, SearchResult, DocumentStatus
from .exceptions import (
    RAGEngineError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ServerError,
)


class RAGClient:
    """
    Async HTTP client for RAG Engine API.

    This client provides methods for all RAG Engine operations including:
    - Asking questions (RAG queries)
    - Document management (upload, search, delete)
    - Chat sessions
    - Query history

    Example:
        >>> client = RAGClient(api_key="your-api-key")
        >>> answer = await client.ask("What is RAG?")
        >>> print(answer.text)

        >>> # Use as context manager for automatic cleanup
        >>> async with RAGClient(api_key="your-api-key") as client:
        ...     answer = await client.ask("What is RAG?")
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        """
        Initialize RAG Engine client.

        Args:
            api_key: API key for authentication
            base_url: Base URL of RAG Engine API (default: http://localhost:8000)
            timeout: Request timeout in seconds (default: 30.0)

        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RAGClient":
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Rate limit exceeded", retry_after=int(retry_after) if retry_after else None
            )
        elif response.status_code == 422:
            raise ValidationError(f"Validation error: {response.text}")
        elif response.status_code >= 500:
            raise ServerError(f"Server error: {response.status_code}")
        elif not response.is_success:
            raise RAGEngineError(f"HTTP {response.status_code}: {response.text}")

    async def ask(
        self,
        question: str,
        k: int = 5,
        use_hybrid: bool = True,
        rerank: bool = True,
    ) -> Answer:
        """
        Ask a question and get a RAG-powered answer.

        Args:
            question: The question to ask
            k: Number of documents to retrieve (default: 5)
            use_hybrid: Use hybrid search (vector + keyword) (default: True)
            rerank: Use reranking for better results (default: True)

        Returns:
            Answer object containing the response and sources

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            ValidationError: If request is invalid
            ServerError: If server error occurs

        Example:
            >>> answer = await client.ask("What is machine learning?")
            >>> print(answer.text)
            >>> print(f"Sources: {answer.sources}")
        """
        client = await self._get_client()

        response = await client.post(
            f"{self.base_url}/api/v1/ask",
            json={
                "question": question,
                "k": k,
                "use_hybrid": use_hybrid,
                "rerank": rerank,
            },
        )

        if not response.is_success:
            self._handle_error(response)

        data = response.json()
        return Answer(
            text=data.get("answer", ""),
            sources=data.get("sources", []),
            retrieval_k=data.get("retrieval_k", k),
            embed_ms=data.get("embed_ms"),
            search_ms=data.get("search_ms"),
            llm_ms=data.get("llm_ms"),
        )

    async def upload_document(
        self,
        file_path: str,
        title: Optional[str] = None,
    ) -> Document:
        """
        Upload a document to the RAG system.

        Args:
            file_path: Path to the file to upload
            title: Optional title for the document

        Returns:
            Document object with metadata

        Example:
            >>> doc = await client.upload_document("./report.pdf", title="Annual Report")
            >>> print(f"Document ID: {doc.id}")
        """
        client = await self._get_client()

        import os

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            files = {"file": (filename, f)}
            data = {"title": title or filename}

            response = await client.post(
                f"{self.base_url}/api/v1/documents",
                files=files,
                data=data,
            )

        if not response.is_success:
            self._handle_error(response)

        doc_data = response.json()
        return Document(
            id=doc_data["id"],
            filename=doc_data["filename"],
            content_type=doc_data.get("content_type", "application/octet-stream"),
            size_bytes=doc_data.get("size_bytes", 0),
            status=DocumentStatus(doc_data.get("status", "created")),
            created_at=doc_data["created_at"],
            updated_at=doc_data.get("updated_at"),
            title=doc_data.get("title"),
        )

    async def search_documents(
        self,
        query: str,
        options: Optional[QueryOptions] = None,
    ) -> SearchResult:
        """
        Search for documents.

        Args:
            query: Search query string
            options: Optional query options (limit, filters, etc.)

        Returns:
            SearchResult with documents and metadata

        Example:
            >>> results = await client.search_documents("machine learning")
            >>> for doc in results.documents:
            ...     print(f"{doc.filename}: {doc.status}")
        """
        client = await self._get_client()

        params = {"q": query}
        if options:
            params.update(options.to_dict())

        response = await client.get(
            f"{self.base_url}/api/v1/documents/search",
            params=params,
        )

        if not response.is_success:
            self._handle_error(response)

        data = response.json()
        documents = [
            Document(
                id=d["id"],
                filename=d["filename"],
                content_type=d.get("content_type", "application/octet-stream"),
                size_bytes=d.get("size_bytes", 0),
                status=DocumentStatus(d.get("status", "created")),
                created_at=d["created_at"],
                updated_at=d.get("updated_at"),
                title=d.get("title"),
            )
            for d in data.get("documents", [])
        ]

        return SearchResult(
            documents=documents,
            total=data.get("total", 0),
            page=data.get("page", 1),
            limit=data.get("limit", 10),
        )

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if deleted successfully

        Example:
            >>> success = await client.delete_document("doc-123")
            >>> print(f"Deleted: {success}")
        """
        client = await self._get_client()

        response = await client.delete(
            f"{self.base_url}/api/v1/documents/{document_id}",
        )

        if not response.is_success:
            self._handle_error(response)

        return response.status_code == 204 or response.status_code == 200

    async def get_query_history(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[QueryHistoryItem]:
        """
        Get query history.

        Args:
            limit: Maximum number of items to return (default: 20)
            offset: Number of items to skip (default: 0)

        Returns:
            List of QueryHistoryItem objects

        Example:
            >>> history = await client.get_query_history(limit=10)
            >>> for item in history:
            ...     print(f"{item.timestamp}: {item.question}")
        """
        client = await self._get_client()

        response = await client.get(
            f"{self.base_url}/api/v1/queries/history",
            params={"limit": limit, "offset": offset},
        )

        if not response.is_success:
            self._handle_error(response)

        data = response.json()
        items = data if isinstance(data, list) else data.get("items", [])

        return [
            QueryHistoryItem(
                question=item["question"],
                answer=item["answer"],
                sources=item.get("sources", []),
                timestamp=item["timestamp"],
                success=item.get("success", True),
            )
            for item in items
        ]


# Synchronous wrapper for convenience
def create_client(api_key: str, **kwargs) -> RAGClient:
    """
    Create a new RAGClient instance.

    This is a convenience function for creating clients.
    For async usage, use the class directly.

    Args:
        api_key: API key for authentication
        **kwargs: Additional arguments passed to RAGClient

    Returns:
        RAGClient instance
    """
    return RAGClient(api_key=api_key, **kwargs)
