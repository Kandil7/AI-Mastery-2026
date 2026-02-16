# Python SDK for RAG Engine
# ==============================

"""
RAG Engine Python SDK
======================

Official Python SDK for RAG Engine API.

RAG Engine Python SDK الرسمي
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import httpx
import asyncio
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""

    CREATED = "created"
    INDEXED = "indexed"
    FAILED = "failed"


class QuerySortBy(str, Enum):
    """Query sorting options."""

    CREATED = "created"
    UPDATED = "updated"
    FILENAME = "filename"
    SIZE = "size"


@dataclass
class Document:
    """Document metadata."""

    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    created_at: str
    updated_at: Optional[str] = None


@dataclass
class Answer:
    """RAG answer with sources."""

    text: str
    sources: List[str]
    retrieval_k: int
    embed_ms: Optional[int] = None
    search_ms: Optional[int] = None
    llm_ms: Optional[int] = None


@dataclass
class QueryHistoryItem:
    """Query history item."""

    question: str
    answer: str
    sources: List[str]
    timestamp: str
    success: bool


class RAGClient:
    """
    RAG Engine API client.

    عميل API لـ RAG Engine
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.rag-engine.com",
        timeout: float = 30.0,
    ):
        """
        Initialize RAG Engine client.

        Args:
            api_key: API key for authentication
            base_url: Base URL of RAG Engine API
            timeout: Request timeout in seconds

        تهيئة عميل RAG Engine
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # -------------------------------------------------------------------------
    # Synchronous API
    # -------------------------------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._get_headers(),
            )
        return self._client

    def ask(
        self,
        question: str,
        k: int = 5,
        document_id: Optional[str] = None,
        expand_query: bool = False,
    ) -> Answer:
        """
        Ask a question to the RAG engine.

        Args:
            question: Question to ask
            k: Number of chunks to retrieve (default: 5)
            document_id: Optional document ID for chat mode
            expand_query: Use query expansion (default: False)

        Returns:
            Answer object with text and sources

        طرح سؤال على محرك RAG
        """
        client = self._get_sync_client()

        response = client.post(
            "/api/v1/ask",
            json={
                "question": question,
                "k": k,
                "document_id": document_id,
                "expand_query": expand_query,
            },
        )
        response.raise_for_status()

        data = response.json()
        return Answer(
            text=data["answer"],
            sources=data.get("sources", []),
            retrieval_k=data.get("retrieval_k", 0),
            embed_ms=data.get("embed_ms"),
            search_ms=data.get("search_ms"),
            llm_ms=data.get("llm_ms"),
        )

    def upload_document(
        self,
        file_path: str,
        filename: Optional[str] = None,
    ) -> Document:
        """
        Upload a document to the RAG engine.

        Args:
            file_path: Path to file to upload
            filename: Optional custom filename

        Returns:
            Document metadata

        رفع مستند إلى محرك RAG
        """
        client = self._get_sync_client()

        with open(file_path, "rb") as f:
            files = {
                "file": (filename or file_path.split("/")[-1], f),
            }
            response = client.post(
                "/api/v1/documents",
                files=files,
            )

        response.raise_for_status()

        data = response.json()
        return Document(
            id=data["id"],
            filename=data["filename"],
            content_type=data["content_type"],
            size_bytes=data["size_bytes"],
            status=DocumentStatus(data["status"]),
            created_at=data["created_at"],
        )

    def search_documents(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: QuerySortBy = QuerySortBy.CREATED,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Document]:
        """
        Search for documents.

        Args:
            query: Search query
            k: Number of results (default: 10)
            filters: Optional filters (status, type, date, size)
            sort_by: Sort order (default: created)
            limit: Max results to return (default: 20)
            offset: Pagination offset (default: 0)

        Returns:
            List of matching documents

        البحث عن المستندات
        """
        client = self._get_sync_client()

        params = {
            "query": query,
            "k": k,
            "sort_by": sort_by.value,
            "limit": limit,
            "offset": offset,
        }

        if filters:
            params.update(filters)

        response = client.get(
            "/api/v1/documents/search",
            params=params,
        )
        response.raise_for_status()

        data = response.json()
        return [
            Document(
                id=d["id"],
                filename=d["filename"],
                content_type=d["content_type"],
                size_bytes=d["size_bytes"],
                status=DocumentStatus(d["status"]),
                created_at=d["created_at"],
                updated_at=d.get("updated_at"),
            )
            for d in data["results"]
        ]

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.

        Args:
            document_id: ID of document to delete

        Returns:
            True if successful

        حذف مستند
        """
        client = self._get_sync_client()
        response = client.delete(f"/api/v1/documents/{document_id}")
        response.raise_for_status()
        return True

    def get_query_history(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[QueryHistoryItem]:
        """
        Get query history.

        Args:
            limit: Max history items (default: 50)
            offset: Pagination offset (default: 0)

        Returns:
            List of query history items

        الحصول على سجل الاستعلامات
        """
        client = self._get_sync_client()

        response = client.get(
            "/api/v1/queries/history",
            params={
                "limit": limit,
                "offset": offset,
            },
        )
        response.raise_for_status()

        data = response.json()
        return [
            QueryHistoryItem(
                question=q["question"],
                answer=q.get("answer", ""),
                sources=q.get("sources", []),
                timestamp=q["timestamp"],
                success=q.get("success", True),
            )
            for q in data["questions"]
        ]

    # -------------------------------------------------------------------------
    # Asynchronous API
    # -------------------------------------------------------------------------

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._get_headers(),
            )
        return self._async_client

    async def ask_async(
        self,
        question: str,
        k: int = 5,
        document_id: Optional[str] = None,
        expand_query: bool = False,
    ) -> Answer:
        """Async version of ask()."""
        client = self._get_async_client()

        response = await client.post(
            "/api/v1/ask",
            json={
                "question": question,
                "k": k,
                "document_id": document_id,
                "expand_query": expand_query,
            },
        )
        response.raise_for_status()

        data = response.json()
        return Answer(
            text=data["answer"],
            sources=data.get("sources", []),
            retrieval_k=data.get("retrieval_k", 0),
            embed_ms=data.get("embed_ms"),
            search_ms=data.get("search_ms"),
            llm_ms=data.get("llm_ms"),
        )

    async def upload_document_async(
        self,
        file_path: str,
        filename: Optional[str] = None,
    ) -> Document:
        """Async version of upload_document()."""
        client = self._get_async_client()

        with open(file_path, "rb") as f:
            files = {
                "file": (filename or file_path.split("/")[-1], f),
            }
            response = await client.post(
                "/api/v1/documents",
                files=files,
            )

        response.raise_for_status()

        data = response.json()
        return Document(
            id=data["id"],
            filename=data["filename"],
            content_type=data["content_type"],
            size_bytes=data["size_bytes"],
            status=DocumentStatus(data["status"]),
            created_at=data["created_at"],
        )

    def close(self):
        """Close HTTP clients and cleanup resources."""
        if self._client:
            self._client.close()
        if self._async_client:
            import asyncio

            asyncio.create_task(self._async_client.aclose())


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize client
    client = RAGClient(
        api_key="sk_your_api_key_here",
        base_url="http://localhost:8000",
    )

    # Sync usage
    print("=== Sync Usage ===")
    answer = client.ask("What is RAG?", k=5)
    print(f"Answer: {answer.text}")
    print(f"Sources: {answer.sources}")

    # Async usage
    async def async_example():
        print("\n=== Async Usage ===")
        answer = await client.ask_async("How does vector search work?", k=10)
        print(f"Answer: {answer.text}")
        print(f"Sources: {answer.sources}")

    asyncio.run(async_example())

    # Cleanup
    client.close()
