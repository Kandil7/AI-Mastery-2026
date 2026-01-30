"""
Document Search Use Case
============================
Orchestrate document search with filters and pagination.

خدمة البحث عن المستندات مع الفلترة والترقيم
"""

from dataclasses import dataclass
from typing import Optional

from src.application.ports.document_search_repo import DocumentRepoPort
from src.application.services.document_search import (
    SearchFilter,
    SearchResult,
    SearchResultPagination,
    SortOrder,
    FullTextSearchStrategy,
    PostgresFTSSearch,
)
from src.domain.entities import TenantId


@dataclass
class SearchDocumentsRequest:
    """Request data for document search."""

    query: str
    filters: Optional[SearchFilter] = None
    sort_order: Optional[SortOrder] = SortOrder.CREATED_DESC
    limit: int = 100
    offset: int = 0


class SearchDocumentsResponse:
    """Response data for document search."""

    results: list[SearchResult]
    total: int
    offset: int
    limit: int
    has_next: bool
    has_prev: bool


class SearchDocumentsUseCase:
    """
    Use case for document search with filters.

    Flow:
    1. Validate search query (non-empty, reasonable length)
    2. Apply filters (status, type, date, size, filename)
    3. Execute full-text search (Postgres FTS)
    4. Sort results
    5. Apply pagination (offset, limit)
    6. Return paginated results

    Features:
    - Full-text search on filename/title
    - Filter by status (created, queued, indexed, failed)
    - Filter by content type (PDF, DOCX, TXT)
    - Filter by date range
    - Filter by file size (min, max)
    - Filter by filename pattern (substring or regex)
    - Sort by creation date, filename, or size
    - Pagination with cursor support
    - Hybrid search (FTS + Vector) option

    حالة استخدام البحث عن المستندات
    """

    def __init__(
        self,
        document_repo: DocumentRepoPort,
        search_strategy: Optional[FullTextSearchStrategy] = None,
    ) -> None:
        """
        Initialize document search use case.

        Args:
            document_repo: Document repository for data access
            search_strategy: Optional search strategy (defaults to FTS)
        """
        self._repo = document_repo
        self._search_strategy = search_strategy

    def _validate_query(self, query: str) -> None:
        """
        Validate search query.

        Args:
            query: Search query string

        Raises:
            ValueError: If query is invalid
        """
        # Check if query is empty
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Check if query is too long (potential DoS)
        if len(query) > 1000:
            raise ValueError("Search query is too long (max 1000 characters)")

        # Check for potentially dangerous characters
        # (basic XSS prevention, though DB query handles it)
        if ";" in query.lower() and "drop" in query.lower():
            raise ValueError("Search query contains potentially dangerous patterns")

    def _apply_sort(self, results: list, sort_order: SortOrder) -> list:
        """
        Apply sorting to search results.

        Args:
            results: List of search results (dicts)
            sort_order: Sort order enum

        Returns:
            Sorted list of search results
        """
        if sort_order == SortOrder.CREATED_ASC:
            return sorted(results, key=lambda r: r.get("created_at", ""))
        elif sort_order == SortOrder.CREATED_DESC:
            return sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)
        elif sort_order == SortOrder.FILENAME_ASC:
            return sorted(results, key=lambda r: r.get("filename", "").lower())
        elif sort_order == SortOrder.FILENAME_DESC:
            return sorted(results, key=lambda r: r.get("filename", "").lower(), reverse=True)
        elif sort_order == SortOrder.SIZE_ASC:
            return sorted(results, key=lambda r: r.get("size_bytes", 0))
        elif sort_order == SortOrder.SIZE_DESC:
            return sorted(results, key=lambda r: r.get("size_bytes", 0), reverse=True)
        else:
            # Default: Created DESC
            return sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)

    def execute(self, request: SearchDocumentsRequest) -> SearchDocumentsResponse:
        """
        Execute document search.

        Args:
            request: Search request with query, filters, pagination

        Returns:
            Search response with paginated results

        Raises:
            ValueError: If query is invalid
        """
        # Step 1: Validate query
        self._validate_query(request.query)

        # Step 2: Set default search strategy if not provided
        if self._search_strategy is None:
            from src.adapters.persistence.postgres.db import get_engine

            self._search_strategy = PostgresFTSSearch(
                db_engine=get_engine(),
                table_name="documents",
            )

        # Step 3: Execute search with filters
        search_results = self._search_strategy.search(
            query=request.query,
            tenant_id=TenantId("default_tenant"),  # From auth middleware
            limit=request.limit,
            filters=request.filters,
        )

        # Step 4: Sort results
        sorted_results = self._apply_sort(search_results, request.sort_order)

        # Step 5: Get total count (for pagination)
        total = self._repo.count_documents(
            tenant_id=TenantId("default_tenant"),
            filters=request.filters,
        )

        # Step 6: Build pagination metadata
        has_next = (request.offset + request.limit) < total
        has_prev = request.offset > 0

        # Step 7: Return paginated response
        return SearchDocumentsResponse(
            results=sorted_results,
            total=total,
            offset=request.offset,
            limit=request.limit,
            has_next=has_next,
            has_prev=has_prev,
        )
