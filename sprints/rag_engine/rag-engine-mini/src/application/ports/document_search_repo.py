"""
Document Repository Port - Search Extension
=========================================
Extended port for document search operations.

منفذ المستندات - بحث متقدم
"""

from typing import List, Optional, Protocol, runtime_checkable

from src.domain.entities import TenantId, DocumentId
from src.application.services.document_search import (
    SearchFilter,
    SearchResult,
    SearchResultPagination,
    SortOrder,
    FullTextSearchStrategy,
)


@runtime_checkable
class DocumentRepoPort(Protocol):
    """Extended document repository with search support."""

    def list_documents(
        self,
        *,
        tenant_id: TenantId,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """
        List documents for a tenant with pagination.

        Args:
            tenant_id: Tenant/user ID
            limit: Number of results
            offset: Number of results to skip

        Returns:
            List of document dicts
        """
        ...

    def search_documents(
        self,
        *,
        tenant_id: TenantId,
        query: str,
        filters: Optional[SearchFilter] = None,
        sort_order: Optional[SortOrder] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> SearchResultPagination:
        """
        Search documents with filters and pagination.

        Args:
            tenant_id: Tenant/user ID
            query: Search query string
            filters: Search filters (status, type, date, etc.)
            sort_order: Sort order (created, filename, size)
            limit: Number of results
            offset: Pagination offset

        Returns:
            Paginated search results
        """
        ...

    def count_documents(
        self, *, tenant_id: TenantId, filters: Optional[SearchFilter] = None
    ) -> int:
        """
        Count documents matching filters.

        Args:
            tenant_id: Tenant/user ID
            filters: Optional search filters

        Returns:
            Total count of matching documents
        """
        ...

    def get_document(self, *, tenant_id: TenantId, document_id: DocumentId) -> dict | None:
        """
        Get full document details by ID.

        Args:
            tenant_id: Tenant/user ID
            document_id: Document ID

        Returns:
            Document dict or None if not found
        """
        ...
