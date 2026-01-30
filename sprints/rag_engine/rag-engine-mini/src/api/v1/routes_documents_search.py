"""
Enhanced Document Routes - Search Extension
==========================================
Document endpoints with advanced search and pagination.

نقاط نهاية المستندات - بحث متقدم
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, Optional

from src.api.v1.deps import get_tenant_id
from src.application.use_cases.search_documents import (
    SearchDocumentsUseCase,
    SearchDocumentsRequest,
    SearchDocumentsResponse,
)
from src.application.services.document_search import (
    SearchFilter,
    SortOrder,
    DocumentStatus,
)
from src.core.bootstrap import get_container

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# ============================================================================
# Request/Response Models
# ============================================================================


class AdvancedSearchRequest(BaseModel):
    """Request model for advanced document search."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")

    # Filters
    status: Optional[DocumentStatus] = Field(None, description="Filter by document status")
    content_type: Optional[str] = Field(None, description="Filter by content type (MIME)")
    created_after: Optional[str] = Field(None, description="Created after (ISO 8601)")
    created_before: Optional[str] = Field(None, description="Created before (ISO 8601)")
    min_size: Optional[int] = Field(None, ge=0, description="Min file size (bytes)")
    max_size: Optional[int] = Field(None, ge=0, description="Max file size (bytes)")
    filename_contains: Optional[str] = Field(None, description="Filename contains substring")
    filename_regex: Optional[str] = Field(None, description="Filename regex pattern")

    # Pagination
    limit: int = Field(100, ge=1, le=1000, description="Results per page")
    offset: int = Field(0, ge=0, description="Pagination offset")

    # Sorting
    sort: Optional[SortOrder] = Field(
        None,
        description="Sort order (created_asc, created_desc, filename_asc, filename_desc, size_asc, size_desc)",
    )


class AdvancedSearchResponse(BaseModel):
    """Response model for advanced document search."""

    total: int = Field(..., description="Total matching documents")
    limit: int = Field(..., description="Results per page")
    offset: int = Field(..., description="Current offset")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")
    results: list = Field(..., description="Search results")


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/search", response_model=AdvancedSearchResponse)
async def search_documents_advanced(
    request: AdvancedSearchRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> AdvancedSearchResponse:
    """
    Advanced document search with filters and pagination.

    Features:
    - Full-text search on filename/title (Postgres FTS)
    - Filter by status, type, date, size, filename
    - Sort by any field
    - Pagination with next/prev links
    - Faceted search preparation (counts by category)

    Args:
        request: Search request with query, filters, pagination

    Returns:
        Paginated search results with metadata

    Usage:
        POST /api/v1/documents/search
        {
            "query": "report",
            "status": "indexed",
            "sort": "created_desc",
            "limit": 20,
            "offset": 0
        }
    """
    # Build search filters
    from datetime import datetime

    filters = SearchFilter()

    # Apply provided filters
    if request.status:
        filters.status = request.status

    if request.content_type:
        filters.content_type = request.content_type

    if request.created_after:
        try:
            filters.created_after = datetime.fromisoformat(request.created_after)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid created_after date format")

    if request.created_before:
        try:
            filters.created_before = datetime.fromisoformat(request.created_before)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid created_before date format")

    if request.min_size:
        filters.min_size = request.min_size

    if request.max_size:
        filters.max_size = request.max_size

    if request.filename_contains:
        filters.filename_contains = request.filename_contains

    if request.filename_regex:
        filters.filename_regex = request.filename_regex

    # Build search request
    search_request = SearchDocumentsRequest(
        query=request.query,
        filters=filters,
        sort_order=request.sort or SortOrder.CREATED_DESC,
        limit=request.limit,
        offset=request.offset,
    )

    # Get use case from container
    container = get_container()
    use_case = container.get("search_documents_use_case")

    if not use_case:
        raise HTTPException(status_code=501, detail="Search documents use case not configured")

    # Execute search
    response = use_case.execute(search_request)

    return AdvancedSearchResponse(
        total=response.total,
        limit=response.limit,
        offset=response.offset,
        has_next=response.has_next,
        has_prev=response.has_prev,
        results=[
            {
                "document_id": r.document_id,
                "filename": r.filename,
                "status": r.status,
                "size_bytes": r.size_bytes,
                "content_type": r.content_type,
                "created_at": r.created_at,
                "chunks_count": r.chunks_count,
                "matches_filters": r.matches_filters,
            }
            for r in response.results
        ],
    )


@router.get("/facets")
async def get_search_facets(
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Get search facets (counts by category).

    Returns:
        - Count by document status (created, queued, indexed, failed)
        - Count by content type (PDF, DOCX, TXT)
        - Count by file size ranges
        - Date ranges (min/max created)

    Usage:
        GET /api/v1/documents/facets
    """
    # Get use case from container
    container = get_container()
    use_case = container.get("search_documents_use_case")

    if not use_case:
        raise HTTPException(status_code=501, detail="Search documents use case not configured")

    # Get all documents (with reasonable limit for facets)
    search_request = SearchDocumentsRequest(
        query="",  # Empty query to get all
        limit=10000,  # High limit for facets
        offset=0,
    )

    response = use_case.execute(search_request)

    # Calculate facets
    facets = {
        "status": {},
        "content_type": {},
        "size_ranges": {
            "small": 0,  # < 1MB
            "medium": 0,  # 1-10MB
            "large": 0,  # 10-100MB
            "xlarge": 0,  # > 100MB
        },
        "date_range": {
            "min": None,
            "max": None,
        },
    }

    # Count by status
    for r in response.results:
        status = r.status
        facets["status"][status] = facets["status"].get(status, 0) + 1

    # Count by content type
    for r in response.results:
        content_type = r.content_type
        facets["content_type"][content_type] = facets["content_type"].get(content_type, 0) + 1

    # Count by size ranges
    for r in response.results:
        size = r.size_bytes
        if size < 1024 * 1024:  # < 1MB
            facets["size_ranges"]["small"] += 1
        elif size < 1024 * 1024 * 10:  # 1-10MB
            facets["size_ranges"]["medium"] += 1
        elif size < 1024 * 1024 * 100:  # 10-100MB
            facets["size_ranges"]["large"] += 1
        else:  # > 100MB
            facets["size_ranges"]["xlarge"] += 1

    # Date range
    if response.results:
        dates = [r.created_at for r in response.results if r.created_at]
        facets["date_range"]["min"] = min(dates)
        facets["date_range"]["max"] = max(dates)

    return facets
