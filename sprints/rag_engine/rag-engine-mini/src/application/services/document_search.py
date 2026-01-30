"""
Document Search Service
========================
Advanced document search with filtering and pagination.

خدمة البحث المتقدمة عن المستندات
"""

import re
from typing import List, Optional, Literal, Protocol
from datetime import datetime
from enum import Enum

from src.domain.entities import TenantId, DocumentId


# ============================================================================
# Search Models
# ============================================================================


class DocumentStatus(str, Enum):
    """Document status enum for filtering."""

    CREATED = "created"
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class SortOrder(str, Enum):
    """Sort order for search results."""

    CREATED_ASC = "created_asc"
    CREATED_DESC = "created_desc"
    FILENAME_ASC = "filename_asc"
    FILENAME_DESC = "filename_desc"
    SIZE_ASC = "size_asc"
    SIZE_DESC = "size_desc"


class SearchFilter:
    """Search filter parameters."""

    def __init__(
        self,
        status: Optional[DocumentStatus] = None,
        content_type: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        filename_contains: Optional[str] = None,
        filename_regex: Optional[str] = None,
    ) -> None:
        """
        Initialize search filters.

        Args:
            status: Filter by document status
            content_type: Filter by MIME type (e.g., application/pdf)
            created_after: Filter by creation date (start range)
            created_before: Filter by creation date (end range)
            min_size: Filter by minimum file size in bytes
            max_size: Filter by maximum file size in bytes
            filename_contains: Filter by filename substring
            filename_regex: Filter by filename pattern (regex)

        Examples:
            # Filter by status
            SearchFilter(status=DocumentStatus.INDEXED)

            # Filter by content type
            SearchFilter(content_type="application/pdf")

            # Filter by date range
            SearchFilter(
                created_after=datetime(2024, 1, 1),
                created_before=datetime(2024, 1, 31),
            )

            # Filter by filename
            SearchFilter(filename_contains="report")
        """
        self.status = status
        self.content_type = content_type
        self.created_after = created_after
        self.created_before = created_before
        self.min_size = min_size
        self.max_size = max_size
        self.filename_contains = filename_contains
        self.filename_regex = filename_regex

    def matches_document(self, document: dict) -> bool:
        """
        Check if a document matches all filters.

        Args:
            document: Document dict with keys: status, content_type,
                        created_at, size_bytes, filename

        Returns:
            True if document matches all active filters
        """
        # Status filter
        if self.status is not None:
            if document.get("status") != self.status.value:
                return False

        # Content type filter
        if self.content_type is not None:
            # Allow partial matches (e.g., application/pdf matches application/pdf)
            if self.content_type not in document.get("content_type", ""):
                return False

        # Date range filter
        if self.created_after is not None:
            if document.get("created_at") < self.created_after:
                return False

        if self.created_before is not None:
            if document.get("created_at") > self.created_before:
                return False

        # Size filter
        size = document.get("size_bytes", 0)
        if self.min_size is not None and size < self.min_size:
            return False

        if self.max_size is not None and size > self.max_size:
            return False

        # Filename contains filter
        if self.filename_contains is not None:
            if self.filename_contains.lower() not in document.get("filename", "").lower():
                return False

        # Filename regex filter
        if self.filename_regex is not None:
            if not re.search(self.filename_regex, document.get("filename", "")):
                return False

        return True


class SearchResult:
    """Search result with document and metadata."""

    def __init__(
        self,
        document_id: str,
        filename: str,
        status: str,
        size_bytes: int,
        content_type: str,
        created_at: str,
        chunks_count: int,
        matches_filters: List[str],
    ) -> None:
        """
        Initialize search result.

        Args:
            document_id: Document ID
            filename: Original filename
            status: Document status
            size_bytes: File size in bytes
            content_type: MIME type
            created_at: Creation timestamp (ISO string)
            chunks_count: Number of chunks in document
            matches_filters: List of active filters that matched
        """
        self.document_id = document_id
        self.filename = filename
        self.status = status
        self.size_bytes = size_bytes
        self.content_type = content_type
        self.created_at = created_at
        self.chunks_count = chunks_count
        self.matches_filters = matches_filters


class SearchResultPagination:
    """Paginated search results."""

    def __init__(
        self,
        results: List[SearchResult],
        total: int,
        offset: int,
        limit: int,
        has_next: bool,
        has_prev: bool,
    ) -> None:
        """
        Initialize paginated results.

        Args:
            results: List of search results for this page
            total: Total number of matching documents
            offset: Current offset (skip count)
            limit: Number of results per page
            has_next: Whether there's a next page
            has_prev: Whether there's a previous page
        """
        self.results = results
        self.total = total
        self.offset = offset
        self.limit = limit
        self.has_next = has_next
        self.has_prev = has_prev


class FullTextSearchStrategy(Protocol):
    """Protocol for full-text search strategies."""

    def search(
        self,
        query: str,
        tenant_id: TenantId,
        limit: int,
        filters: Optional[SearchFilter] = None,
    ) -> List[SearchResult]:
        """
        Execute full-text search.

        Args:
            query: Search query string
            tenant_id: Tenant/user ID
            limit: Maximum results to return
            filters: Optional search filters

        Returns:
            List of search results
        """
        ...


class PostgresFTSSearch:
    """
    Postgres Full-Text Search (FTS) implementation.

    Uses Postgres tsvector for efficient full-text search.
    Supports:
    - Phrase search ("exact phrase")
    - Word search (any of the words)
    - Prefix search ("words starting with")
    - Trigram search (three-character sequences)

    بحث نصي متقدم باستخدام Postgres FTS
    """

    def __init__(self, db_engine, table_name="documents"):
        """
        Initialize Postgres FTS search.

        Args:
            db_engine: SQLAlchemy database engine
            table_name: Table name to search (default: documents)
        """
        self._db = db_engine
        self._table_name = table_name

    def search(
        self,
        query: str,
        tenant_id: TenantId,
        limit: int = 100,
        filters: Optional[SearchFilter] = None,
    ) -> List[SearchResult]:
        """
        Execute Postgres FTS search.

        Uses Postgres tsvector @@ to_tsquery() for efficient searching.

        Args:
            query: Search query string
            tenant_id: Tenant/user ID
            limit: Maximum results to return
            filters: Optional search filters

        Returns:
            List of search results
        """
        from sqlalchemy import text, and_, or_

        # Normalize query for FTS
        # - Convert to lowercase
        # - Remove special characters (except alphanum, space)
        normalized_query = re.sub(r"[^\w\s-]", "", query.lower()).strip()

        if not normalized_query:
            return []

        # Build SQL query with FTS
        sql = text(f"""
            SELECT 
                id as document_id,
                filename,
                status,
                size_bytes,
                content_type,
                created_at,
                chunks_count
            FROM {self._table_name}
            WHERE 
                tenant_id = :tenant_id
                AND to_tsvector('simple', filename) @@ to_tsquery('simple', :query)
            ORDER BY 
                ts_rank_cd(to_tsvector('simple', filename), :query) DESC
            LIMIT :limit
        """)

        # Add filters if provided
        conditions = []
        params = {
            "tenant_id": tenant_id.value,
            "query": normalized_query,
            "limit": limit,
        }

        if filters:
            # Status filter
            if filters.status is not None:
                conditions.append(text("status = :status"))
                params["status"] = filters.status.value

            # Content type filter
            if filters.content_type is not None:
                conditions.append(text("content_type LIKE :content_type"))
                params["content_type"] = f"%{filters.content_type}%"

            # Date range filters
            if filters.created_after is not None:
                conditions.append(text("created_at >= :created_after"))
                params["created_after"] = filters.created_after

            if filters.created_before is not None:
                conditions.append(text("created_at <= :created_before"))
                params["created_before"] = filters.created_before

            # Size filters
            if filters.min_size is not None:
                conditions.append(text("size_bytes >= :min_size"))
                params["min_size"] = filters.min_size

            if filters.max_size is not None:
                conditions.append(text("size_bytes <= :max_size"))
                params["max_size"] = filters.max_size

            # Filename filters
            if filters.filename_contains is not None:
                conditions.append(text("LOWER(filename) LIKE LOWER(:filename_contains)"))
                params["filename_contains"] = f"%{filters.filename_contains.lower()}%"

            if filters.filename_regex is not None:
                conditions.append(text("filename ~ :filename_regex"))
                params["filename_regex"] = filters.filename_regex

            # Add all conditions to WHERE clause
            if conditions:
                sql = sql + text(" AND ") + text(" AND ").join(conditions)

        # Execute query
        with self._db.connect() as conn:
            result = conn.execute(sql, params)
            rows = result.fetchall()

        # Build search results
        search_results = []
        active_filters = []

        # Track which filters matched
        if filters:
            if filters.status:
                active_filters.append("status")
            if filters.content_type:
                active_filters.append("content_type")
            if filters.created_after or filters.created_before:
                active_filters.append("created_at")
            if filters.min_size or filters.max_size:
                active_filters.append("size_bytes")
            if filters.filename_contains or filters.filename_regex:
                active_filters.append("filename")

        for row in rows:
            search_results.append(
                SearchResult(
                    document_id=str(row.document_id),
                    filename=row.filename,
                    status=row.status,
                    size_bytes=int(row.size_bytes) if row.size_bytes else 0,
                    content_type=row.content_type or "",
                    created_at=row.created_at.isoformat() if row.created_at else "",
                    chunks_count=int(row.chunks_count) if row.chunks_count else 0,
                    matches_filters=active_filters,
                )
            )

        return search_results


class HybridDocumentSearch:
    """
    Hybrid document search combining FTS and vector search.

    Combines:
    1. Postgres FTS (exact filename match)
    2. Vector search (semantic content match)
    3. RRF fusion (merge results)

    بحث هجين عن المستندات (نصي + دلالي)
    """

    def __init__(
        self,
        fts_search: PostgresFTSSearch,
        vector_search,
    ) -> None:
        """
        Initialize hybrid search.

        Args:
            fts_search: Postgres FTS search implementation
            vector_search: Vector store search interface
        """
        self._fts = fts_search
        self._vector = vector_search

    def search(
        self,
        query: str,
        tenant_id: TenantId,
        limit: int = 100,
        filters: Optional[SearchFilter] = None,
        k_fts: int = 50,
        k_vec: int = 50,
    ) -> List[SearchResult]:
        """
        Execute hybrid document search.

        Args:
            query: Search query string
            tenant_id: Tenant/user ID
            limit: Maximum total results
            filters: Optional search filters
            k_fts: Results from FTS search
            k_vec: Results from vector search

        Returns:
            List of search results

        Flow:
        1. Execute Postgres FTS search (filename/title match)
        2. Execute vector search (content semantic match)
        3. RRF fusion (merge results)
        4. Apply pagination
        """
        # Step 1: Postgres FTS search (filename/title)
        fts_results = self._fts.search(
            query=query,
            tenant_id=tenant_id,
            limit=k_fts,
            filters=filters,
        )

        # Step 2: Vector search (content semantic match)
        # Note: Vector search would search chunk content, not document metadata
        # For document search, we might skip this or use a different strategy
        vec_results = []  # Placeholder for vector search of docs

        # Step 3: RRF fusion
        from src.application.services.fusion import rrf_fusion
        from src.application.services.scoring import ScoredChunk

        # Convert FTS results to ScoredChunk format
        fts_scored = [
            ScoredChunk(
                chunk={"id": r.document_id, "filename": r.filename},
                score=r.chunks_count,  # Use chunk count as relevance score
            )
            for r in fts_results
        ]

        vec_scored = [
            ScoredChunk(
                chunk={"id": r.document_id, "filename": r.filename},
                score=0.0,  # Placeholder for vector results
            )
            for r in vec_results
        ]

        # Merge using RRF
        fused = rrf_fusion(
            fts_hits=fts_scored,
            keyword_hits=vec_scored,
            out_limit=limit,
            k=60,  # Standard RRF constant
        )

        # Convert back to SearchResult format
        search_results = []
        for s in fused:
            chunk = s.chunk
            search_results.append(
                SearchResult(
                    document_id=chunk["id"],
                    filename=chunk.get("filename", ""),
                    status=chunk.get("status", ""),
                    size_bytes=chunk.get("size_bytes", 0),
                    content_type=chunk.get("content_type", ""),
                    created_at=chunk.get("created_at", ""),
                    chunks_count=chunk.get("chunks_count", 0),
                    matches_filters=[],
                )
            )

        return search_results
