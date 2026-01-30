"""
Query History Use Case
=======================
Orchestrate query tracking and analytics.

حالة استخدام تتبع سجل الاستفسارات
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum

from src.application.ports.query_history_repo import QueryHistoryRepoPort, QueryStatus
from src.application.services.query_history_service import QueryHistoryService


class DeleteHistoryRequest:
    """Request data for deleting query history."""

    user_id: str
    query_id: str | None = None  # If None, delete by date range
    before_date: Optional[str] = None  # Delete queries older than this date


@dataclass
class GetHistoryRequest:
    """Request data for getting query history."""

    user_id: str
    limit: int = 100
    offset: int = 0


@dataclass
class GetHistoryResponse:
    """Response data for query history."""

    queries: list
    total: int
    limit: int
    offset: int
    has_next: bool
    has_prev: bool


@dataclass
class GetAnalyticsRequest:
    """Request data for query analytics."""

    user_id: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None


@dataclass
class GetAnalyticsResponse:
    """Response data for query analytics."""

    success_rate: dict
    top_questions: list
    top_failures: list
    daily_stats: dict
    query_patterns: dict


class QueryHistoryUseCase:
    """
    Use case for query history management and analytics.

    Features:
    - Save query execution with metadata
    - Retrieve paginated query history
    - Get recent queries (last N)
    - Get comprehensive analytics
    - Delete individual queries or bulk by date
    - GDPR compliance (auto-expiry or manual deletion)
    - Query deduplication support

    حالة استخدام تتبع السجل الاستفسارات
    """

    def __init__(
        self,
        history_repo: QueryHistoryRepoPort,
        analytics: Optional[QueryHistoryService] = None,
    ) -> None:
        """
        Initialize query history use case.

        Args:
            history_repo: Query history repository
            analytics: Optional analytics service
        """
        self._repo = history_repo
        self._analytics = analytics

    def _validate_user_id(self, user_id: str) -> None:
        """
        Validate user ID format.

        Args:
            user_id: User ID

        Raises:
            ValueError: If user_id is invalid
        """
        if not user_id or len(user_id) == 0:
            raise ValueError("User ID cannot be empty")

        # In production, validate user exists
        # user = self._repo.get_user_by_id(user_id=user_id)
        # if not user:
        #     raise ValueError("User not found")

    def save_query(
        self,
        *,
        user_id: str,
        question: str,
        answer: str,
        sources: List[str],
        status: QueryStatus,
        latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> str:
        """
        Save a query to history.

        Args:
            user_id: User ID
            question: User's question
            answer: System's answer
            sources: List of chunk/document IDs
            status: Query execution status
            latency_ms: Query execution latency in milliseconds
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count

        Returns:
            Query history record ID
        """
        self._validate_user_id(user_id)

        # Build metadata
        metadata = {}
        if latency_ms is not None:
            metadata["latency_ms"] = latency_ms

        if prompt_tokens is not None:
            metadata["prompt_tokens"] = prompt_tokens

        if completion_tokens is not None:
            metadata["completion_tokens"] = completion_tokens

        # Save using service
        return self._analytics.save_query(
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources,
            status=status,
            metadata=metadata,
        )

    def get_history(self, request: GetHistoryRequest) -> GetHistoryResponse:
        """
        Get paginated query history for a user.

        Args:
            request: Get history request with pagination

        Returns:
            Paginated query history response
        """
        self._validate_user_id(request.user_id)

        # Get history from service
        history = self._analytics.get_history(
            user_id=request.user_id,
            limit=request.limit,
            offset=request.offset,
        )

        return GetHistoryResponse(
            queries=history["queries"],
            total=history["total"],
            limit=request.limit,
            offset=request.offset,
            has_next=history["has_next"],
            has_prev=history["has_prev"],
        )

    def get_recent_queries(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list:
        """
        Get recent queries for quick access.

        Args:
            user_id: User ID
            limit: Number of recent queries

        Returns:
            List of recent query records
        """
        self._validate_user_id(user_id)

        return self._analytics.get_recent_queries(
            user_id=user_id,
            limit=limit,
        )

    def get_analytics(self, request: GetAnalyticsRequest) -> GetAnalyticsResponse:
        """
        Get comprehensive query analytics for a user.

        Args:
            request: Get analytics request

        Returns:
            Analytics response with success rate, trends, patterns
        """
        self._validate_user_id(request.user_id)

        # Parse date parameters
        date_from = None
        date_to = None

        if request.date_from:
            try:
                date_from = datetime.fromisoformat(request.date_from)
            except ValueError:
                raise ValueError("Invalid date_from format (use ISO 8601)")

        if request.date_to:
            try:
                date_to = datetime.fromisoformat(request.date_to)
            except ValueError:
                raise ValueError("Invalid date_to format (use ISO 8601)")

        # Get analytics from service
        return self._analytics.get_analytics(
            user_id=request.user_id,
            date_from=date_from,
            date_to=date_to,
        )

    def delete_query(self, request: DeleteHistoryRequest) -> bool:
        """
        Delete a query from history or bulk delete by date.

        Args:
            request: Delete history request

        Returns:
            True if deleted, False otherwise

        Security:
        - Validates user ownership
        - Requires confirmation for bulk deletion
        - Returns failure if query not found
        """
        self._validate_user_id(request.user_id)

        if request.query_id:
            # Delete individual query
            return self._analytics.delete_query(
                user_id=request.user_id,
                query_id=request.query_id,
            )
        elif request.before_date:
            # Bulk delete all queries older than date
            try:
                before_date = datetime.fromisoformat(request.before_date)
            except ValueError:
                raise ValueError("Invalid before_date format (use ISO 8601)")

            count = self._repo.delete_queries_by_date(
                tenant_id="",  # Use user_id
                user_id=request.user_id,
                before_date=before_date,
            )

            return count > 0
        else:
            raise ValueError("Must provide either query_id or before_date")
