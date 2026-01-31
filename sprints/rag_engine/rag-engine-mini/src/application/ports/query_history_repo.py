"""
Query History Repository Port
=============================
Interface for storing and retrieving user query history.

منفذ سجل الاستفسارات للمستخدمين
"""

from typing import List, Optional, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum


class QueryStatus(str, Enum):
    """Status of a query execution."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@runtime_checkable
class QueryHistoryRepoPort(Protocol):
    """Repository for query history operations."""

    def save_query(
        self,
        *,
        tenant_id: str,
        user_id: str,
        question: str,
        answer: str,
        sources: List[str],
        status: QueryStatus,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Save a query to history.

        Args:
            tenant_id: Tenant/user ID
            user_id: User ID
            question: User's question
            answer: System's answer
            sources: List of chunk/document IDs used
            status: Query execution status
            metadata: Optional metadata (model, tokens, latency)

        Returns:
            Query history record ID
        """
        ...

    def get_query_history(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """
        Get paginated query history for a user.

        Args:
            tenant_id: Tenant/user ID
            user_id: User ID
            limit: Number of results to return
            offset: Pagination offset

        Returns:
            List of query history records
        """
        ...

    def get_query_by_id(
        self,
        *,
        tenant_id: str,
        query_id: str,
    ) -> dict | None:
        """
        Get a specific query by ID.

        Args:
            tenant_id: Tenant/user ID
            query_id: Query history record ID

        Returns:
            Query history record or None if not found
        """
        ...

    def get_recent_queries(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        Get recent queries (last N queries).

        Useful for:
        - Quick access to common queries
        - Search suggestions
        - Analytics

        Args:
            tenant_id: Tenant/user ID
            user_id: User ID
            limit: Number of recent queries

        Returns:
            List of recent query records
        """
        ...

    def delete_query(
        self,
        *,
        tenant_id: str,
        query_id: str,
    ) -> bool:
        """
        Delete a query from history.

        Args:
            tenant_id: Tenant/user ID
            query_id: Query history record ID

        Returns:
            True if deleted, False otherwise
        """
        ...

    def delete_queries_by_date(
        self,
        *,
        tenant_id: str,
        user_id: str,
        before_date: datetime,
    ) -> int:
        """
        Delete all queries before a date.

        Useful for:
        - GDPR right to be forgotten
        - Privacy compliance
        - Cleanup old data

        Args:
            tenant_id: Tenant/user ID
            user_id: User ID
            before_date: Delete queries older than this date

        Returns:
            Number of deleted records
        """
        ...

    def get_query_analytics(
        self,
        *,
        tenant_id: str,
        user_id: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> dict:
        """
        Get query analytics for a user.

        Returns:
        {
            "total_queries": int,
            "success_rate": float,
            "avg_response_time_ms": float,
            "top_questions": List[str],
            "top_failures": List[str],
            "daily_stats": dict,
        }
        """
        ...
