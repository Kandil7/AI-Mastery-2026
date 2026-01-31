"""
Query History Service
======================
Service for tracking and managing user query history.

خدمة تتبع وإدارة سجل الاستفسارات
"""

import time
from typing import List, Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from enum import Enum

from src.application.ports.query_history_repo import QueryStatus


class QueryAnalytics:
    """
    Service for analyzing query history and patterns.

    Analytical Capabilities:
    - Success rate calculation
    - Average response time
    - Top questions (frequency)
    - Top failures (frequency)
    - Daily query counts
    - Question patterns (length, words)
    - Query similarity (for deduplication)

    خدمة تحليلات السجل الاستفسارات
    """

    def __init__(self, history_repo):
        """
        Initialize query analytics service.

        Args:
            history_repo: Query history repository
        """
        self._repo = history_repo

    def calculate_success_rate(
        self,
        *,
        user_id: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> dict:
        """
        Calculate query success rate for a period.

        Returns:
            {
                "total_queries": int,
                "successful_queries": int,
                "success_rate": float,
                "failed_queries": int,
                "failure_rate": float,
            }
        """
        # Get all queries in date range
        queries = self._repo.get_query_history(
            tenant_id="",  # Use user_id
            user_id=user_id,
            limit=100000,  # High limit for analytics
            offset=0,
        )

        # Filter by date range
        if date_from:
            queries = [q for q in queries if q.get("created_at", datetime.utcnow()) >= date_from]

        if date_to:
            queries = [q for q in queries if q.get("created_at", datetime.utcnow()) <= date_to]

        # Calculate metrics
        total = len(queries)
        successful = len([q for q in queries if q.get("status") == QueryStatus.SUCCESS.value])
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0
        failure_rate = failed / total if total > 0 else 0.0

        # Calculate average response time (from metadata if available)
        total_time = sum(q.get("metadata", {}).get("latency_ms", 0) for q in queries)
        avg_time = total_time / total if total > 0 else 0

        return {
            "total_queries": total,
            "successful_queries": successful,
            "success_rate": round(success_rate * 100, 2),
            "failed_queries": failed,
            "failure_rate": round(failure_rate * 100, 2),
            "avg_response_time_ms": round(avg_time, 2),
        }

    def get_top_questions(
        self,
        *,
        user_id: str,
        limit: int = 10,
        include_failed: bool = False,
    ) -> List[dict]:
        """
        Get most frequently asked questions.

        Args:
            user_id: User ID
            limit: Number of top questions to return
            include_failed: Whether to include failed queries

        Returns:
            List of {question, count, last_asked} dicts
        """
        queries = self._repo.get_query_history(
            tenant_id="",
            user_id=user_id,
            limit=10000,  # Large sample
            offset=0,
        )

        # Filter by status if needed
        if not include_failed:
            queries = [q for q in queries if q.get("status") == QueryStatus.SUCCESS.value]

        # Count questions
        question_counts = Counter(q["question"] for q in queries)
        top_questions = question_counts.most_common(limit)

        # Get last asked time for each
        result = []
        for question, count in top_questions:
            question_instances = [q for q in queries if q["question"] == question]
            last_instance = max(
                question_instances, key=lambda q: q.get("created_at", datetime.utcnow())
            )

            result.append(
                {
                    "question": question,
                    "count": count,
                    "last_asked": last_instance.get("created_at", datetime.utcnow()).isoformat()
                    if last_instance.get("created_at")
                    else "",
                    "success_rate": round(
                        len(
                            [
                                q
                                for q in question_instances
                                if q.get("status") == QueryStatus.SUCCESS.value
                            ]
                        )
                        / count
                        * 100,
                        2,
                    ),
                }
            )

        return sorted(result, key=lambda x: x["count"], reverse=True)

    def get_top_failures(
        self,
        *,
        user_id: str,
        limit: int = 10,
    ) -> List[dict]:
        """
        Get most frequently failed queries.

        Useful for:
        - Identifying confusing queries
        - Improving retrieval quality
        - User education (what to ask instead)

        Args:
            user_id: User ID
            limit: Number of top failures to return

        Returns:
            List of {question, count, last_failed} dicts
        """
        queries = self._repo.get_query_history(
            tenant_id="",
            user_id=user_id,
            limit=10000,
            offset=0,
        )

        # Filter failed queries
        failed_queries = [q for q in queries if q.get("status") == QueryStatus.FAILED.value]

        # Count failures
        question_counts = Counter(q["question"] for q in failed_queries)
        top_failures = question_counts.most_common(limit)

        # Get last failed time for each
        result = []
        for question, count in top_failures:
            question_instances = [q for q in failed_queries if q["question"] == question]
            last_instance = max(
                question_instances, key=lambda q: q.get("created_at", datetime.utcnow())
            )

            result.append(
                {
                    "question": question,
                    "count": count,
                    "last_failed": last_instance.get("created_at", datetime.utcnow()).isoformat()
                    if last_instance.get("created_at")
                    else "",
                    "failure_rate": round(
                        count / len([q for q in failed_queries if q["question"] == question]) * 100,
                        2,
                    ),
                }
            )

        return sorted(result, key=lambda x: x["count"], reverse=True)

    def get_daily_stats(
        self,
        *,
        user_id: str,
        days: int = 7,
    ) -> dict:
        """
        Get daily query statistics for the last N days.

        Args:
            user_id: User ID
            days: Number of days to analyze

        Returns:
            {
                "daily_stats": List of daily summaries,
                "trend": "increasing" | "stable" | "decreasing",
            }
        """
        end_date = datetime.utcnow()
        daily_stats = []

        for day in range(days):
            start_date = end_date - timedelta(days=1)
            end_date_inclusive = end_date - timedelta(seconds=1)

            # Get queries for this day
            queries = self._repo.get_query_history(
                tenant_id="",
                user_id=user_id,
                date_from=start_date,
                date_to=end_date_inclusive,
                limit=100000,
                offset=0,
            )

            # Calculate daily metrics
            total = len(queries)
            successful = len([q for q in queries if q.get("status") == QueryStatus.SUCCESS.value])
            daily_stats.append(
                {
                    "date": start_date.strftime("%Y-%m-%d"),
                    "total_queries": total,
                    "successful_queries": successful,
                    "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
                }
            )

            end_date = start_date

        # Calculate trend (last 3 days)
        if len(daily_stats) >= 3:
            recent_success = [d["success_rate"] for d in daily_stats[-3:]]
            trend = "stable"
            if all(
                recent_success[i] > recent_success[i - 1] for i in range(1, len(recent_success))
            ):
                trend = "increasing"
            elif all(
                recent_success[i] < recent_success[i - 1] for i in range(1, len(recent_success))
            ):
                trend = "decreasing"
        else:
            trend = "insufficient_data"

        return {
            "daily_stats": daily_stats,
            "trend": trend,
        }

    def get_query_patterns(
        self,
        *,
        user_id: str,
        sample_size: int = 1000,
    ) -> dict:
        """
        Analyze query patterns (length, words, complexity).

        Args:
            user_id: User ID
            sample_size: Number of queries to analyze

        Returns:
            {
                "avg_question_length": float,
                "length_distribution": dict,
                "word_frequency": dict,
                "complexity_score": float,
            }
        """
        queries = self._repo.get_query_history(
            tenant_id="",
            user_id=user_id,
            limit=sample_size,
            offset=0,
        )

        if not queries:
            return {
                "avg_question_length": 0,
                "length_distribution": {},
                "word_frequency": {},
                "complexity_score": 0,
            }

        # Calculate average question length
        question_lengths = [len(q["question"]) for q in queries]
        avg_length = sum(question_lengths) / len(question_lengths)

        # Length distribution
        lengths = [len(q) for q in question_lengths]
        length_dist = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": avg_length,
            "median": sorted(lengths)[len(lengths) // 2],
        }

        # Word frequency analysis
        all_words = " ".join([q["question"] for q in queries]).lower().split()
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(10)

        # Complexity score (simple heuristic)
        complexity_factors = []

        # Factor 1: Question length (longer = more complex)
        length_factor = min(avg_length / 50.0, 2.0)  # Normalize to 0-2
        complexity_factors.append(length_factor)

        # Factor 2: Unique words per query (more unique = more complex)
        unique_words = set()
        for q in queries:
            words = set(q["question"].lower().split())
            unique_words.update(words)
        complexity_factors.append(len(unique_words) / 20.0)  # Normalize to 0-2

        # Factor 3: Special characters (quotes, operators, wildcards)
        special_chars = ['"', '"', "*", "?", ":", "(", ")"]
        special_count = sum(q["question"].count(c) for q in queries) / len(queries)
        complexity_factors.append(min(special_count / 5.0, 2.0))  # Normalize to 0-2

        complexity_score = (
            sum(complexity_factors) / len(complexity_factors) * 100 if complexity_factors else 0
        )

        return {
            "avg_question_length": round(avg_length, 2),
            "length_distribution": length_dist,
            "word_frequency": [{"word": w, "count": c} for w, c in top_words],
            "complexity_score": round(complexity_score, 2),
            "complexity_factors": complexity_factors,
        }


class QueryHistoryService:
    """
    Service for managing query history and analytics.

    Features:
    - Save query execution metadata
    - Retrieve query history with pagination
    - Query analytics (success rate, patterns, trends)
    - Recent queries for quick access
    - Query deduplication (detect and flag)
    - Privacy controls (auto-expiry)

    خدمة تتبع السجل الاستفسارات
    """

    def __init__(self, history_repo, analytics: Optional[QueryAnalytics] = None):
        """
        Initialize query history service.

        Args:
            history_repo: Query history repository
            analytics: Optional analytics service
        """
        self._repo = history_repo
        self._analytics = analytics or QueryAnalytics(history_repo)

    def save_query(
        self,
        *,
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
            user_id: User ID
            question: User's question
            answer: System's answer
            sources: List of chunk/document IDs used
            status: Query execution status
            metadata: Optional metadata (latency, tokens, etc.)

        Returns:
            Query history record ID
        """
        # Add timestamp to metadata if not present
        if metadata is None:
            metadata = {}

        metadata["created_at"] = datetime.utcnow().isoformat()

        # Add default metadata if not provided
        if "latency_ms" not in metadata:
            metadata["latency_ms"] = 0

        if "prompt_tokens" not in metadata:
            metadata["prompt_tokens"] = 0

        if "completion_tokens" not in metadata:
            metadata["completion_tokens"] = 0

        # Save to repository
        return self._repo.save_query(
            tenant_id="",  # Use user_id
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources,
            status=status,
            metadata=metadata,
        )

    def get_history(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """
        Get paginated query history for a user.

        Args:
            user_id: User ID
            limit: Number of results to return
            offset: Pagination offset

        Returns:
            {
                "queries": List of query records,
                "total": int,
                "limit": int,
                "offset": int,
                "has_next": bool,
            }
        """
        queries = self._repo.get_query_history(
            tenant_id="",
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        # Get total count (for pagination)
        total = self._repo.count_queries(
            tenant_id="",
            user_id=user_id,
        )

        has_next = (offset + limit) < total
        has_prev = offset > 0

        return {
            "queries": queries,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": has_next,
            "has_prev": has_prev,
        }

    def get_recent_queries(
        self,
        *,
        user_id: str,
        limit: int = 10,
    ) -> list:
        """
        Get recent queries for quick access.

        Useful for:
        - Search suggestions
        - Quick re-ask of common questions
        - User behavior analysis

        Args:
            user_id: User ID
            limit: Number of recent queries

        Returns:
            List of query records
        """
        queries = self._repo.get_recent_queries(
            tenant_id="",
            user_id=user_id,
            limit=limit,
        )

        return queries

    def delete_query(
        self,
        *,
        user_id: str,
        query_id: str,
    ) -> bool:
        """
        Delete a query from history.

        Args:
            user_id: User ID
            query_id: Query history record ID

        Returns:
            True if deleted, False otherwise
        """
        return self._repo.delete_query(
            tenant_id="",
            user_id=user_id,
            query_id=query_id,
        )

    def get_analytics(
        self,
        *,
        user_id: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> dict:
        """
        Get comprehensive query analytics.

        Args:
            user_id: User ID
            date_from: Start date for analytics
            date_to: End date for analytics

        Returns:
            {
                "success_rate": dict,
                "top_questions": list,
                "top_failures": list,
                "daily_stats": dict,
                "query_patterns": dict,
            }
        """
        return {
            "success_rate": self._analytics.calculate_success_rate(
                user_id=user_id,
                date_from=date_from,
                date_to=date_to,
            ),
            "top_questions": self._analytics.get_top_questions(user_id=user_id),
            "top_failures": self._analytics.get_top_failures(user_id=user_id),
            "daily_stats": self._analytics.get_daily_stats(user_id=user_id),
            "query_patterns": self._analytics.get_query_patterns(user_id=user_id),
        }
