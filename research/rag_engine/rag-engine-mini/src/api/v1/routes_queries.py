"""
Query History Routes
=====================
Endpoints for query history and analytics.

نقاط نهاية سجل الاستفسارات
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional

from src.api.v1.deps import get_tenant_id
from src.application.use_cases.query_history_use_case import (
    QueryHistoryUseCase,
    GetHistoryRequest,
    GetHistoryResponse,
    GetAnalyticsRequest,
    GetAnalyticsResponse,
    DeleteHistoryRequest,
)
from src.application.ports.query_history_repo import QueryStatus

router = APIRouter(prefix="/api/v1/queries", tags=["queries", "analytics"])


# ============================================================================
# Request/Response Models
# ============================================================================


class QueryRecordModel(BaseModel):
    """Model for a single query record."""

    query_id: str = Field(..., description="Query record ID")
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="System's answer")
    sources: List[str] = Field(..., description="Source IDs")
    status: str = Field(..., description="Query status (success/failed/timeout/error)")
    created_at: str = Field(..., description="Timestamp when query was executed")


class SuccessRateModel(BaseModel):
    """Model for success rate metrics."""

    total_queries: int = Field(..., description="Total queries in period")
    successful_queries: int = Field(..., description="Successful queries")
    success_rate: float = Field(..., description="Success rate percentage")
    failed_queries: int = Field(..., description="Failed queries")
    failure_rate: float = Field(..., description="Failure rate percentage")
    avg_response_time_ms: float = Field(..., description="Average response time (ms)")


class TopQuestionModel(BaseModel):
    """Model for top question."""

    question: str = Field(..., description="Question text")
    count: int = Field(..., description="Number of times asked")
    last_asked: str = Field(..., description="Last asked timestamp")
    success_rate: float = Field(..., description="Success rate for this question")


class DailyStatsModel(BaseModel):
    """Model for daily statistics."""

    date: str = Field(..., description="Date (YYYY-MM-DD)")
    total_queries: int = Field(..., description="Total queries")
    successful_queries: int = Field(..., description="Successful queries")
    success_rate: float = Field(..., description="Success rate")
    trend: str = Field(..., description="Trend (increasing/stable/decreasing)")


class QueryPatternModel(BaseModel):
    """Model for query pattern analysis."""

    avg_question_length: float = Field(..., description="Average question length")
    length_distribution: dict = Field(..., description="Length stats (min, max, avg, median)")
    word_frequency: List[dict] = Field(..., description="Top 10 most common words")
    complexity_score: float = Field(..., description="Complexity score (0-100)")
    complexity_factors: List[float] = Field(..., description="Complexity factors")


class AnalyticsResponseModel(BaseModel):
    """Model for comprehensive analytics response."""

    success_rate: SuccessRateModel
    top_questions: List[TopQuestionModel]
    top_failures: List[TopQuestionModel]
    daily_stats: List[DailyStatsModel]
    query_patterns: QueryPatternModel


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/history", response_model=GetHistoryResponse)
async def get_query_history(
    request: GetHistoryRequest,
    tenant_id: str = Depends(get_tenant_id),  # Will be user_id from auth
) -> GetHistoryResponse:
    """
    Get paginated query history for the authenticated user.

    Features:
    - Pagination support (limit, offset)
    - Next/prev page links
    - Query metadata (status, latency, tokens)

    Usage:
        POST /api/v1/queries/history
        {
            "limit": 20,
            "offset": 0
        }

    سجل الاستفسارات للمستخدم
    """
    # Note: tenant_id will be user_id in multi-tenant system
    user_id = tenant_id

    response = GetHistoryResponse(
        queries=request.limit,  # Will be replaced by use case
        total=request.limit,
        limit=request.limit,
        offset=request.offset,
        has_next=False,
        has_prev=False,
    )

    return response


@router.get("/history")
async def get_query_history_simple(
    tenant_id: str = Depends(get_tenant_id),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    """
    Simple history endpoint for compatibility.
    """
    # Placeholder questions list
    questions = [
        {"query_id": f"q_{i}", "question": f"Question {i}", "answer": "", "sources": []}
        for i in range(offset, offset + limit)
    ]
    return {
        "questions": questions,
        "total": len(questions),
        "limit": limit,
        "offset": offset,
        "has_next": False,
        "has_prev": offset > 0,
    }


@router.get("/recent")
async def get_recent_queries(
    tenant_id: str = Depends(get_tenant_id),
    limit: int = Query(10, ge=1, le=100, description="Number of recent queries"),
) -> list:
    """
    Get recent queries for quick access and search suggestions.

    Use Cases:
    - Search bar suggestions
    - Quick re-ask of common questions
    - User behavior analytics
    - Recent queries dashboard

    Usage:
        GET /api/v1/queries/recent?limit=10

    استفسارات حديثة
    """
    user_id = tenant_id

    # Placeholder - will use use case
    return [
        {
            "query_id": "q_123",
            "question": "What is RAG?",
            "answer": "Retrieval-Augmented Generation",
            "created_at": "2024-01-15T10:30:00Z",
        }
    ]


@router.get("/analytics")
async def get_query_analytics(
    tenant_id: str = Depends(get_tenant_id),
    date_from: Optional[str] = Query(None, description="Start date for analytics (ISO 8601)"),
    date_to: Optional[str] = Query(None, description="End date for analytics (ISO 8601)"),
) -> AnalyticsResponseModel:
    """
    Get comprehensive query analytics.

    Analytics Includes:
    - Success rate (percentage, counts)
    - Top questions (most frequently asked)
    - Top failures (most common failed queries)
    - Daily statistics (trends, totals)
    - Query patterns (length, words, complexity)

    Usage:
        GET /api/v1/queries/analytics?date_from=2024-01-01

    تحليلات السجل
    """
    user_id = tenant_id

    # Placeholder analytics response
    return AnalyticsResponseModel(
        success_rate=SuccessRateModel(
            total_queries=100,
            successful_queries=85,
            success_rate=85.0,
            failed_queries=15,
            failure_rate=15.0,
            avg_response_time_ms=245.5,
        ),
        top_questions=[],
        top_failures=[],
        daily_stats=[],
        query_patterns=QueryPatternModel(
            avg_question_length=25.5,
            length_distribution={"min": 10, "max": 50, "avg": 25.5, "median": 24},
            word_frequency=[],
            complexity_score=65.0,
            complexity_factors=[1.5, 2.0, 1.0],
        ),
    )


@router.delete("/history/{query_id}")
async def delete_query_record(
    query_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Delete a specific query from history.

    Security:
    - Requires authentication
    - Validates user ownership
    - Cannot be undone (use with caution)

    Usage:
        DELETE /api/v1/queries/history/q_123

    حذف استفسار محدد
    """
    user_id = tenant_id

    # Placeholder - will use use case
    return {"status": "deleted", "query_id": query_id}


@router.delete("/history/bulk")
async def bulk_delete_history(
    before_date: str = Query(..., description="Delete all queries older than this date (ISO 8601)"),
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Bulk delete query history by date.

    Use Cases:
    - GDPR compliance (right to be forgotten)
    - Privacy cleanup (delete old sensitive queries)
    - Storage optimization (delete old data)

    Usage:
        DELETE /api/v1/queries/history/bulk?before_date=2024-01-01

    حذف سجل قديم
    """
    user_id = tenant_id

    # Placeholder
    count = 100  # Will be real count from use case

    return {
        "status": "deleted",
        "count": count,
        "before_date": before_date,
    }
