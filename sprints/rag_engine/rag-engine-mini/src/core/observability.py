"""
Observability and Metrics
=========================
Prometheus metrics setup for monitoring RAG performance.

إعدادات القياسات والمراقبة لـ Prometheus
"""

from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import Response
from functools import wraps
import time
from typing import Callable, Any

# Create a registry
registry = CollectorRegistry()

# -----------------------------------------------------------------------------
# API Metrics
# -----------------------------------------------------------------------------
# Request counter
API_REQUEST_COUNT = Counter(
    "rag_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

# Request latency
API_REQUEST_LATENCY = Histogram(
    "rag_api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry,
)

# -----------------------------------------------------------------------------
# RAG Core Metrics
# -----------------------------------------------------------------------------
# Retrieval scores
RETRIEVAL_SCORE = Histogram(
    "rag_retrieval_score",
    "Similarity scores of retrieved chunks",
    ["method"],  # vector, keyword, hybrid
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    registry=registry,
)

# Token usage
TOKEN_USAGE = Counter(
    "rag_llm_tokens_total",
    "Total tokens used by LLM",
    ["model", "type"],  # prompt, completion
    registry=registry,
)

# Cache hit ratio
EMBEDDING_CACHE_HIT = Counter(
    "rag_embedding_cache_total",
    "Embedding cache hits and misses",
    ["result"],  # hit, miss
    registry=registry,
)

# Reranking metrics
RERANK_DURATION = Histogram(
    "rag_rerank_duration_seconds",
    "Time spent reranking results",
    ["method"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry,
)

# Query expansion metrics
QUERY_EXPANSION_COUNT = Counter(
    "rag_query_expansion_total", "Total query expansions performed", ["expanded"], registry=registry
)

# -------------------------------------------------------------------------
# Celery Task Metrics
# -------------------------------------------------------------------------
CELERY_TASK_COUNT = Counter(
    "rag_celery_tasks_total",
    "Total number of Celery tasks executed",
    ["task", "status"],
    registry=registry,
)

CELERY_TASK_DURATION = Histogram(
    "rag_celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["task"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=registry,
)

# -----------------------------------------------------------------------------
# System Functions
# -----------------------------------------------------------------------------


def metrics_endpoint() -> Response:
    """
    FastAPI endpoint for Prometheus metrics.

    نقطة نهاية Prometheus للقياسات
    """
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def setup_observability(app) -> None:
    """
    Attach metrics routes to FastAPI.
    """

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return metrics_endpoint()


# -----------------------------------------------------------------------------
# Decorators for Use Cases
# -----------------------------------------------------------------------------


def track_request_latency(metric: Histogram, labels: dict[str, str] | None = None):
    """
    Decorator to track request latency.

    Usage:
        @track_request_latency(API_REQUEST_LATENCY, {"endpoint": "/ask"})
        def my_function():
            ...

    مصمم لتتبع وقت استجابة الطلب
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            metric_labels = labels or {}
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                metric.labels(**metric_labels).observe(time.time() - start_time)
                return result
            except Exception:
                metric.labels(**metric_labels).observe(time.time() - start_time)
                raise

        return wrapper

    return decorator


def track_token_usage(token_counter: Counter, model: str = "default"):
    """
    Decorator to track token usage for LLM calls.

    مصمم لتتبع استخدام الرموز لـ LLM
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Try to extract token counts from result
            if hasattr(result, "prompt_tokens"):
                token_counter.labels(model=model, type="prompt").inc(result.prompt_tokens)
            if hasattr(result, "completion_tokens"):
                token_counter.labels(model=model, type="completion").inc(result.completion_tokens)

            return result

        return wrapper

    return decorator


def track_cache_hit(cache_counter: Counter):
    """
    Decorator to track cache hits/misses.

    مصمم لتتبع إصابات/إخفاقات التخزين المؤقت
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if result was cached (implementation-specific)
            result = func(*args, **kwargs)

            # Example: if function has _was_cached attribute
            if hasattr(result, "_was_cached") and result._was_cached:
                cache_counter.labels(result="hit").inc()
            else:
                cache_counter.labels(result="miss").inc()

            return result

        return wrapper

    return decorator


def track_retrieval_score(score_metric: Histogram, method: str = "vector"):
    """
    Decorator to track retrieval scores.

    مصمم لتتبع درجات الاسترجاع
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Track scores for each retrieved chunk
            if hasattr(result, "__iter__"):
                for item in result:
                    if hasattr(item, "score"):
                        score_metric.labels(method=method).observe(item.score)

            return result

        return wrapper

    return decorator
