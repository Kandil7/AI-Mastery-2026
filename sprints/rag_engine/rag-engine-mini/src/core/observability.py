"""
Observability and Metrics
==========================
Prometheus metrics setup for monitoring RAG performance.

إعدادات القياسات والمراقبة لـ Prometheus
"""

from prometheus_client import Counter, Histogram, Summary, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

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
    registry=registry
)

# Request latency
API_REQUEST_LATENCY = Histogram(
    "rag_api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry
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
    registry=registry
)

# Token usage
TOKEN_USAGE = Counter(
    "rag_llm_tokens_total",
    "Total tokens used by LLM",
    ["model", "type"],  # prompt, completion
    registry=registry
)

# Cache hit ratio
EMBEDDING_CACHE_HIT = Counter(
    "rag_embedding_cache_total",
    "Embedding cache hits and misses",
    ["result"],  # hit, miss
    registry=registry
)

# -----------------------------------------------------------------------------
# System Functions
# -----------------------------------------------------------------------------

def metrics_endpoint() -> Response:
    """
    FastAPI endpoint for Prometheus metrics.
    
    نقطة نهاية Prometheus للقياسات
    """
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )

def setup_observability(app) -> None:
    """
    Attach metrics routes to FastAPI.
    """
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return metrics_endpoint()
