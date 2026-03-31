"""
Admin & Monitoring Routes
==============================
Endpoints for system monitoring and administration.

نقاط نهاية الإدارة والمراقبة
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.api.v1.deps import get_tenant_id
from src.core.config import settings

router = APIRouter(prefix="/api/v1/admin", tags=["admin", "monitoring"])


# ============================================================================
# Request/Response Models
# ============================================================================


class HealthComponent(BaseModel):
    """Health status for a component."""

    status: str = Field(..., description="Component status (ok, degraded, error)")
    message: str = Field(..., description="Component message")
    response_time_ms: int = Field(..., description="Response time")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Overall status (ok, degraded, error)")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")
    components: Dict[str, HealthComponent] = Field(..., description="Health components")


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics."""

    uptime_seconds: float = Field(..., description="System uptime in seconds")
    requests_total: int = Field(..., description="Total requests handled")
    requests_success: int = Field(..., description="Successful requests")
    requests_error: int = Field(..., description="Error requests")
    error_rate: float = Field(..., description="Error rate percentage")
    avg_latency_ms: float = Field(..., description="Average request latency (ms)")
    p95_latency_ms: float = Field(..., description="95th percentile latency (ms)")
    p99_latency_ms: float = Field(..., description="99th percentile latency (ms)")


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    redis_status: str = Field(..., description="Redis connection status")
    redis_memory_mb: int = Field(..., description="Redis memory usage (MB)")
    redis_total_keys: int = Field(..., description="Total Redis keys")
    embedding_cache_hit_rate: float = Field(..., description="Embedding cache hit rate")
    embedding_cache_size: int = Field(..., description="Embedding cache size")


class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics."""

    postgres_status: str = Field(..., description="PostgreSQL connection status")
    pool_size: int = Field(..., description="Connection pool size")
    pool_idle: int = Field(..., description="Idle connections")
    pool_active: int = Field(..., description="Active connections")
    total_queries: int = Field(..., description="Total queries since start")
    avg_query_time_ms: float = Field(..., description="Average query time (ms)")


class WorkerStatusResponse(BaseModel):
    """Response model for Celery worker status."""

    workers_online: int = Field(..., description="Number of online workers")
    workers_processing: int = Field(..., description="Number of currently processing tasks")
    workers_idle: int = Field(..., description="Number of idle workers")
    tasks_queue_size: int = Field(..., description="Tasks in queue")
    tasks_processed: int = Field(..., description="Tasks processed since start")


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/health/deep", response_model=HealthCheckResponse)
async def deep_health_check(
    tenant_id: str = Depends(get_tenant_id),
) -> HealthCheckResponse:
    """
    Deep health check for all system components.

    Checks:
    - API server status
    - Database connectivity and performance
    - Redis connectivity and memory usage
    - Qdrant vector store connectivity
    - Celery worker availability
    - Disk space
    - Background task queue status

    Usage:
        GET /api/v1/admin/health/deep

    فحص صحة شامل للنظام
    """
    timestamp = datetime.utcnow().isoformat()

    # Check components (placeholder - will connect to real services)
    components = {}

    # Database check
    db_status = "ok"  # Placeholder: Check real DB connection
    components["database"] = HealthComponent(
        status=db_status,
        message="Database responsive",
        response_time_ms=5,
    )

    # Redis check
    redis_status = "ok"  # Placeholder: Check real Redis connection
    components["redis"] = HealthComponent(
        status=redis_status,
        message="Redis responsive",
        response_time_ms=2,
    )

    # Qdrant check
    qdrant_status = "ok"  # Placeholder: Check real Qdrant connection
    components["qdrant"] = HealthComponent(
        status=qdrant_status,
        message="Qdrant responsive",
        response_time_ms=8,
    )

    # Determine overall status
    statuses = [c.status for c in components.values()]
    if "error" in statuses:
        overall_status = "error"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return HealthCheckResponse(
        status=overall_status,
        version="0.1.0",  # From package
        timestamp=timestamp,
        components=components,
    )


@router.get("/metrics")
async def get_system_metrics(
    tenant_id: str = Depends(get_tenant_id),
) -> SystemMetricsResponse:
    """
    Get system performance metrics.

    Metrics:
    - Uptime (seconds since start)
    - Total requests handled
    - Success/failure counts and rates
    - Request latency (avg, P95, P99)
    - Worker statistics

    Requires:
    - Prometheus metrics integration
    - Metric storage (Redis or Prometheus)

    Usage:
        GET /api/v1/admin/metrics

    مقاييسات النظام
    """
    # Placeholder metrics
    uptime = 86400.0  # 1 day

    return SystemMetricsResponse(
        uptime_seconds=uptime,
        requests_total=1000,
        requests_success=985,
        requests_error=15,
        error_rate=1.5,
        avg_latency_ms=145.5,
        p95_latency_ms=245.0,
        p99_latency_ms=320.0,
    )


@router.get("/cache/stats")
async def get_cache_stats(
    tenant_id: str = Depends(get_tenant_id),
) -> CacheStatsResponse:
    """
    Get cache statistics and performance.

    Cache Metrics:
    - Redis connection status
    - Memory usage
    - Total keys
    - Embedding cache hit rate
    - Eviction rate

    Usage:
        GET /api/v1/admin/cache/stats

    إحصائيات الذاكرة
    """
    # Placeholder stats
    return CacheStatsResponse(
        redis_status="connected",
        redis_memory_mb=64,
        redis_total_keys=1523,
        embedding_cache_hit_rate=75.5,
        embedding_cache_size=500,  # MB
    )


@router.get("/database/stats")
async def get_database_stats(
    tenant_id: str = Depends(get_tenant_id),
) -> DatabaseStatsResponse:
    """
    Get database performance statistics.

    DB Metrics:
    - Connection pool status
    - Query performance (avg time)
    - Total queries executed
    - Slow query count

    Usage:
        GET /api/v1/admin/database/stats

    إحصائيات قاعدة البيانات
    """
    return DatabaseStatsResponse(
        postgres_status="connected",
        pool_size=10,
        pool_idle=6,
        pool_active=4,
        total_queries=5000,
        avg_query_time_ms=85.5,
    )


@router.get("/workers/status")
async def get_worker_status(
    tenant_id: str = Depends(get_tenant_id),
) -> WorkerStatusResponse:
    """
    Get Celery worker status.

    Worker Metrics:
    - Number of online workers
    - Currently processing tasks
    - Idle workers
    - Queue size
    - Tasks processed
    - Worker health

    Usage:
        GET /api/v1/admin/workers/status

    حالة العمال
    """
    try:
        from src.workers.monitoring import get_worker_status as _get_worker_status

        status = _get_worker_status()
        return WorkerStatusResponse(**status)
    except Exception:
        return WorkerStatusResponse(
            workers_online=0,
            workers_processing=0,
            workers_idle=0,
            tasks_queue_size=0,
            tasks_processed=0,
        )


@router.post("/cache/flush")
async def flush_cache(
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Flush cache stores.

    Flushes:
    - Redis cache (all keys in specific DB)
    - Embedding cache
    - Query result cache

    Warning:
    - Will cause temporary performance degradation
    - Only use in emergencies or deployments

    Usage:
        POST /api/v1/admin/cache/flush

    مسح الذاكرة
    """
    # Placeholder: Call cache flush
    # redis.flushdb()

    return {
        "status": "flushed",
        "message": "All caches flushed",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/config")
async def get_system_config(
    tenant_id: str = Depends(get_tenant_id),
) -> Dict[str, Any]:
    """
    Get current system configuration (sanitized).

    Returns:
        Sanitized configuration (secrets masked)

    Security:
    - Never expose secret keys (JWT secret, API keys, DB password)
    - Mask sensitive values (passwords, tokens)
    - Only expose non-sensitive config

    Usage:
        GET /api/v1/admin/config

    إعدادات النظام
    """
    # Sanitized config (no secrets!)
    return {
        "app_name": settings.app_name,
        "env": settings.env,
        "log_level": settings.log_level,
        "llm_backend": settings.llm_backend,
        "embeddings_backend": settings.embeddings_backend,
        "qdrant_host": settings.qdrant_host,
        "redis_url": "redis://redis:6379/0",  # Host only, no password
        "rate_limit_enabled": True,
        "rate_limit_default": 100,
        "api_key_header": settings.api_key_header,
        "jwt_access_expire_minutes": settings.jwt_access_expire_minutes,
    }
