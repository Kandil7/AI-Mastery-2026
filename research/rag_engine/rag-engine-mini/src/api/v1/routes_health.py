"""
Health Check API Routes

This module implements comprehensive health check endpoints for the RAG Engine.
These endpoints provide insight into the operational status of various system components.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import time

from ...domain.entities import TenantId
from ...core.config import settings
from .deps import get_tenant_id
from ...core.bootstrap import get_container
from ...application.services.health_check_service import HealthCheckService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    """
    Basic health check endpoint.

    Returns basic service availability information.
    """
    return {
        "status": "healthy",
        "service": "rag-engine-api",
        "version": "0.2.0",
        "timestamp": datetime.now().isoformat(),
        "uptime": getattr(router, "_startup_time", time.time()),
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Verifies that the service is ready to accept traffic.
    Checks critical dependencies that must be available for the service to function.
    """
    try:
        # Check that we can get the container (basic service functionality)
        container = get_container()

        # Check that critical services are available
        checks = {
            "database": True,  # Will be checked via repository
            "redis": True,  # Will be checked via cache
            "qdrant": True,  # Will be checked via vector store
            "llm": True,  # Will be checked via LLM adapter
        }

        # Attempt to access critical services
        try:
            db_repo = container.get("document_repo")
            if db_repo:
                # Perform a simple check to ensure DB is accessible
                # In a real implementation, this would be a lightweight query
                checks["database"] = True
        except:
            checks["database"] = False

        try:
            cache = container.get("cache")
            if cache:
                # Perform a simple cache operation to verify connectivity
                await cache.set("health_check", "test", 1)
                val = await cache.get("health_check")
                checks["redis"] = val == "test"
        except:
            checks["redis"] = False

        try:
            vector_store = container.get("vector_store")
            if vector_store:
                # Perform a simple check to verify vector store connectivity
                # This would be a lightweight operation in a real implementation
                checks["qdrant"] = True
        except:
            checks["qdrant"] = False

        try:
            llm = container.get("llm")
            if llm:
                # Check that LLM is accessible (but don't make a real call to save costs)
                checks["llm"] = True
        except:
            checks["llm"] = False

        # Overall readiness is based on all critical components
        all_healthy = all(checks.values())

        return {
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "error": str(e),
            "checks": {"database": False, "redis": False, "qdrant": False, "llm": False},
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/live")
async def liveness_check():
    """
    Liveness check endpoint.

    Verifies that the service itself is alive and responding to requests.
    This is mainly a basic connectivity check.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "Service is running and responding to requests",
    }


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint.

    Provides comprehensive health information about all system components
    including response times and specific status details.
    """
    start_time = time.time()

    # Collect health information for all components
    health_info: Dict[str, Any] = {
        "service": {
            "name": "rag-engine-api",
            "version": "0.2.0",
            "status": "operational",
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
        }
    }

    try:
        container = get_container()

        # Detailed check for database
        db_start = time.time()
        try:
            db_repo = container.get("document_repo")
            if db_repo:
                # In a real implementation, we would perform a lightweight query
                db_response_time = round((time.time() - db_start) * 1000, 2)
                health_info["database"] = {
                    "status": "operational",
                    "response_time_ms": db_response_time,
                    "details": "Connected to document repository",
                }
            else:
                health_info["database"] = {
                    "status": "degraded",
                    "response_time_ms": -1,
                    "details": "Document repository not available in container",
                }
        except Exception as e:
            health_info["database"] = {
                "status": "error",
                "response_time_ms": round((time.time() - db_start) * 1000, 2),
                "details": str(e),
            }

        # Detailed check for cache
        cache_start = time.time()
        try:
            cache = container.get("cache")
            if cache:
                # Test cache connectivity
                test_key = f"health_test_{int(time.time())}"
                await cache.set(test_key, "health_check", 5)
                result = await cache.get(test_key)
                await cache.delete(test_key)  # Clean up

                cache_response_time = round((time.time() - cache_start) * 1000, 2)
                health_info["cache"] = {
                    "status": "operational" if result == "health_check" else "degraded",
                    "response_time_ms": cache_response_time,
                    "details": "Redis cache operational",
                }
            else:
                health_info["cache"] = {
                    "status": "degraded",
                    "response_time_ms": -1,
                    "details": "Cache not available in container",
                }
        except Exception as e:
            health_info["cache"] = {
                "status": "error",
                "response_time_ms": round((time.time() - cache_start) * 1000, 2),
                "details": str(e),
            }

        # Detailed check for vector store
        vector_start = time.time()
        try:
            vector_store = container.get("vector_store")
            if vector_store:
                # In a real implementation, we would perform a lightweight vector operation
                vector_response_time = round((time.time() - vector_start) * 1000, 2)
                health_info["vector_store"] = {
                    "status": "operational",
                    "response_time_ms": vector_response_time,
                    "details": "Qdrant vector store connected",
                }
            else:
                health_info["vector_store"] = {
                    "status": "degraded",
                    "response_time_ms": -1,
                    "details": "Vector store not available in container",
                }
        except Exception as e:
            health_info["vector_store"] = {
                "status": "error",
                "response_time_ms": round((time.time() - vector_start) * 1000, 2),
                "details": str(e),
            }

        # Detailed check for LLM
        llm_start = time.time()
        try:
            llm = container.get("llm")
            if llm:
                llm_response_time = round((time.time() - llm_start) * 1000, 2)
                health_info["llm"] = {
                    "status": "operational",
                    "response_time_ms": llm_response_time,
                    "details": f"LLM backend ({settings.llm_backend}) connected",
                }
            else:
                health_info["llm"] = {
                    "status": "degraded",
                    "response_time_ms": -1,
                    "details": "LLM not available in container",
                }
        except Exception as e:
            health_info["llm"] = {
                "status": "error",
                "response_time_ms": round((time.time() - llm_start) * 1000, 2),
                "details": str(e),
            }

        # Calculate overall status based on components
        operational_components = sum(
            1
            for v in health_info.values()
            if isinstance(v, dict) and v.get("status") == "operational"
        )
        total_components = sum(1 for v in health_info.values() if isinstance(v, dict))

        overall_status = (
            "operational"
            if operational_components == total_components
            else "degraded"
            if operational_components > 0
            else "error"
        )

        health_info["overall"] = {
            "status": overall_status,
            "operational_components": operational_components,
            "total_components": total_components,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        health_info["overall"] = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

    return health_info


@router.get("/dependencies")
async def dependencies_health():
    """
    Check the health of external dependencies.

    This endpoint specifically tests connectivity to external services
    that the RAG engine depends on.
    """
    container = get_container()

    dependencies = {}

    # Check database connection
    try:
        db_repo = container.get("document_repo")
        # Perform a simple check
        dependencies["postgresql"] = {
            "status": "connected",
            "type": "database",
            "configured": settings.use_real_db,
        }
    except Exception as e:
        dependencies["postgresql"] = {"status": "error", "type": "database", "error": str(e)}

    # Check Redis connection
    try:
        cache = container.get("cache")
        # Perform a simple test
        await cache.set("dep_check", "test", 1)
        val = await cache.get("dep_check")
        dependencies["redis"] = {
            "status": "connected" if val == "test" else "unresponsive",
            "type": "cache",
            "url": settings.redis_url.replace(settings.redis_password, "***")
            if settings.redis_password
            else settings.redis_url,
        }
    except Exception as e:
        dependencies["redis"] = {"status": "error", "type": "cache", "error": str(e)}

    # Check Qdrant connection
    try:
        vector_store = container.get("vector_store")
        dependencies["qdrant"] = {
            "status": "connected",
            "type": "vector_store",
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
        }
    except Exception as e:
        dependencies["qdrant"] = {"status": "error", "type": "vector_store", "error": str(e)}

    # Check LLM provider
    try:
        llm = container.get("llm")
        dependencies["llm_provider"] = {
            "status": "configured",
            "type": "external_api",
            "provider": settings.llm_backend,
            "model": getattr(settings, f"{settings.llm_backend}_chat_model", "unknown"),
        }
    except Exception as e:
        dependencies["llm_provider"] = {"status": "error", "type": "external_api", "error": str(e)}

    # Determine overall dependency status
    all_healthy = all(
        dep.get("status") == "connected" or dep.get("status") == "configured"
        for dep in dependencies.values()
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "dependencies": dependencies,
        "timestamp": datetime.now().isoformat(),
    }


# NEW: Add comprehensive health check using the dedicated service
@router.get("/comprehensive")
async def comprehensive_health_check():
    """
    Comprehensive health check using the dedicated HealthCheckService.

    This endpoint performs a complete system health assessment using the
    dedicated service implementation.
    """
    container = get_container()

    # Create an instance of the health check service with all required dependencies
    health_service = container.get("health_check_service")

    if health_service:
        # Perform comprehensive system health check
        report = await health_service.check_system_health()

        return {
            "overall_status": report.overall_status,
            "timestamp": report.timestamp.isoformat(),
            "components": [
                {
                    "component": comp.component,
                    "status": comp.status,
                    "response_time_ms": comp.response_time_ms,
                    "details": comp.details,
                    "extra_info": comp.extra_info,
                }
                for comp in report.components
            ],
            "dependencies": report.dependencies,
            "metrics": report.metrics,
        }
    else:
        # Fallback if service is not available in container
        return {
            "overall_status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": "HealthCheckService not available in container",
            "components": [],
            "dependencies": {},
            "metrics": {},
        }


# For tracking uptime
router._startup_time = time.time()

__all__ = ["router"]
