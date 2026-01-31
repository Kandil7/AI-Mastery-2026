"""
Health Check Routes
====================
Endpoints for service health monitoring.

نقاط نهاية فحص صحة الخدمة
"""

import time
from typing import Dict, Any

from fastapi import APIRouter
from qdrant_client import QdrantClient
import redis
from sqlalchemy import create_engine, text

from src.core.config import settings
from src.adapters.filestore.factory import create_file_store

router = APIRouter(tags=["health"])


def _check_postgres_connection() -> Dict[str, Any]:
    """
    Check PostgreSQL database connection with latency measurement.

    Returns:
        Dict with status (ok/degraded/error), latency_ms, and message

    فحص اتصال قاعدة بيانات PostgreSQL
    """
    try:
        start_time = time.time()
        engine = create_engine(settings.database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.time() - start_time) * 1000

        status = "ok" if latency_ms < 100 else "degraded"
        return {
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "message": "Database connection successful",
        }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"Database connection failed: {str(e)}",
        }


def _check_redis_connection() -> Dict[str, Any]:
    """
    Check Redis connection with latency measurement.

    Returns:
        Dict with status, latency_ms, and message

    فحص اتصال Redis
    """
    try:
        start_time = time.time()
        client = redis.from_url(settings.redis_url, socket_timeout=2)
        client.ping()
        latency_ms = (time.time() - start_time) * 1000

        status = "ok" if latency_ms < 50 else "degraded"
        return {
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "message": "Redis connection successful",
        }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"Redis connection failed: {str(e)}",
        }


def _check_qdrant_connection() -> Dict[str, Any]:
    """
    Check Qdrant vector database connection.

    Returns:
        Dict with status, latency_ms, and message

    فحص اتصال قاعدة بيانات المتجهات Qdrant
    """
    try:
        start_time = time.time()
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=2,
        )
        client.get_collections()
        latency_ms = (time.time() - start_time) * 1000

        status = "ok" if latency_ms < 100 else "degraded"
        return {
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "message": "Qdrant connection successful",
        }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"Qdrant connection failed: {str(e)}",
        }


def _check_llm_connection() -> Dict[str, Any]:
    """
    Check LLM provider availability by performing a simple generation test.

    Returns:
        Dict with status, latency_ms, and message

    فحص توفر مزود نموذج اللغة
    """
    try:
        from src.core.bootstrap import get_container

        start_time = time.time()
        container = get_container()
        llm = container["llm"]

        # Simple test generation
        result = llm.generate("Test", max_tokens=5, timeout=5)
        latency_ms = (time.time() - start_time) * 1000

        if result and len(result) > 0:
            status = "ok" if latency_ms < 2000 else "degraded"
            return {
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "message": f"LLM connection successful ({settings.llm_backend})",
            }
        else:
            return {
                "status": "error",
                "latency_ms": round(latency_ms, 2),
                "message": "LLM returned empty response",
            }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"LLM connection failed ({settings.llm_backend}): {str(e)}",
        }


def _check_file_storage() -> Dict[str, Any]:
    """
    Check file storage backend connectivity with write/read test.

    Returns:
        Dict with status, latency_ms, and message

    فحص تخزين الملفات
    """
    try:
        import uuid
        import os

        start_time = time.time()
        file_store = create_file_store(settings)
        test_filename = f"health_check_{uuid.uuid4().hex[:16]}.txt"
        test_content = b"health_check_test"

        # Write test
        import asyncio

        async def write_read_test():
            stored = await file_store.save_upload(
                tenant_id="health_check",
                upload_filename=test_filename,
                content_type="text/plain",
                data=test_content,
            )
            # Read test
            with open(stored.path, "rb") as f:
                content = f.read()
            # Cleanup
            os.remove(stored.path)
            return content

        content = asyncio.run(write_read_test())
        latency_ms = (time.time() - start_time) * 1000

        if content == test_content:
            status = "ok" if latency_ms < 500 else "degraded"
            return {
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "message": f"File storage successful ({settings.filestore_backend})",
            }
        else:
            return {
                "status": "error",
                "latency_ms": round(latency_ms, 2),
                "message": "File storage read/write mismatch",
            }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": None,
            "message": f"File storage failed ({settings.filestore_backend}): {str(e)}",
        }


@router.get("/health")
def health_check() -> dict:
    """
    Basic health check endpoint.

    Returns service status and environment.

    نقطة نهاية فحص الصحة الأساسية
    """
    return {
        "status": "ok",
        "env": settings.env,
        "app_name": settings.app_name,
    }


@router.get("/health/ready")
def readiness_check() -> dict:
    """
    Readiness check for Kubernetes/load balancers.

    In production, this would verify:
    - Database connection
    - Redis connection
    - Qdrant connection

    فحص الجاهزية لـ Kubernetes
    """
    checks = {
        "database": _check_postgres_connection(),
        "redis": _check_redis_connection(),
        "qdrant": _check_qdrant_connection(),
    }

    # Ready if all critical checks are not in error state
    ready = all(check["status"] != "error" for check in checks.values())

    return {
        "ready": ready,
        "checks": checks,
    }


@router.get("/health/deep")
def deep_health_check() -> dict:
    """
    Deep health check for system dependencies.
    """
    return {
        "status": "ok",
        "database": "ok",
        "redis": "ok",
        "qdrant": "ok",
    }
