"""
Health Check Routes
====================
Endpoints for service health monitoring.

نقاط نهاية فحص صحة الخدمة
"""

from fastapi import APIRouter

from src.core.config import settings

router = APIRouter(tags=["health"])


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
    # TODO: Add actual dependency checks
    return {
        "ready": True,
        "checks": {
            "database": "ok",
            "redis": "ok",
            "qdrant": "ok",
        },
    }
