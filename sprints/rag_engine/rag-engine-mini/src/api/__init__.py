"""API layer package - FastAPI routes and dependencies."""

from src.api.v1.routes_health import router as health_router
from src.api.v1.routes_documents import router as documents_router
from src.api.v1.routes_queries import router as queries_router

__all__ = [
    "health_router",
    "documents_router",
    "queries_router",
]
