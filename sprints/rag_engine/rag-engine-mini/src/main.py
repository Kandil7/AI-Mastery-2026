"""
FastAPI Application Entry Point
================================
Main application factory and ASGI entry point.

نقطة دخول التطبيق الرئيسية
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import setup_logging, get_logger
from src.api.v1.routes_health import router as health_router
from src.api.v1.routes_documents import router as documents_router
from src.api.v1.routes_queries import router as queries_router

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    
    إدارة دورة حياة التطبيق
    """
    # Startup
    log.info(
        "application_starting",
        app_name=settings.app_name,
        env=settings.env,
    )
    
    # Initialize resources here (e.g., database connections)
    # The container is lazily initialized on first request
    
    yield
    
    # Shutdown
    log.info("application_stopping")
    # Cleanup resources here


def create_app() -> FastAPI:
    """
    Application factory.
    
    Creates and configures the FastAPI application.
    
    مصنع التطبيق
    """
    # Setup logging
    setup_logging(
        level=settings.log_level,
        json_format=settings.env != "dev",
    )
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="Production-Ready RAG Starter Template",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(queries_router)
    
    return app


# Create app instance for ASGI server
app = create_app()


def main() -> None:
    """Entry point for running with uvicorn."""
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
