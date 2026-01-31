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
from src.api.v1.routes_chat import router as chat_router
from src.api.v1.routes_ask import router as ask_router
from src.api.v1.routes_documents_search import router as documents_search_router
from src.api.v1.routes_documents_bulk import router as documents_bulk_router
from src.api.v1.routes_auth import router as auth_router
from src.api.v1.routes_admin import router as admin_router
from src.api.v1.graphql import schema as graphql_schema
from strawberry.fastapi import GraphQLRouter
from src.application.services.event_manager import get_event_manager


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

    yield

    # Shutdown
    log.info("application_stopping")


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
    app.include_router(documents_search_router)
    app.include_router(documents_bulk_router)
    app.include_router(queries_router)
    app.include_router(chat_router)
    app.include_router(ask_router)
    app.include_router(auth_router)
    app.include_router(admin_router)

    # Include GraphQL with context injection
    async def get_graphql_context(request):
        from src.core.bootstrap import get_container

        container = get_container()
        return {
            "request": request,
            "doc_repo": container.get("document_repo"),
            "chat_repo": container.get("chat_repo"),
            "query_history_repo": container.get("query_history_repo"),
            "search_service": container.get("search_documents_use_case"),
            "ask_hybrid_use_case": container.get("ask_hybrid_use_case"),
            "event_manager": get_event_manager(),
            "db_repo": container.get("db_repo"),
            "redis_client": container.get("redis_client"),
            "vector_store": container.get("vector_store"),
            "llm": container.get("llm"),
            "file_storage": container.get("file_storage"),
        }

    graphql_app = GraphQLRouter(
        graphql_schema,
        context_getter=get_graphql_context,
    )
    app.mount("/graphql", graphql_app, name="graphql")

    # Setup observability
    from src.core.observability import setup_observability

    setup_observability(app)

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
