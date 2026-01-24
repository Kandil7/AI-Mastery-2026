from __future__ import annotations

from fastapi import FastAPI

from app.routers import eval as eval_router
from app.routers import feedback, health, ingest, query
from core.settings import load_settings


def create_app() -> FastAPI:
    settings = load_settings()
    app = FastAPI(title="Week5 RAG Backend", version=settings.app_version)

    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(eval_router.router)
    app.include_router(feedback.router)

    return app


app = create_app()
