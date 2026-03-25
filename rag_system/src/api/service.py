"""
FastAPI Service for Arabic Islamic Literature RAG System

Following RAG Pipeline Guide 2026 - Production Deployment
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Data Models ====================


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="User question")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters"
    )
    stream: bool = Field(False, description="Stream the response")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    tokens_used: int
    model: str


class IndexRequest(BaseModel):
    """Request model for indexing endpoint."""

    limit: Optional[int] = Field(None, description="Limit number of books to index")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")


class IndexStatus(BaseModel):
    """Status of indexing operation."""

    status: str
    message: str
    progress: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    stats: Dict[str, Any]


class SourceFilter(BaseModel):
    """Filter for source metadata."""

    category: Optional[str] = None
    author: Optional[str] = None
    book_title: Optional[str] = None


# ==================== Application State ====================


class AppState:
    """Application state."""

    pipeline: Any = None
    indexing_task: Optional[asyncio.Task] = None
    indexing_status: Dict[str, Any] = {}

    def __init__(self):
        self._init_pipeline()

    def _init_pipeline(self):
        """Initialize the RAG pipeline."""
        try:
            from ..pipeline.complete_pipeline import create_rag_pipeline, RAGConfig

            config = RAGConfig(
                datasets_path=os.getenv(
                    "DATASETS_PATH",
                    "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets",
                ),
                output_path=os.getenv(
                    "OUTPUT_PATH",
                    "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data",
                ),
                embedding_model=os.getenv(
                    "EMBEDDING_MODEL",
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                ),
                embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
                llm_provider=os.getenv("LLM_PROVIDER", "mock"),
                llm_model=os.getenv("LLM_MODEL", "gpt-4o"),
            )

            self.pipeline = create_rag_pipeline(config)
            logger.info("RAG pipeline initialized")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.pipeline = None


# ==================== FastAPI App ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""

    # Startup
    logger.info("Starting Arabic Islamic Literature RAG API...")

    # Initialize state
    app.state = AppState()

    # Try to load existing indexes
    if app.state.pipeline:
        try:
            app.state.pipeline.load_indexes()
            logger.info("Loaded existing indexes")
        except Exception as e:
            logger.warning(f"Could not load existing indexes: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Arabic Islamic Literature RAG API...")


# Create FastAPI app
app = FastAPI(
    title="Arabic Islamic Literature RAG API",
    description="Retrieval-Augmented Generation API for Arabic Islamic literature",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Endpoints ====================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Arabic Islamic Literature RAG API",
        "version": "1.0.0",
        "description": "RAG system for Islamic literature corpus",
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""

    stats = {}
    if app.state.pipeline:
        try:
            stats = app.state.pipeline.get_stats()
        except Exception as e:
            logger.error(f"Error getting stats: {e}")

    return HealthResponse(
        status="healthy" if app.state.pipeline else "degraded",
        version="1.0.0",
        stats=stats,
    )


@app.get("/stats", tags=["Information"])
async def get_stats():
    """Get pipeline statistics."""

    if not app.state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        return app.state.pipeline.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories", tags=["Information"])
async def get_categories():
    """Get list of available categories."""

    if not app.state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        categories = app.state.pipeline.get_categories()
        return {"categories": sorted(categories)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", tags=["Indexing"], response_model=IndexStatus)
async def start_indexing(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start indexing documents.

    This endpoint starts the indexing process in the background.
    Use /index/status to check progress.
    """

    if not app.state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Check if already indexing
    if app.state.indexing_task and not app.state.indexing_task.done():
        return IndexStatus(
            status="in_progress",
            message="Indexing already in progress",
            progress=app.state.indexing_status,
        )

    # Start indexing in background
    async def index_documents():
        try:
            app.state.indexing_status = {"status": "started", "progress": 0}

            await app.state.pipeline.index_documents(
                limit=request.limit,
                categories=request.categories,
                progress_callback=lambda current, total: setattr(
                    app.state.indexing_status, "progress", int(current / total * 100)
                ),
            )

            app.state.indexing_status = {"status": "completed", "progress": 100}

        except Exception as e:
            logger.error(f"Indexing error: {e}")
            app.state.indexing_status = {"status": "error", "error": str(e)}

    app.state.indexing_task = asyncio.create_task(index_documents())

    return IndexStatus(
        status="started",
        message="Indexing started in background",
        progress={"status": "started"},
    )


@app.get("/index/status", tags=["Indexing"])
async def get_index_status():
    """Get indexing status."""

    if not app.state.indexing_task:
        return IndexStatus(
            status="not_started",
            message="No indexing task running",
        )

    if app.state.indexing_task.done():
        return IndexStatus(
            status="completed",
            message="Indexing completed",
            progress=app.state.indexing_status,
        )

    return IndexStatus(
        status="in_progress",
        message="Indexing in progress",
        progress=app.state.indexing_status,
    )


@app.post("/query", tags=["Query"], response_model=QueryResponse)
async def query(
    request: QueryRequest,
):
    """
    Query the RAG system.

    Provide a question and get an answer with sources.
    """

    if not app.state.pipeline:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Please index documents first.",
        )

    try:
        # Check if indexed
        if not app.state.pipeline._indexed:
            # Try to load existing index
            app.state.pipeline.load_indexes()

            if not app.state.pipeline._indexed:
                raise HTTPException(
                    status_code=503,
                    detail="No index available. Please call /index first.",
                )

        # Execute query
        result = await app.state.pipeline.query(
            question=request.query,
            top_k=request.top_k,
            filters=request.filters,
        )

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            sources=result.sources,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            model=result.model,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
):
    """
    Query the RAG system with streaming response.
    """

    if not app.state.pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Check if indexed
        if not app.state.pipeline._indexed:
            app.state.pipeline.load_indexes()

            if not app.state.pipeline._indexed:
                raise HTTPException(
                    status_code=503,
                    detail="No index available. Please call /index first.",
                )

        # Return streaming response
        return StreamingResponse(
            app.state.pipeline.query_stream(
                question=request.query,
                top_k=request.top_k,
            ),
            media_type="text/plain",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
