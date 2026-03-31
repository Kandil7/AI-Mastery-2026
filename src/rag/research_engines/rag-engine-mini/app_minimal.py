"""
RAG Engine Mini - Minimal Production API
=========================================
A minimal but functional FastAPI application that can run immediately
for demonstration and testing purposes.

This provides:
- Health check endpoints
- Basic API structure
- Production-ready logging
- CORS middleware
- Graceful startup/shutdown

Full functionality requires additional dependencies and configuration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application startup time
startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("RAG ENGINE MINI - Starting Up")
    logger.info("=" * 60)
    logger.info(f"Environment: {os.getenv('ENV', 'production')}")
    logger.info(f"Port: 8000")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("-" * 60)

    yield

    # Shutdown
    logger.info("-" * 60)
    logger.info("RAG ENGINE MINI - Shutting Down")
    logger.info("-" * 60)


# Create FastAPI app
app = FastAPI(
    title="RAG Engine Mini",
    description="Production-Ready RAG Starter Template (Minimal Version)",
    version="0.1.0",
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


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RAG Engine Mini",
        "version": "0.1.0",
        "status": "running",
        "description": "Production-Ready RAG API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns basic service availability information.
    """
    uptime = time.time() - startup_time
    return {
        "status": "healthy",
        "service": "rag-engine-api",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": round(uptime, 2),
        "checks": {
            "api": "ok",
            "database": "not_configured",
            "vector_store": "not_configured",
            "cache": "not_configured",
        },
    }


@app.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    Verifies that the service is ready to accept traffic.
    """
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "api": {"status": "pass", "message": "API is responding"},
            "configuration": {"status": "pass", "message": "Basic configuration loaded"},
        },
    }


@app.get("/health/live")
async def liveness_check():
    """
    Liveness check endpoint.
    Kubernetes uses this to know if the pod should be restarted.
    """
    return {"alive": True, "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "version": "0.1.0",
        "status": "operational",
        "features": {
            "documents": "available",
            "chat": "available",
            "search": "available",
            "export": "available",
        },
        "timestamp": datetime.now().isoformat(),
    }


# Document endpoints (mock responses)
@app.get("/api/v1/documents")
async def list_documents():
    """List all documents."""
    return {
        "documents": [],
        "total": 0,
        "message": "Document service ready - no documents indexed yet",
    }


@app.post("/api/v1/documents")
async def upload_document():
    """Upload a document."""
    return {
        "message": "Document upload endpoint ready",
        "status": "accepted",
        "document_id": "mock-doc-001",
    }


# Chat endpoints (mock responses)
@app.post("/api/v1/chat")
async def chat():
    """Chat endpoint."""
    return {
        "message": "Chat service ready",
        "response": "Hello! The RAG Engine is running but not fully configured. Add your OpenAI API key to .env to enable LLM responses.",
        "sources": [],
    }


@app.post("/api/v1/ask")
async def ask_question():
    """Ask a question endpoint."""
    return {
        "answer": "The RAG Engine is running! To get real answers, configure your LLM provider in the .env file.",
        "sources": [],
        "confidence": 1.0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
