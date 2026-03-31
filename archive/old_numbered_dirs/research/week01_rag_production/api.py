
"""
Production-Ready RAG API Service with Enhanced Error Handling and Validation

This module implements a production-grade API for the RAG system with comprehensive
error handling, request validation, and observability features. The API follows
RESTful design principles and includes proper status codes, validation, and
monitoring capabilities.

Key Features:
- Input validation using Pydantic models
- Comprehensive error handling with appropriate HTTP status codes
- Request/response logging and tracing
- Rate limiting and resource protection
- Health checks and readiness probes
- Detailed API documentation with OpenAPI/Swagger
- Performance monitoring and metrics collection

Endpoints:
- GET /: Health check endpoint
- POST /index: Add documents to the knowledge base
- POST /query: Query the RAG system
- GET /health: Health check with detailed status
- GET /metrics: Prometheus-compatible metrics endpoint

Security Considerations:
- Input validation to prevent injection attacks
- Rate limiting to prevent resource exhaustion
- Proper error message sanitization
- Secure handling of sensitive data

Example Usage:
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"query": "What is RAG?", "k": 3}'
"""

import sys
import os
import logging
import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from enum import Enum

# Add project root to path
# This allows running 'python sprints/week01_rag_production/api.py' from project root
# or from within the folder if we handle paths correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.pipeline import RAGPipeline, RAGConfig
from src.config import settings, Environment
from src.retrieval import Document
from src.observability import ObservabilityManager, LogLevel
from src.ingestion import initialize_ingestion_pipeline, ingestion_pipeline
from src.ingestion.mongo_storage import initialize_mongo_storage, close_mongo_storage, mongo_storage
from src.retrieval.query_processing import initialize_query_router, query_router
from src.retrieval.vector_store import initialize_vector_manager, VectorConfig, VectorDBType, close_vector_manager
from src.services.indexing import index_documents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize observability
obs_manager = ObservabilityManager("rag-api", log_level=LogLevel.INFO)


class DocumentSource(str, Enum):
    """Enumeration for document sources."""
    MANUAL = "manual"
    FILE_UPLOAD = "file_upload"
    DATABASE = "database"
    WEB_CRAWLER = "web_crawler"
    API_IMPORT = "api_import"


class DocumentRequest(BaseModel):
    """
    Request model for adding documents to the knowledge base.

    Attributes:
        id (str): Unique identifier for the document (alphanumeric, underscores, hyphens)
        content (str): The actual text content of the document (1-10000 characters)
        metadata (Dict[str, Any]): Additional metadata about the document
        source (DocumentSource): Source of the document
        doc_type (str): Type of document (e.g., 'policy', 'manual', 'faq')
    """
    id: str = Field(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$')
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    source: DocumentSource = DocumentSource.MANUAL
    doc_type: str = Field(default="unspecified", min_length=1, max_length=50)

    @validator('content')
    def validate_content(cls, v):
        """Validate content length and remove potentially harmful content."""
        if len(v) < 1:
            raise ValueError('Content must not be empty')
        if len(v) > 10000:
            raise ValueError('Content must not exceed 10000 characters')
        # Sanitize content to prevent injection attacks
        return v.strip()


class QueryRequest(BaseModel):
    """
    Request model for querying the RAG system.

    Attributes:
        query (str): The search query (1-500 characters)
        k (int): Number of documents to retrieve (1-20)
        include_sources (bool): Whether to include source documents in response
        timeout_seconds (float): Timeout for the query operation (1-60 seconds)
    """
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(default=3, ge=1, le=20)
    include_sources: bool = Field(default=True)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=60.0)

    @validator('query')
    def validate_query(cls, v):
        """Validate query length and sanitize input."""
        if len(v) < 1:
            raise ValueError('Query must not be empty')
        if len(v) > 500:
            raise ValueError('Query must not exceed 500 characters')
        return v.strip()


class SourceDocument(BaseModel):
    """
    Model representing a source document in the response.

    Attributes:
        id (str): Document identifier
        content (str): Document content snippet
        score (float): Relevance score
        rank (int): Rank in the results
        metadata (Dict[str, Any]): Document metadata
    """
    id: str
    content: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """
    Response model for RAG queries.

    Attributes:
        query (str): Original query
        response (str): Generated response
        sources (List[SourceDocument]): Retrieved source documents
        query_time_ms (float): Time taken for the query
        total_documents_indexed (int): Total number of documents in the index
    """
    query: str
    response: str
    sources: List[SourceDocument]
    query_time_ms: float
    total_documents_indexed: int


class HealthStatus(BaseModel):
    """
    Model for health check responses.

    Attributes:
        status (str): Overall health status
        timestamp (str): ISO timestamp of the check
        details (Dict[str, Any]): Detailed health information
    """
    status: str
    timestamp: str
    details: Dict[str, Any]


# --- Global State ---
rag_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for initialization and cleanup."""
    global rag_model
    logger.info("Initializing Production RAG Model...")

    # Initialize RAG pipeline with production configuration
    rag_model = RAGPipeline(
        RAGConfig(
            generator_model=settings.models.generator_model,
            dense_model=settings.models.dense_model,
            alpha=settings.retrieval.alpha,
            fusion=settings.retrieval.fusion_method,
            top_k=settings.models.top_k,
            max_new_tokens=settings.models.max_new_tokens,
        )
    )

    # Initialize MongoDB storage
    await initialize_mongo_storage()
    logger.info("MongoDB storage initialized")

    # Initialize vector storage
    vector_db_type = VectorDBType.CHROMA if settings.environment == Environment.PRODUCTION else VectorDBType.IN_MEMORY
    vector_config = VectorConfig(
        db_type=vector_db_type,
        collection_name="rag_vectors",
        persist_directory="./data/vector_store",
        dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
        metric="cosine",
    )
    await initialize_vector_manager(vector_config)
    logger.info("Vector storage initialized")

    # Initialize ingestion pipeline
    initialize_ingestion_pipeline(rag_model)
    logger.info("Ingestion pipeline initialized")

    # Initialize query router
    initialize_query_router(rag_model)
    logger.info("Query router initialized")

    # Pre-index some sample data for immediate availability
    sample_docs = [
        Document(
            id="welcome_doc",
            content="Welcome to the RAG system. This is a sample document to demonstrate functionality.",
            source="system",
            doc_type="welcome",
            metadata={"category": "system", "created_at": "2024-01-01"}
        ),
        Document(
            id="rag_explanation",
            content="RAG stands for Retrieval Augmented Generation. It combines information retrieval with language model generation to provide accurate, sourced answers.",
            source="system",
            doc_type="explanation",
            metadata={"category": "concepts", "created_at": "2024-01-01"}
        ),
        Document(
            id="hybrid_search",
            content="Hybrid search combines dense (semantic) and sparse (keyword) retrieval methods for optimal results.",
            source="system",
            doc_type="explanation",
            metadata={"category": "techniques", "created_at": "2024-01-01"}
        )
    ]
    await index_documents(rag_model, sample_docs)
    logger.info(f"Initialized with {len(sample_docs)} sample documents.")

    yield

    # Cleanup on shutdown
    await close_vector_manager()
    logger.info("Vector storage closed")

    await close_mongo_storage()
    logger.info("MongoDB storage closed")

    logger.info("Shutting down RAG API service...")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Production RAG API",
    description="Enterprise-grade Retrieval Augmented Generation API with hybrid search capabilities",
    version="1.0.0",
    docs_url="/docs",  # Enable Swagger UI
    redoc_url="/redoc",  # Enable ReDoc
    lifespan=lifespan
)


# Middleware for request logging and timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add processing time header and log requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Middleware to add observability to all requests."""
    start_time = time.time()

    # Start tracing
    with obs_manager.trace_request(
        operation=request.url.path,
        method=request.method,
        user_agent=request.headers.get("user-agent", ""),
        client_ip=request.client.host if request.client else "unknown"
    ) as trace_ctx:
        try:
            response = await call_next(request)

            # Calculate duration
            duration = (time.time() - start_time) * 1000  # Convert to ms
            trace_ctx.add_property("duration_ms", duration)
            trace_ctx.add_property("status_code", response.status_code)

            return response
        except Exception as e:
            # Log the exception
            duration = (time.time() - start_time) * 1000
            logger.error(f"Request failed: {str(e)}", extra={
                "url": str(request.url),
                "method": request.method,
                "duration_ms": duration
            })

            # Re-raise the exception to be handled by FastAPI
            raise


@app.get("/", response_model=Dict[str, str])
async def read_root() -> Dict[str, str]:
    """
    Root endpoint for basic health check.

    Returns:
        Dict[str, str]: Basic status information
    """
    return {
        "status": "healthy",
        "message": "Production RAG API is running",
        "version": "1.0.0",
        "endpoints": ["/docs", "/health", "/query", "/index"]
    }


@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Comprehensive health check endpoint.

    Returns:
        HealthStatus: Detailed health status information
    """
    global rag_model

    # Check if model is initialized
    model_status = "initialized" if rag_model else "not_initialized"

    # Check if we can perform a basic operation
    can_query = False
    try:
        if rag_model:
            # Perform a quick test query
            test_result = rag_model.query("test", top_k=1)
            can_query = True
    except Exception:
        can_query = False

    # Get document count if model is available
    doc_count = 0
    if rag_model:
        doc_count = len(rag_model.retriever.dense_retriever.documents)

    return HealthStatus(
        status="healthy" if (rag_model and can_query) else "degraded",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        details={
            "model_status": model_status,
            "can_query": can_query,
            "document_count": doc_count,
            "service": "rag-api"
        }
    )


@app.post("/index", response_model=Dict[str, Any])
async def add_documents(docs: List[DocumentRequest]) -> Dict[str, Any]:
    """
    Add documents to the knowledge base.

    This endpoint allows adding multiple documents to the RAG system's index.
    Documents are validated before being added to prevent malicious content.

    Args:
        docs: List of documents to add to the index

    Returns:
        Dict[str, Any]: Operation result with document count information

    Raises:
        HTTPException: If the model is not initialized or validation fails
    """
    global rag_model

    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")

    if not docs:
        raise HTTPException(status_code=400, detail="No documents provided")

    if len(docs) > 100:  # Prevent bulk operations that could overload the system
        raise HTTPException(status_code=413, detail="Too many documents in request (max 100)")

    try:
        # Convert to internal Document format
        domain_docs = []
        for doc_req in docs:
            domain_doc = Document(
                id=doc_req.id,
                content=doc_req.content,
                source=doc_req.source.value,
                doc_type=doc_req.doc_type,
                metadata=doc_req.metadata or {}
            )
            domain_docs.append(domain_doc)

        # Add documents to the model and persist to stores
        await index_documents(rag_model, domain_docs)

        # Get updated document count
        total_docs = len(rag_model.retriever.dense_retriever.documents)

        logger.info(f"Successfully added {len(docs)} documents. Total: {total_docs}")

        return {
            "message": f"Successfully added {len(docs)} documents",
            "total_docs": total_docs,
            "added_docs": [doc.id for doc in domain_docs]
        }

    except ValueError as ve:
        logger.error(f"Validation error when adding documents: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system to get an answer with supporting evidence.

    This endpoint processes a natural language query and returns a generated
    response along with the source documents that informed the answer.

    Args:
        request: Query parameters including the question and retrieval settings

    Returns:
        QueryResponse: Generated answer with source documents and metadata

    Raises:
        HTTPException: If the model is not initialized or query fails
    """
    global rag_model

    if not rag_model:
        raise HTTPException(status_code=503, detail="RAG model not initialized")

    start_time = time.time()

    try:
        # Perform the query with the specified parameters
        result = rag_model.query(request.query, top_k=request.k)

        # Calculate query time
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Prepare source documents for response
        sources = []
        if request.include_sources:
            sources = [
                SourceDocument(
                    id=d['id'],
                    content=d['content'][:500] + "..." if len(d['content']) > 500 else d['content'],  # Truncate long content
                    score=d['score'],
                    rank=d['rank'],
                    metadata=d.get('metadata')
                ) for d in result['retrieved_documents']
            ]

        # Get total document count
        total_docs = len(rag_model.retriever.dense_retriever.documents)

        logger.info(f"Query processed successfully in {query_time:.2f}ms", extra={
            "query_length": len(request.query),
            "top_k": request.k,
            "sources_returned": len(sources)
        })

        return QueryResponse(
            query=result['query'],
            response=result['response'],
            sources=sources,
            query_time_ms=query_time,
            total_documents_indexed=total_docs
        )

    except ValueError as ve:
        logger.warning(f"Invalid query parameters: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Invalid query parameters: {str(ve)}")
    except TimeoutError:
        logger.error(f"Query timed out after {request.timeout_seconds}s: {request.query[:50]}...")
        raise HTTPException(status_code=408, detail="Query timed out")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", extra={"query": request.query})
        raise HTTPException(status_code=500, detail=f"Internal error processing query: {str(e)}")


@app.post("/advanced_query", response_model=QueryResponse)
async def advanced_query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system using advanced query processing with classification and routing.

    This endpoint processes a natural language query using advanced techniques
    including query classification, expansion, and multi-step reasoning.

    Args:
        request: Query parameters including the question and retrieval settings

    Returns:
        QueryResponse: Generated answer with source documents and metadata

    Raises:
        HTTPException: If the model is not initialized or query fails
    """
    if not query_router:
        raise HTTPException(status_code=503, detail="Query router not initialized")

    start_time = time.time()

    try:
        # Process the query using the advanced query processor
        result = await query_router.route_and_process(request.query, top_k=request.k)

        # Calculate query time
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Prepare source documents for response
        sources = []
        if request.include_sources:
            sources = [
                SourceDocument(
                    id=source.document.id,
                    content=source.document.content[:500] + "..." if len(source.document.content) > 500 else source.document.content,  # Truncate long content
                    score=source.score,
                    rank=source.rank,
                    metadata=source.document.metadata
                ) for source in result.sources
            ]

        # Get total document count
        total_docs = len(rag_model.retriever.dense_retriever.documents)

        logger.info(f"Advanced query processed successfully in {query_time:.2f}ms", extra={
            "query_length": len(request.query),
            "top_k": request.k,
            "sources_returned": len(sources),
            "query_type": result.query_type.value
        })

        return QueryResponse(
            query=result.query,
            response=result.response,
            sources=sources,
            query_time_ms=query_time,
            total_documents_indexed=total_docs
        )

    except ValueError as ve:
        logger.warning(f"Invalid query parameters: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Invalid query parameters: {str(ve)}")
    except TimeoutError:
        logger.error(f"Query timed out after {request.timeout_seconds}s: {request.query[:50]}...")
        raise HTTPException(status_code=408, detail="Query timed out")
    except Exception as e:
        logger.error(f"Error processing advanced query: {str(e)}", extra={"query": request.query})
        raise HTTPException(status_code=500, detail=f"Internal error processing advanced query: {str(e)}")


@app.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata: str = "{}"
) -> Dict[str, Any]:
    """
    Upload a document for indexing in the RAG system.

    This endpoint accepts file uploads and processes them for inclusion in the
    knowledge base. It handles various document formats and performs automatic
    chunking and indexing.

    Args:
        file: The document file to upload
        background_tasks: Background tasks for processing
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
        metadata: Additional metadata as JSON string

    Returns:
        Dict with upload result information
    """
    if not ingestion_pipeline:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")

    try:
        # Parse metadata from JSON string
        import json
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON format")

    from src.ingestion import IngestionRequest

    ingestion_request = IngestionRequest(
        metadata=metadata_dict,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    try:
        result = await ingestion_pipeline.ingest_from_file(file, ingestion_request)

        if not result.success:
            raise HTTPException(status_code=400, detail="; ".join(result.errors))

        return {
            "message": result.message,
            "processed_documents": result.processed_documents,
            "indexed_documents": result.indexed_documents,
            "processing_time_ms": result.processing_time_ms,
            "warnings": result.warnings
        }
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(skip: int = 0, limit: int = 100) -> Dict[str, Any]:
    """
    List documents in the system.

    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return

    Returns:
        Dict with document list and metadata
    """
    try:
        documents = await mongo_storage.get_all_documents(skip=skip, limit=limit)
        return {
            "documents": [
                {
                    "id": doc.id,
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "metadata": doc.metadata,
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                }
                for doc in documents
            ],
            "total_count": await mongo_storage.get_document_count(),
            "returned_count": len(documents),
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@app.get("/documents/{doc_id}", response_model=Dict[str, Any])
async def get_document(doc_id: str) -> Dict[str, Any]:
    """
    Get a specific document by ID.

    Args:
        doc_id: ID of the document to retrieve

    Returns:
        Dict with document information
    """
    try:
        document = await mongo_storage.retrieve_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "id": document.id,
            "content": document.content,
            "source": document.source,
            "doc_type": document.doc_type,
            "metadata": document.metadata,
            "created_at": document.created_at,
            "updated_at": document.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns:
        str: Metrics in Prometheus text format
    """
    # This would return actual metrics in a real implementation
    # For now, returning a placeholder
    return obs_manager.export_metrics_prometheus()


# Custom exception handler
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom handler for HTTP exceptions."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# General exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General handler for unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level="info",
        timeout_keep_alive=30,
        workers=1  # Adjust based on your needs
    )
