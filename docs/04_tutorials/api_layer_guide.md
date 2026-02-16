# RAG Engine Mini - API Layer Deep Dive

## Introduction

The API layer in RAG Engine Mini provides the external interface to the application, exposing the business capabilities through RESTful endpoints and GraphQL. Built with FastAPI, this layer handles HTTP requests, authentication, validation, and response formatting. It acts as the boundary between the external world and the internal business logic.

## Architecture Overview

The API layer is organized into versioned routes following REST conventions:

- `/api/v1/documents/*` - Document management endpoints
- `/api/v1/queries/*` - Query history and analytics
- `/api/v1/chat/*` - Conversational interfaces
- `/api/v1/ask/*` - Direct question answering
- `/api/v1/auth/*` - Authentication and authorization
- `/api/v1/admin/*` - Administrative functions

## Core Components

### FastAPI Application Factory

The main application is created using a factory pattern:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events.
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
    
    # Include routers
    app.include_router(health_router)
    app.include_router(documents_router)
    # ... other routers
```

### Dependency Injection

The API layer accesses application services through the DI container:

```python
def ask_question(
    request: AskRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> AskResponse:
    container = get_container()
    use_case: AskQuestionHybridUseCase | None = container.get("ask_hybrid_use_case")
    if not use_case:
        raise HTTPException(status_code=501, detail="Ask use case not configured")
    
    # Execute the use case
    answer = use_case.execute(ask_request)
    return AskResponse(answer=answer.text, sources=list(answer.sources))
```

## Authentication and Authorization

### Tenant Isolation

All endpoints implement tenant-based isolation:

```python
def get_tenant_id(api_key: str = Security(get_api_key)) -> str:
    """
    Extract tenant ID from API key.
    
    In a real implementation, this would validate the API key
    and return the associated tenant ID.
    """
    # In this example, we're using the API key as tenant ID
    # In production, this would map API keys to tenant IDs
    return api_key
```

### API Key Authentication

The system uses API key-based authentication:

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

def get_api_key(security_scheme: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Extract and validate API key from request headers.
    """
    api_key = security_scheme.credentials
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Validate API key (implementation depends on your auth system)
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key
```

## Document Management Endpoints

### Upload Endpoint

Handles document uploads with validation:

```python
@router.post("/upload", response_model=UploadResultModel)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
) -> UploadResultModel:
    """
    Upload a document for processing.
    
    The document is stored and queued for background indexing.
    Supported formats: PDF, DOCX, TXT
    """
    # Validate file type
    if file.content_type not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {settings.allowed_extensions}"
        )
    
    # Validate file size
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_upload_mb}MB"
        )
    
    # Process upload through use case
    container = get_container()
    upload_use_case: UploadDocumentUseCase = container["upload_use_case"]
    
    result = upload_use_case.execute(
        tenant_id=tenant_id,
        file_data=content,
        filename=file.filename,
        content_type=file.content_type
    )
    
    return UploadResultModel(
        document_id=result.document_id.value,
        status=result.status,
        message=result.message
    )
```

### Status Checking

Monitors document processing status:

```python
@router.get("/{document_id}/status", response_model=DocumentStatusModel)
async def get_document_status(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> DocumentStatusModel:
    """
    Get the processing status of a document.
    """
    container = get_container()
    document_repo = container["document_repo"]
    
    status = document_repo.get_status(
        tenant_id=TenantId(tenant_id),
        document_id=DocumentId(document_id)
    )
    
    if not status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentStatusModel(
        document_id=status.document_id.value,
        filename=status.filename,
        status=status.status,
        error=status.error,
        chunks_count=status.chunks_count,
        created_at=status.created_at,
        updated_at=status.updated_at
    )
```

## Query and Answering Endpoints

### Hybrid RAG Endpoint

The core question-answering endpoint:

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    document_id: str | None = None
    k: int | None = Field(default=None, ge=1, le=200, description="Top-K override")
    k_vec: int | None = Field(default=None, ge=1, le=200)
    k_kw: int | None = Field(default=None, ge=1, le=200)
    fused_limit: int | None = Field(default=None, ge=1, le=200)
    rerank_top_n: int | None = Field(default=None, ge=1, le=50)
    expand_query: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> AskResponse:
    container = get_container()
    use_case: AskQuestionHybridUseCase | None = container.get("ask_hybrid_use_case")
    if not use_case:
        raise HTTPException(status_code=501, detail="Ask use case not configured")

    k_override = request.k
    ask_request = AskHybridRequest(
        tenant_id=tenant_id,
        question=request.question,
        document_id=request.document_id,
        k_vec=request.k_vec or k_override or 30,
        k_kw=request.k_kw or k_override or 30,
        fused_limit=request.fused_limit or 40,
        rerank_top_n=request.rerank_top_n or 8,
        expand_query=request.expand_query,
    )

    answer = use_case.execute(ask_request)
    return AskResponse(answer=answer.text, sources=list(answer.sources))
```

### Streaming Endpoint

Provides real-time response streaming:

```python
@router.post("/ask-stream")
async def ask_question_stream(
    request: AskRequest,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Stream the answer to a question in real-time.
    """
    container = get_container()
    use_case: AskQuestionHybridUseCase | None = container.get("ask_hybrid_use_case")
    if not use_case:
        raise HTTPException(status_code=501, detail="Ask use case not configured")

    # Convert to internal request format
    k_override = request.k
    ask_request = AskHybridRequest(
        tenant_id=tenant_id,
        question=request.question,
        document_id=request.document_id,
        k_vec=request.k_vec or k_override or 30,
        k_kw=request.k_kw or k_override or 30,
        fused_limit=request.fused_limit or 40,
        rerank_top_n=request.rerank_top_n or 8,
        expand_query=request.expand_query,
    )

    # Create streaming response
    async def generate():
        for chunk in use_case.execute_stream(ask_request):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Health and Monitoring Endpoints

### Health Check

Basic health check endpoint:

```python
@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint to verify system status.
    """
    # Check if we can access the container
    try:
        container = get_container()
        # Try to get a service to verify connectivity
        llm = container.get("llm")
        if not llm:
            return HealthCheckResponse(status="degraded", message="LLM service not available")
        
        return HealthCheckResponse(status="healthy", message="All systems operational")
    except Exception as e:
        return HealthCheckResponse(status="unhealthy", message=str(e))
```

### Metrics Endpoint

Prometheus metrics endpoint:

```python
@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    """
    from prometheus_client import generate_latest
    return Response(content=generate_latest(), media_type="text/plain")
```

## GraphQL Integration

The system also provides a GraphQL interface:

```python
# In main.py
async def get_graphql_context(request):
    from src.core.bootstrap import get_container

    container = get_container()
    return {
        "request": request,
        "doc_repo": container.get("document_repo"),
        "chat_repo": container.get("chat_repo"),
        "search_service": container.get("search_documents_use_case"),
        "ask_hybrid_use_case": container.get("ask_hybrid_use_case"),
        # ... other services
    }

graphql_app = GraphQLRouter(
    graphql_schema,
    context_getter=get_graphql_context,
)
app.mount("/graphql", graphql_app, name="graphql")
```

## Error Handling

### Global Exception Handler

The API layer implements comprehensive error handling:

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors globally.
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": [
                {
                    "loc": error["loc"],
                    "msg": error["msg"],
                    "type": error["type"],
                }
                for error in exc.errors()
            ],
        },
    )
```

### Business Logic Errors

Custom error responses for business logic issues:

```python
@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> AskResponse:
    try:
        # ... business logic
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error("Ask request failed", error=str(e), tenant_id=tenant_id)
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Request/Response Models

### Pydantic Models

All API requests and responses use Pydantic models for validation:

```python
class DocumentStatusModel(BaseModel):
    document_id: str
    filename: str
    status: str
    error: str | None = None
    chunks_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}
```

### Field Validation

Models include comprehensive validation:

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    document_id: str | None = None
    k: int | None = Field(default=None, ge=1, le=200, description="Top-K override")
    expand_query: bool = False
```

## Middleware Integration

### CORS

Cross-origin resource sharing configuration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Observability Middleware

Request tracing and metrics collection:

```python
# In main.py
from src.core.observability import setup_observability
setup_observability(app)
```

## Performance Considerations

### Rate Limiting

API rate limiting can be implemented:

```python
# Example rate limiting implementation
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/ask")
@limiter.limit("10/minute")  # Example rate limit
def ask_question_with_limit(...):
    # ... implementation
```

### Request Size Limits

File upload size limits:

```python
# In upload endpoint
if len(content) > settings.max_upload_bytes:
    raise HTTPException(
        status_code=400,
        detail=f"File too large. Max size: {settings.max_upload_mb}MB"
    )
```

## Security Measures

### Input Validation

All inputs are validated using Pydantic:

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)  # Prevents abuse
    document_id: str | None = Field(default=None, pattern=r'^[a-zA-Z0-9_-]+$')  # Sanitizes IDs
```

### Authentication Headers

Secure API key handling:

```python
def get_api_key(security_scheme: HTTPAuthorizationCredentials = Security(security)) -> str:
    api_key = security_scheme.credentials
    # Additional validation here
    return api_key
```

## Testing Considerations

### API Testing

The API layer is designed for easy testing:

```python
# Example test
def test_ask_endpoint():
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    response = client.post(
        "/api/v1/ask",
        json={"question": "What is RAG?"},
        headers={"X-API-KEY": "test-key"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
```

The API layer in RAG Engine Mini provides a robust, secure, and scalable interface to the underlying RAG system, following RESTful principles and leveraging FastAPI's powerful features for validation, documentation, and performance.