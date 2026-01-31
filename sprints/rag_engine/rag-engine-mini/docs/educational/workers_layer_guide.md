# RAG Engine Mini - Workers Layer Deep Dive

## Introduction

The workers layer in RAG Engine Mini handles background processing tasks using Celery, a distributed task queue system. This layer is responsible for computationally intensive operations like document indexing, which would be too slow to perform synchronously in API requests. The workers layer enables scalability and responsiveness by offloading heavy processing to background workers.

## Architecture Overview

The workers layer consists of:

1. **Celery Application**: The core task queue system
2. **Background Tasks**: Specific operations performed asynchronously
3. **Task Orchestration**: How tasks are triggered and coordinated
4. **Monitoring**: Tracking task execution and performance

## Celery Configuration

### Celery Application Setup

The Celery application is configured in `celery_app.py`:

```python
from celery import Celery
from src.core.config import settings

celery_app = Celery(
    "rag_workers",
    broker=settings.celery_broker_url,  # Redis or RabbitMQ
    backend=settings.celery_result_backend,  # Results storage
    include=[
        "src.workers.tasks",  # Import task modules
    ],
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=settings.celery_worker_concurrency,
    task_time_limit=settings.celery_task_time_limit,
)
```

### Configuration Options

- **Broker**: Message broker (typically Redis) for queuing tasks
- **Backend**: Storage for task results
- **Concurrency**: Number of concurrent worker processes
- **Time Limits**: Maximum execution time for tasks

## Core Background Tasks

### Document Indexing Task

The primary task is document indexing, implemented in `index_document`:

```python
@celery_app.task(
    name="index_document",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
)
def index_document(
    self,
    *,
    tenant_id: str,
    document_id: str,
    force_rechunk: bool = False,
    force_reembed: bool = False,
    use_new_strategy: bool = False,
) -> dict:
    """
    Index a document with Stage 4 features:
    1. Extract Text & Tables (High Precision)
    2. Extract & Describe Images (LLM-Vision)
    3. Hierarchical & Contextual linking
    4. Graph Triplet extraction
    """
```

#### Indexing Process Steps

1. **Setup and Validation**
   - Retrieve document from storage
   - Update status to "processing"
   - Prepare dependencies from DI container

2. **Text Extraction**
   - Extract text from document using appropriate parser
   - Handle different file formats (PDF, DOCX, etc.)

3. **Multi-Modal Extraction**
   - Extract images from PDF documents
   - Use vision service to describe images
   - Store image descriptions as chunks

4. **Document Summarization**
   - Generate document summary using LLM
   - Use summary as context for chunks

5. **Hierarchical Chunking**
   - Split document into parent-child chunks
   - Maintain context between related chunks

6. **Embedding Generation**
   - Create embeddings for chunks
   - Use caching to avoid redundant API calls

7. **Storage**
   - Store chunks in database
   - Store vectors in vector store
   - Create knowledge graph triplets

8. **Finalization**
   - Update document status to "indexed"
   - Record metrics and observability data

### Bulk Upload Task

Handles multiple document uploads efficiently:

```python
@celery_app.task(
    name="bulk_upload_documents",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def bulk_upload_documents(
    self,
    *,
    tenant_id: str,
    files: List[Dict[str, Any]],
) -> dict:
    """
    Process multiple document uploads in bulk.
    """
```

This task:
- Processes multiple files in a single operation
- Stores each file in the file system
- Creates document records in the database
- Queues individual indexing tasks

### Bulk Delete Task

Handles deletion of multiple documents:

```python
@celery_app.task(
    name="bulk_delete_documents",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def bulk_delete_documents(
    self,
    *,
    tenant_id: str,
    document_ids: List[str],
) -> dict:
    """
    Delete multiple documents in bulk.
    """
```

This task ensures complete cleanup:
- Removes vectors from vector store
- Deletes chunks from database
- Removes document records

### PDF Merging Task

Combines multiple PDFs into one:

```python
@celery_app.task(
    name="merge_pdfs",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def merge_pdfs(
    self,
    *,
    tenant_id: str,
    source_document_ids: List[str],
    merged_filename: str,
    target_document_id: str | None = None,
) -> dict:
    """
    Merge multiple PDF documents into one.
    """
```

### Chat Enhancement Tasks

Additional tasks for improving chat experiences:

#### Title Generation
```python
@celery_app.task(
    name="generate_chat_title",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
)
def generate_chat_title(
    self,
    *,
    tenant_id: str,
    session_id: str,
) -> dict:
    """
    Generate a title for a chat session using LLM.
    """
```

#### Session Summarization
```python
@celery_app.task(
    name="summarize_chat_session",
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
)
def summarize_chat_session(
    self,
    *,
    tenant_id: str,
    session_id: str,
) -> dict:
    """
    Generate a summary for a completed chat session.
    """
```

## Task Retry and Error Handling

### Automatic Retries

Tasks are configured with automatic retry mechanisms:

```python
@celery_app.task(
    name="index_document",
    bind=True,
    autoretry_for=(Exception,),  # Retry on any exception
    retry_backoff=True,          # Exponential backoff
    retry_kwargs={"max_retries": 5},  # Max 5 retries
)
```

### Error Handling

Comprehensive error handling ensures proper cleanup:

```python
try:
    # Task execution
    result = perform_indexing(...)
    return result
except Exception as e:
    # Update document status to failed
    document_repo.set_status(
        tenant_id=tenant, 
        document_id=doc_id, 
        status="failed", 
        error=str(e)
    )
    logger.exception("indexing_failed", error=str(e))
    # Increment failure metrics
    CELERY_TASK_COUNT.labels(task="index_document", status="failure").inc()
    raise  # Re-raise to trigger retry
```

## Task Monitoring and Observability

### Metrics Collection

Tasks collect various metrics:

```python
from src.core.observability import CELERY_TASK_COUNT, CELERY_TASK_DURATION

# At the beginning of task
start_time = time.time()

# On success
CELERY_TASK_COUNT.labels(task="index_document", status="success").inc()
CELERY_TASK_DURATION.labels(task="index_document").observe(time.time() - start_time)

# On failure
CELERY_TASK_COUNT.labels(task="index_document", status="failure").inc()
CELERY_TASK_DURATION.labels(task="index_document").observe(time.time() - start_time)
```

### Logging

Structured logging with correlation:

```python
import structlog
log = structlog.get_logger()

log.info("indexing_complete", chunks=len(chunk_ids_in_order))
log.exception("indexing_failed", error=str(e))
```

## Task Coordination

### Triggering Tasks

Tasks are triggered from various parts of the system:

From upload use case:
```python
# After creating document record
index_document.delay(
    tenant_id=tenant_id,
    document_id=document_id.value,
)
```

From API endpoints:
```python
# For bulk operations
result = bulk_upload_documents.delay(
    tenant_id=current_user.id,
    files=processed_files
)
```

### Task Dependencies

Some tasks depend on others:

```python
# After indexing completes, trigger title generation
index_result = index_document.apply_async(
    kwargs={
        "tenant_id": tenant_id,
        "document_id": doc_id
    }
)

# Chain with cleanup task
cleanup_task = cleanup_temp_files.s(doc_id)
index_result.link(cleanup_task)
```

## Multi-Tenancy in Workers

All worker tasks maintain tenant isolation:

```python
def index_document(
    self,
    *,
    tenant_id: str,  # Passed from API layer
    document_id: str,
):
    # Convert to domain object
    tenant = TenantId(tenant_id)
    doc_id = DocumentId(document_id)
    
    # All operations are tenant-aware
    document_repo.set_status(tenant_id=tenant, document_id=doc_id, status="processing")
    
    # Vector store operations include tenant filter
    vector_store.upsert_points(
        ids=[c_id],
        vectors=[vec_map[c_text]],
        tenant_id=tenant.value,  # Tenant isolation in vector store
        document_id=doc_id.value,
    )
```

## Performance Considerations

### Memory Management

Workers handle large documents efficiently:

```python
def index_document(...):
    # Process documents in chunks to manage memory
    hierarchy = chunk_hierarchical(full_text, spec)
    
    # Batch operations for efficiency
    child_texts = [h["child_text"] for h in hierarchy]
    unique_child_texts = list(set(child_texts))  # Remove duplicates
    unique_vectors = cached_embeddings.embed_many(unique_child_texts)
```

### Concurrency Control

Celery workers are configured for optimal concurrency:

```python
# In celery configuration
worker_concurrency=settings.celery_worker_concurrency  # Typically 4-8 processes
task_time_limit=settings.celery_task_time_limit       # Prevent hanging tasks
```

### Resource Cleanup

Tasks properly clean up resources:

```python
def index_document(...):
    try:
        # Task execution
        pass
    finally:
        # Ensure cleanup happens even if task fails
        if 'pdf_doc' in locals():
            pdf_doc.close()
```

## Security Considerations

### Input Validation

Worker tasks validate inputs:

```python
def index_document(
    self,
    *,
    tenant_id: str,
    document_id: str,
):
    # Validate tenant_id format
    if not tenant_id or len(tenant_id) > 100:
        raise ValueError("Invalid tenant_id")
    
    # Validate document_id format
    if not document_id or len(document_id) > 100:
        raise ValueError("Invalid document_id")
```

### File Path Security

When processing files, paths are validated:

```python
def get_stored_file(...):
    # Sanitize file paths to prevent directory traversal
    stored_file = document_reader.get_stored_file(...)
    if not stored_file.path.startswith(settings.upload_dir):
        raise ValueError("Invalid file path")
```

## Testing Workers

### Unit Testing

Tasks can be tested synchronously:

```python
def test_index_document_task():
    # Call task synchronously for testing
    result = index_document(
        tenant_id="test-tenant",
        document_id="test-doc"
    )
    
    # Assert results
    assert result["ok"] is True
    assert result["chunks"] > 0
```

### Integration Testing

Test task integration with the system:

```python
def test_document_upload_integration():
    # Upload document via API
    response = client.post("/api/v1/documents/upload", ...)
    
    # Wait for indexing to complete
    time.sleep(2)  # Simulate async processing
    
    # Verify document is indexed
    search_response = client.post("/api/v1/ask", json={"question": "test"})
    assert search_response.status_code == 200
```

The workers layer in RAG Engine Mini provides a robust, scalable solution for handling computationally intensive background tasks while maintaining system responsiveness and reliability.