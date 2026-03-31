# Cloud File Storage + Celery Tasks (Deep Dive)

This guide explains, step-by-step, how RAG Engine Mini wires cloud file storage (S3/GCS/Azure), adds streaming uploads, and processes documents asynchronously via Celery. It is intentionally detailed so you can understand every decision and how the pieces fit together.

## What You Will Build

- A filestore factory that selects a backend (local, S3, GCS, Azure) based on configuration.
- Streaming upload support that avoids loading large files in memory.
- Celery tasks for background processing and monitoring hooks for observability.
- A bulk async upload endpoint that queues work via Celery.

## Prerequisites

- Python 3.10+
- FastAPI running (for API testing)
- Redis running (for Celery broker/backend)
- Optional: Cloud SDK libraries
  - S3: boto3
  - GCS: google-cloud-storage
  - Azure: azure-storage-blob

## High-Level Architecture

1. Upload API receives files and validates them.
2. FileStorePort abstracts storage.
3. FileStore factory returns the right adapter (Local/S3/GCS/Azure).
4. Upload use case streams the file and computes SHA256 for idempotency.
5. TaskQueuePort abstracts task queue.
6. CeleryTaskQueue enqueues long-running work.
7. Celery workers execute indexing, bulk operations, and merge jobs.
8. Observability exports task counters and duration metrics.

## Step 1: Configuration (.env)

Use .env.example as a starting point and choose a filestore backend:

```
FILESTORE_BACKEND=local
# or: s3, gcs, azure
```

Set the backend-specific variables:

- S3
```
S3_BUCKET=rag-engine-uploads
S3_REGION=us-east-1
S3_ACCESS_KEY_ID=
S3_SECRET_ACCESS_KEY=
S3_PREFIX=uploads/
```

- GCS
```
GCS_BUCKET=rag-engine-uploads
GCS_PROJECT_ID=
GCS_CREDENTIALS_PATH=
GCS_PREFIX=uploads/
```

- Azure Blob
```
AZURE_CONTAINER=rag-engine-uploads
AZURE_CONNECTION_STRING=
AZURE_ACCOUNT_URL=
AZURE_ACCOUNT_NAME=
AZURE_ACCOUNT_KEY=
AZURE_PREFIX=uploads/
```

Celery broker/backend (Redis):

```
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

## Step 2: FileStorePort (Contract)

FileStorePort defines the required interface for storage adapters. It now includes a streaming method:

- save_upload(...) for in-memory bytes
- save_upload_stream(...) for streaming chunks
- delete(...), exists(...)

See: src/application/ports/file_store.py

Why streaming?
- Prevents large files from being fully loaded into RAM.
- Allows chunk-by-chunk hashing and size enforcement.

## Step 3: FileStore Factory

The factory creates a backend adapter based on config:

```
create_file_store(settings)
```

Backends:
- local -> LocalFileStore
- s3 -> S3FileStore
- gcs -> GCSFileStore
- azure -> AzureBlobFileStore

See: src/adapters/filestore/factory.py

Why a factory?
- Keeps DI wiring clean.
- Centralizes backend selection.
- Makes tests simpler (swap backend with config).

## Step 4: Streaming Upload in LocalFileStore

Local storage now supports streaming:

- Creates a temp file.
- Writes chunks and updates SHA256.
- Enforces size limit.
- Renames temp file into final name.

See: src/adapters/filestore/local_store.py

Key ideas:
- tempfile.NamedTemporaryFile(delete=False)
- aiofiles for async writes
- sha256 streaming hash

## Step 5: Streaming Upload in Cloud Backends

S3/GCS/Azure all implement save_upload_stream(...):

- Stream bytes into a temp file (local).
- Compute SHA256.
- Upload to cloud using the SDK.
- Return a URI (s3://, gs://, azure://).

See:
- src/adapters/filestore/s3_store.py
- src/adapters/filestore/gcs_store.py
- src/adapters/filestore/azure_blob_store.py

Why temp file instead of direct streaming?
- Keeps code simple and avoids SDK streaming edge cases.
- Works consistently across providers.

## Step 6: Upload Use Case (Streaming)

UploadDocumentUseCase now provides:

- execute(...) for in-memory uploads
- execute_stream(...) for streaming uploads

Flow for execute_stream:
1. Stream file to storage and compute SHA256
2. Check idempotency by hash
3. If duplicate -> delete stored file
4. Create document record
5. Set status to queued
6. Enqueue indexing task

See: src/application/use_cases/upload_document.py

## Step 7: API Upload Endpoint (Streaming)

The document upload endpoint now streams the file:

- Reads chunks from UploadFile.read()
- Passes the async generator to execute_stream(...)

See: src/api/v1/routes_documents.py

Benefits:
- Lower memory usage
- Consistent hash + size enforcement

## Step 8: Celery Task Queue Port

TaskQueuePort now includes methods for:

- enqueue_index_document
- enqueue_bulk_upload
- enqueue_bulk_delete
- enqueue_merge_pdfs

See: src/application/ports/task_queue.py

## Step 9: Celery Adapter

CeleryTaskQueue maps port calls to Celery tasks:

- index_document
- bulk_upload_documents
- bulk_delete_documents
- merge_pdfs

See: src/adapters/queue/celery_queue.py

## Step 10: Celery Tasks (Background Processing)

src/workers/tasks.py defines tasks for:

- indexing a document
- bulk upload
- bulk delete
- merge PDFs
- chat title generation
- chat session summarization

Key improvements:
- No await inside sync tasks
- _run_async(...) helper for bridging
- try/except wraps for robust metrics

## Step 11: Monitoring + Metrics

Added Prometheus metrics:

- rag_celery_tasks_total
- rag_celery_task_duration_seconds

See: src/core/observability.py

Worker status endpoint uses Celery inspect:

GET /api/v1/admin/workers/status

See: src/workers/monitoring.py and src/api/v1/routes_admin.py

## Step 12: Async Bulk Upload Endpoint

New endpoint:

POST /api/v1/documents/bulk-upload-async

Flow:
1. Validate input
2. Read files once
3. Enqueue Celery bulk upload task
4. Return operation_id + task_id

See: src/api/v1/routes_documents_bulk.py

## Exercises

1. Switch FILESTORE_BACKEND between local and s3 and observe URIs in StoredFile.
2. Upload a large PDF and confirm memory stays stable (watch top).
3. Enable Celery workers and call /bulk-upload-async.
4. Check /api/v1/admin/workers/status for worker state.

## Troubleshooting

- S3/GCS/Azure errors: verify credentials and bucket/container names.
- Celery not processing: ensure Redis is running and worker is started.
- Large uploads fail: check MAX_UPLOAD_MB and total size limits.

## Summary

You now have:
- Cloud-capable file storage
- Streaming uploads
- Robust background task processing
- Task monitoring and metrics

This foundation allows you to scale ingestion and indexing without blocking API requests.
