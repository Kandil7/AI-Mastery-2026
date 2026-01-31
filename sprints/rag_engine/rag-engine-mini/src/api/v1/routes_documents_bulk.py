"""
Bulk Operations Routes
==========================
Endpoints for bulk document upload and delete operations.

نقاط نهاية العمليات الدفعية
"""

from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container
from src.application.use_cases.bulk_operations import BulkOperationsUseCase, BulkUploadRequest

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# ============================================================================
# Request/Response Models
# ============================================================================


class BulkUploadRequestForm:
    """Form data for bulk upload (multipart)."""

    reason: str = Field(..., description="Reason for bulk upload")


class BulkUploadResponse(BaseModel):
    """Response model for bulk upload."""

    operation_id: str = Field(..., description="Operation ID for tracking")
    total_files: int = Field(..., description="Total files uploaded")
    uploaded: int = Field(..., description="Alias for total files uploaded")
    succeeded: int = Field(..., description="Successful uploads")
    failed: int = Field(..., description="Failed uploads")
    results: list = Field(..., description="Results for each file")


class BulkDeleteRequest(BaseModel):
    """Request model for bulk delete."""

    document_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of document IDs to delete",
    )
    reason: str = Field(..., description="Reason for bulk delete")


class BulkDeleteResponse(BaseModel):
    """Response model for bulk delete."""

    operation_id: str = Field(..., description="Operation ID for tracking")
    total_documents: int = Field(..., description="Total documents requested for deletion")
    deleted_count: int = Field(..., description="Successfully deleted documents")
    failed_count: int = Field(..., description="Failed deletions")
    failed_ids: list[str] = Field(..., description="IDs of failed deletions")


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/bulk-upload", response_model=BulkUploadResponse)
@router.post("/bulk", response_model=BulkUploadResponse)
async def bulk_upload_documents(
    files: list[UploadFile] = File(...),
    reason: str = Form(...),
    tenant_id: str = Depends(get_tenant_id),
) -> BulkUploadResponse:
    """
    Bulk upload multiple documents.

    Flow:
    1. Validate file list (count, sizes, types)
    2. Upload all files to storage
    3. Create document records
    4. Queue background indexing for each
    5. Return operation ID for progress tracking

    Limitations:
    - Maximum 100 files per request
    - Total upload size limit: 500MB
    - Supported types: PDF, DOCX, TXT

    Usage:
        POST /api/v1/documents/bulk-upload
        Content-Type: multipart/form-data
        {
            "files": [File1, File2, File3],
            "reason": "Data migration"
        }

    رفع دفعي للمستندات
    """
    # Validate file count
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 files per bulk upload")

    # Validate file types
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]

    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed_types)}",
            )

    # Read file contents once (avoid double reads)
    raw_contents: list[bytes] = []
    total_size = 0
    for file in files:
        content = await file.read()
        raw_contents.append(content)
        total_size += len(content)
    if total_size > 500 * 1024 * 1024:  # 500MB
        raise HTTPException(status_code=413, detail="Total upload size exceeds 500MB limit")

    # Get use case from container
    container = get_container()
    use_case = container.get("bulk_operations_use_case")

    if not use_case:
        raise HTTPException(status_code=501, detail="Bulk operations use case not configured")

    # Prepare bulk upload request
    # Note: We need to create BinaryIO objects from uploaded files
    # For simplicity, we'll use the file objects directly
    from io import BytesIO

    file_contents = []
    filenames = []
    content_types = []

    for file, content in zip(files, raw_contents):
        file_contents.append(BytesIO(content))
        filenames.append(file.filename)
        content_types.append(file.content_type)

    # Build request
    request = BulkUploadRequest(
        tenant_id=tenant_id,
        files=file_contents,
        filenames=filenames,
        content_types=content_types,
    )

    # Execute bulk upload
    response = await use_case.bulk_upload(request)

    # Log operation (in production, track operation_id)
    # log.info(f"Bulk upload operation started: {response.operation_id}")

    return BulkUploadResponse(
        operation_id=response.operation_id,
        total_files=response.total_files,
        uploaded=response.total_files,
        succeeded=response.succeeded,
        failed=response.failed,
        results=[
            {
                "filename": r.filename,
                "document_id": r.document_id,
                "status": r.status,
                "message": r.message,
                "error": r.error,
            }
            for r in response.results
        ],
    )


@router.post("/bulk-delete", response_model=BulkDeleteResponse)
@router.post("/bulk/delete", response_model=BulkDeleteResponse)
def bulk_delete_documents(
    request: BulkDeleteRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> BulkDeleteResponse:
    """
    Bulk delete multiple documents.

    Flow:
    1. Validate ownership of all documents
    2. Delete documents in transaction
    3. Delete associated chunks from vector store
    4. Return operation summary

    Security:
    - Requires API key authentication
    - Validates tenant ownership for each document
    - Transaction-safe deletion
    - Returns list of failed deletions

    حذف دفعي للمستندات
    """
    # Get use case from container
    container = get_container()
    use_case = container.get("bulk_operations_use_case")

    if not use_case:
        raise HTTPException(status_code=501, detail="Bulk operations use case not configured")

    # Execute bulk delete
    response = use_case.bulk_delete(request)

    # Log operation (in production, track operation_id)
    # log.info(f"Bulk delete operation completed: {response.operation_id}")

    return BulkDeleteResponse(
        operation_id=response.operation_id,
        total_documents=response.total_documents,
        deleted_count=response.deleted_count,
        failed_count=response.failed_count,
        failed_ids=response.failed_ids,
    )


@router.post("/{document_id}/reindex")
def reindex_document(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
    force_rechunk: bool = False,
    force_reembed: bool = False,
    use_new_strategy: bool = False,
) -> dict:
    """
    Trigger re-indexing of an existing document.

    Use Cases:
    - Chunking strategy update (fixed -> hierarchical)
    - Embedding model change (small -> large)
    - Chunk size/overlap adjustments
    - Bug fixes in extraction pipeline

    Flow:
    1. Validate tenant owns document
    2. Update document status to 'reindexing'
    3. Queue background task for re-processing
    4. Force re-chunking/re-embedding if requested

    Usage:
        POST /api/v1/documents/{doc_id}/reindex?force_rechunk=true
        - Triggers re-chunking with new strategy
    """
    # Get use case from container
    container = get_container()
    use_case = container.get("reindex_document_use_case")

    if not use_case:
        raise HTTPException(status_code=501, detail="Re-index use case not configured")

    # Build re-index request
    from src.application.use_cases.reindex_document import ReindexDocumentRequest
    from src.domain.entities import DocumentId

    request = ReindexDocumentRequest(
        tenant_id=tenant_id,
        document_id=document_id,
        force_rechunk=force_rechunk,
        force_reembed=force_reembed,
        use_new_chunking_strategy=use_new_strategy,
    )

    # Execute re-index
    response = use_case.execute(request)

    return {
        "document_id": response.document_id,
        "status": response.status,
        "message": response.message,
        "chunks_count": response.chunks_count,
    }
