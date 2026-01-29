"""
Document Routes
================
Endpoints for document upload and management.

نقاط نهاية رفع وإدارة المستندات
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container
from src.application.use_cases.upload_document import (
    UploadDocumentRequest,
    UploadDocumentUseCase,
)
from src.domain.errors import (
    DomainError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class UploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    status: str
    message: str = ""


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
) -> UploadResponse:
    """
    Upload a document for indexing.
    
    The document will be:
    1. Validated (size, type)
    2. Stored
    3. Queued for async indexing
    
    Supports PDF, DOCX, and TXT files.
    
    رفع مستند للفهرسة - يدعم PDF و DOCX و TXT
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    # Read file data
    data = await file.read()
    
    # Get use case from container
    container = get_container()
    use_case: UploadDocumentUseCase = container["upload_use_case"]
    
    try:
        result = await use_case.execute(
            UploadDocumentRequest(
                tenant_id=tenant_id,
                filename=file.filename,
                content_type=file.content_type or "application/octet-stream",
                data=data,
            )
        )
        
        return UploadResponse(
            document_id=result.document_id.value,
            status=result.status,
            message=result.message,
        )
        
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except DomainError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Get document processing status.
    
    Returns current status (queued, processing, indexed, failed).
    
    الحصول على حالة معالجة المستند
    """
    # TODO: Implement with document_repo.get_status()
    return {
        "document_id": document_id,
        "status": "indexed",  # Placeholder
        "chunks_count": 0,
    }


@router.get("")
async def list_documents(
    tenant_id: str = Depends(get_tenant_id),
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    List documents for the current tenant.
    
    قائمة المستندات للمستأجر الحالي
    """
    # TODO: Implement with document_repo.list_documents()
    return {
        "documents": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
    }
