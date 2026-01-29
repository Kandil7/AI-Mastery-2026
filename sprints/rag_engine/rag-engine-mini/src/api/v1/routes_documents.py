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
    from src.domain.entities import DocumentId, TenantId
    
    container = get_container()
    repo = container["document_repo"]
    
    doc = repo.get_document(
        tenant_id=TenantId(tenant_id),
        document_id=DocumentId(document_id),
    )
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return {
        "document_id": doc.id.value,
        "filename": doc.filename,
        "status": doc.status.value,
        "chunks_count": doc.chunks_count,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
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
    from src.domain.entities import TenantId
    
    container = get_container()
    repo = container["document_repo"]
    
    docs = repo.list_documents(
        tenant_id=TenantId(tenant_id),
        limit=limit,
        offset=offset,
    )
    
    total = repo.count_documents(tenant_id=TenantId(tenant_id))
    
    return {
        "documents": [
            {
                "document_id": d.id.value,
                "filename": d.filename,
                "status": d.status.value,
                "chunks_count": d.chunks_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> dict:
    """
    Delete a document and all its chunks.
    
    حذف مستند وجميع متعلقه
    """
    from src.domain.entities import DocumentId, TenantId
    
    container = get_container()
    repo = container["document_repo"]
    
    success = repo.delete_document(
        tenant_id=TenantId(tenant_id),
        document_id=DocumentId(document_id),
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return {"status": "deleted", "document_id": document_id}
