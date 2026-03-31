"""
Upload Document Use Case
=========================
Orchestrates document upload with idempotency.

حالة استخدام رفع المستندات
"""

import hashlib
from typing import AsyncIterator
from dataclasses import dataclass

from src.application.ports.document_idempotency import DocumentIdempotencyPort
from src.application.ports.document_repo import DocumentRepoPort
from src.application.ports.file_store import FileStorePort
from src.application.ports.task_queue import TaskQueuePort
from src.domain.entities import TenantId, UploadResult


@dataclass
class UploadDocumentRequest:
    """Request data for document upload."""
    tenant_id: str
    filename: str
    content_type: str
    data: bytes


@dataclass
class UploadDocumentStreamRequest:
    """Request data for streaming document upload."""
    tenant_id: str
    filename: str
    content_type: str
    data_stream: AsyncIterator[bytes]


class UploadDocumentUseCase:
    """
    Use case for uploading and queueing documents for indexing.
    
    Flow:
    1. Calculate file hash (SHA256)
    2. Check for existing document with same hash (idempotency)
    3. If exists, return existing document ID
    4. Store file
    5. Create document record with hash
    6. Enqueue for async indexing
    7. Return document ID with status
    
    Design Decision: Idempotent upload prevents:
    - Wasted storage for duplicate files
    - Wasted compute for re-indexing
    - Duplicate chunks in search results
    
    قرار التصميم: الرفع المتساوي يمنع الهدر في التخزين والحوسبة
    
    Example:
        >>> uc = UploadDocumentUseCase(...)
        >>> result = await uc.execute(UploadDocumentRequest(...))
        >>> result.status  # "queued" or "already_exists"
    """
    
    def __init__(
        self,
        *,
        file_store: FileStorePort,
        document_repo: DocumentRepoPort,
        idempotency_repo: DocumentIdempotencyPort,
        task_queue: TaskQueuePort,
    ) -> None:
        """
        Initialize the use case with required ports.
        
        All ports are injected - no direct adapter dependencies.
        كل المنافذ محقونة - لا تبعيات محول مباشرة
        """
        self._file_store = file_store
        self._repo = document_repo
        self._idem = idempotency_repo
        self._queue = task_queue
    
    async def execute(self, request: UploadDocumentRequest) -> UploadResult:
        """
        Execute the upload document use case.
        
        Args:
            request: Upload request with file data
            
        Returns:
            Upload result with document ID and status
        """
        tenant = TenantId(request.tenant_id)
        
        # Step 1: Calculate file hash
        file_sha256 = hashlib.sha256(request.data).hexdigest()
        
        # Step 2: Check for existing document (idempotency)
        existing = self._idem.get_by_file_hash(
            tenant_id=tenant,
            file_sha256=file_sha256,
        )
        
        if existing:
            # Document already exists with same content
            return UploadResult(
                document_id=existing,
                status="already_exists",
                message="Document with same content already indexed",
            )
        
        # Step 3: Store the file
        stored = await self._file_store.save_upload(
            tenant_id=tenant.value,
            upload_filename=request.filename,
            content_type=request.content_type,
            data=request.data,
        )
        
        # Step 4: Create document record with hash
        doc_id = self._idem.create_document_with_hash(
            tenant_id=tenant,
            stored_file=stored,
            file_sha256=file_sha256,
        )
        
        # Step 5: Set initial status
        self._repo.set_status(
            tenant_id=tenant,
            document_id=doc_id,
            status="queued",
        )
        
        # Step 6: Enqueue for async indexing
        self._queue.enqueue_index_document(
            tenant_id=tenant,
            document_id=doc_id,
        )
        
        # Step 7: Return result
        return UploadResult(
            document_id=doc_id,
            status="queued",
            message="Document queued for indexing",
        )

    async def execute_stream(self, request: UploadDocumentStreamRequest) -> UploadResult:
        """
        Execute the upload document use case using a byte stream.

        Streams the file to storage, then uses the computed hash for idempotency.
        """
        tenant = TenantId(request.tenant_id)

        # Step 1: Stream to storage and compute hash
        stored, file_sha256 = await self._file_store.save_upload_stream(
            tenant_id=tenant.value,
            upload_filename=request.filename,
            content_type=request.content_type,
            data_stream=request.data_stream,
        )

        # Step 2: Check for existing document (idempotency)
        existing = self._idem.get_by_file_hash(
            tenant_id=tenant,
            file_sha256=file_sha256,
        )

        if existing:
            # Duplicate found; remove stored file
            await self._file_store.delete(stored.path)
            return UploadResult(
                document_id=existing,
                status="already_exists",
                message="Document with same content already indexed",
            )

        # Step 3: Create document record with hash
        doc_id = self._idem.create_document_with_hash(
            tenant_id=tenant,
            stored_file=stored,
            file_sha256=file_sha256,
        )

        # Step 4: Set initial status
        self._repo.set_status(
            tenant_id=tenant,
            document_id=doc_id,
            status="queued",
        )

        # Step 5: Enqueue for async indexing
        self._queue.enqueue_index_document(
            tenant_id=tenant,
            document_id=doc_id,
        )

        return UploadResult(
            document_id=doc_id,
            status="queued",
            message="Document queued for indexing",
        )
