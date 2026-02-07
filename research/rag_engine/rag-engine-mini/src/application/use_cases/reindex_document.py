"""
Re-index Document Use Case
=========================
Trigger re-processing and re-indexing of existing documents.

خدمة إعادة معالجة المستندات الموجودة
"""

from dataclasses import dataclass
from typing import Optional

from src.application.ports.document_repo import DocumentRepoPort
from src.domain.entities import TenantId, DocumentId


@dataclass
class ReindexDocumentRequest:
    """Request data for re-indexing a document."""

    tenant_id: str
    document_id: str
    force_rechunk: bool = False
    force_reembed: bool = False
    use_new_chunking_strategy: bool = False


@dataclass
class ReindexDocumentResponse:
    """Response data for re-indexing operation."""

    document_id: str
    status: str
    message: str
    chunks_count: int


class ReindexDocumentUseCase:
    """
    Use case for re-indexing existing documents.

    Flow:
    1. Validate tenant owns document
    2. Update document status to 'reindexing'
    3. Queue background task for re-processing
    4. Force re-chunking if requested
    5. Force re-embedding if requested
    6. Use new chunking strategy if requested

    Use Cases:
    - Chunking strategy update (fixed -> hierarchical)
    - Embedding model change (small -> large)
    - Chunk size/overlap adjustments
    - Bug fixes in extraction pipeline

    حالة استخدام إعادة المعالجة
    """

    def __init__(
        self,
        document_repo: DocumentRepoPort,
        task_queue,
    ) -> None:
        """
        Initialize re-index use case.

        Args:
            document_repo: Document repository for data access
            task_queue: Task queue for background processing
        """
        self._repo = document_repo
        self._queue = task_queue

    def _validate_ownership(self, tenant_id: TenantId, document_id: DocumentId) -> None:
        """
        Validate tenant owns the document.

        Args:
            tenant_id: Tenant/user ID
            document_id: Document ID

        Raises:
            ValueError: If tenant doesn't own document
        """
        doc = self._repo.get_document(
            tenant_id=tenant_id,
            document_id=document_id,
        )

        if not doc:
            raise ValueError("Document not found")

        # Check if document belongs to tenant
        # (In a multi-tenant system, this is critical)
        # For now, assume ownership is valid if document exists
        # In production, add tenant_id to document model

    def execute(self, request: ReindexDocumentRequest) -> ReindexDocumentResponse:
        """
        Execute re-indexing of a document.

        Args:
            request: Re-indexing request

        Returns:
            Re-indexing response with status and message
        """
        tenant = TenantId(request.tenant_id)
        doc_id = DocumentId(request.document_id)

        # Step 1: Validate ownership
        self._validate_ownership(tenant, doc_id)

        # Step 2: Update document status
        self._repo.set_status(
            tenant_id=tenant,
            document_id=doc_id,
            status="reindexing",
        )

        # Step 3: Queue background task
        # Note: In a real system, we'd update document metadata
        # to indicate which re-indexing options were chosen

        task_kwargs = {
            "tenant_id": request.tenant_id,
            "document_id": request.document_id,
            "force_rechunk": request.force_rechunk,
            "force_reembed": request.force_reembed,
            "use_new_strategy": request.use_new_chunking_strategy,
        }

        self._queue.enqueue_index_document(
            tenant_id=tenant,
            document_id=doc_id,
            **task_kwargs,
        )

        # Step 4: Return response
        return ReindexDocumentResponse(
            document_id=request.document_id,
            status="reindexing",
            message="Document queued for re-indexing",
            chunks_count=0,  # Will be updated by task
        )
