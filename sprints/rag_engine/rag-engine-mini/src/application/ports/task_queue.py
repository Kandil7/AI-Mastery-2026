"""
Task Queue Port
================
Interface for background task queuing.

منفذ قائمة المهام
"""

from typing import Protocol

from src.domain.entities import DocumentId, TenantId


class TaskQueuePort(Protocol):
    """
    Port for enqueueing background tasks.
    
    Implementation: Celery with Redis broker
    
    Design Decision: Async indexing for:
    - Non-blocking API responses
    - Better resource utilization
    - Retry handling
    - Scalable workers
    
    قرار التصميم: فهرسة غير متزامنة لاستجابات API غير محجوبة
    """
    
    def enqueue_index_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> str:
        """
        Enqueue a document for indexing.
        
        Args:
            tenant_id: Owner tenant
            document_id: Document to index
            
        Returns:
            Task ID for tracking
            
        Note:
            The task will:
            1. Extract text from document
            2. Chunk the text
            3. Generate embeddings (batch)
            4. Upsert to vector store
            5. Store chunks in Postgres
            6. Update document status
            
            ستقوم المهمة بالاستخراج والتقطيع والتضمين والتخزين
        """
        ...

    def enqueue_bulk_upload(
        self,
        *,
        tenant_id: TenantId,
        files: list[dict],
    ) -> str:
        """Enqueue bulk upload task and return task ID."""
        ...

    def enqueue_bulk_delete(
        self,
        *,
        tenant_id: TenantId,
        document_ids: list[str],
    ) -> str:
        """Enqueue bulk delete task and return task ID."""
        ...

    def enqueue_merge_pdfs(
        self,
        *,
        tenant_id: TenantId,
        source_document_ids: list[str],
        merged_filename: str,
        target_document_id: str | None = None,
    ) -> str:
        """Enqueue PDF merge task and return task ID."""
        ...
