"""
Celery Task Queue Adapter
==========================
Implementation of TaskQueuePort for Celery.

محول قائمة المهام Celery
"""

from celery import Celery

from src.domain.entities import DocumentId, TenantId


class CeleryTaskQueue:
    """
    Celery adapter implementing TaskQueuePort.
    
    محول Celery لقائمة المهام
    """
    
    def __init__(self, celery_app: Celery) -> None:
        """
        Initialize with Celery app.
        
        Args:
            celery_app: Configured Celery application
        """
        self._app = celery_app
    
    def enqueue_index_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
        **kwargs,
    ) -> str:
        """
        Enqueue document for indexing.
        
        Returns task ID for tracking.
        """
        result = self._app.send_task(
            "index_document",
            kwargs={
                "tenant_id": tenant_id.value,
                "document_id": document_id.value,
                **kwargs,
            },
            queue="indexing",
        )
        return result.id

    def enqueue_bulk_upload(
        self,
        *,
        tenant_id: TenantId,
        files: list[dict],
    ) -> str:
        """Enqueue bulk upload task."""
        result = self._app.send_task(
            "bulk_upload_documents",
            kwargs={
                "tenant_id": tenant_id.value,
                "files": files,
            },
            queue="indexing",
        )
        return result.id

    def enqueue_bulk_delete(
        self,
        *,
        tenant_id: TenantId,
        document_ids: list[str],
    ) -> str:
        """Enqueue bulk delete task."""
        result = self._app.send_task(
            "bulk_delete_documents",
            kwargs={
                "tenant_id": tenant_id.value,
                "document_ids": document_ids,
            },
            queue="indexing",
        )
        return result.id

    def enqueue_merge_pdfs(
        self,
        *,
        tenant_id: TenantId,
        source_document_ids: list[str],
        merged_filename: str,
        target_document_id: str | None = None,
    ) -> str:
        """Enqueue PDF merge task."""
        result = self._app.send_task(
            "merge_pdfs",
            kwargs={
                "tenant_id": tenant_id.value,
                "source_document_ids": source_document_ids,
                "merged_filename": merged_filename,
                "target_document_id": target_document_id,
            },
            queue="indexing",
        )
        return result.id
