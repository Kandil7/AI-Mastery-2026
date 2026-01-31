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
