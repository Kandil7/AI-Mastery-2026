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
