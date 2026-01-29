"""
Document Reader Port
=====================
Interface for reading stored file information.

منفذ قراءة المستندات
"""

from typing import Protocol

from src.domain.entities import DocumentId, StoredFile, TenantId


class DocumentReaderPort(Protocol):
    """
    Port for reading stored file metadata.
    
    Design Decision: Separate read port following ISP.
    Workers need to read file info but don't need full repo operations.
    
    قرار التصميم: منفذ قراءة منفصل يتبع مبدأ فصل الواجهات
    """
    
    def get_stored_file(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> StoredFile | None:
        """
        Get stored file metadata for a document.
        
        Args:
            tenant_id: Owner tenant (for isolation)
            document_id: Document to query
            
        Returns:
            Stored file info or None if not found
            
        Note:
            Used by indexing worker to locate the file.
            يستخدم من قبل عامل الفهرسة لتحديد موقع الملف
        """
        ...
