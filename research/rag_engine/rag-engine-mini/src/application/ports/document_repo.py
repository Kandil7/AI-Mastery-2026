"""
Document Repository Port
=========================
Interface for document metadata persistence.

منفذ مستودع المستندات
"""

from typing import Protocol, Sequence, Any

from src.domain.entities import DocumentId, DocumentStatus, StoredFile, TenantId


class DocumentRepoPort(Protocol):
    """
    Port for document metadata operations.
    
    Implementation: PostgreSQL
    
    Design Decision: Separate from file storage.
    Documents table stores metadata, file system stores actual files.
    
    قرار التصميم: منفصل عن تخزين الملفات
    """
    
    def create_document(
        self,
        *,
        tenant_id: TenantId,
        stored_file: StoredFile,
        file_sha256: str | None = None,
    ) -> DocumentId:
        """
        Create a new document record.
        
        Args:
            tenant_id: Owner tenant
            stored_file: File storage information
            file_sha256: Optional hash for idempotency
            
        Returns:
            New document ID
        """
        ...
    
    def set_status(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
        status: str,
        error: str | None = None,
    ) -> None:
        """
        Update document processing status.
        
        Args:
            tenant_id: Owner tenant (for isolation)
            document_id: Document to update
            status: New status (created/queued/processing/indexed/failed)
            error: Error message if status is 'failed'
        """
        ...
    
    def get_status(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> DocumentStatus | None:
        """
        Get document status.
        
        Args:
            tenant_id: Owner tenant
            document_id: Document to query
            
        Returns:
            Document status or None if not found
        """
        ...
    
    def list_documents(
        self,
        *,
        tenant_id: TenantId,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[DocumentStatus]:
        """
        List documents for a tenant.
        
        Args:
            tenant_id: Owner tenant
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of document statuses
        """
        ...
    
    def delete_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> bool:
        """
        Delete a document.
        
        Args:
            tenant_id: Owner tenant
            document_id: Document to delete
            
        Returns:
            True if deleted, False if not found
        """
        ...

    def count_documents(
        self,
        *,
        tenant_id: TenantId,
        filters: Any | None = None,
    ) -> int:
        """
        Count documents for a tenant.

        Args:
            tenant_id: Owner tenant
            filters: Optional filters (search-specific)

        Returns:
            Total count
        """
        ...
