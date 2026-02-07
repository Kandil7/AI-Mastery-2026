"""
Document Idempotency Port
==========================
Interface for file hash-based idempotency.

منفذ تفرد المستندات
"""

from typing import Protocol

from src.domain.entities import DocumentId, StoredFile, TenantId


class DocumentIdempotencyPort(Protocol):
    """
    Port for document upload idempotency via file hash.
    
    Design Decision: Separate port following ISP (Interface Segregation).
    This is a specific concern that not all document repos need.
    
    Rationale:
    - Prevents re-indexing the same file content
    - Saves compute/cost on duplicate uploads
    - Uses SHA256 of file content as unique key per tenant
    
    قرار التصميم: منفذ منفصل يتبع مبدأ فصل الواجهات
    """
    
    def get_by_file_hash(
        self,
        *,
        tenant_id: TenantId,
        file_sha256: str,
    ) -> DocumentId | None:
        """
        Find existing document with same file hash.
        
        Args:
            tenant_id: Owner tenant
            file_sha256: SHA256 hash of file content
            
        Returns:
            Existing document ID if found, None otherwise
            
        Note:
            Hash is unique per tenant, not globally.
            Same file can exist for different tenants.
            
            التجزئة فريدة لكل مستأجر، وليست عالمياً
        """
        ...
    
    def create_document_with_hash(
        self,
        *,
        tenant_id: TenantId,
        stored_file: StoredFile,
        file_sha256: str,
    ) -> DocumentId:
        """
        Create document with file hash (concurrent-safe).
        
        Args:
            tenant_id: Owner tenant
            stored_file: File storage information
            file_sha256: SHA256 hash of file content
            
        Returns:
            Document ID (new or existing if concurrent insert)
            
        Note:
            Handles race conditions via unique constraint.
            If concurrent insert, returns existing document.
            
            يتعامل مع ظروف السباق عبر قيد الفرادة
        """
        ...
