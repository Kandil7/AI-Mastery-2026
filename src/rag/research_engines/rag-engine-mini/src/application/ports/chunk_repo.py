"""
Chunk Repository Port
======================
Interface for chunk storage with deduplication.

منفذ مستودع القطع مع إزالة التكرار
"""

from typing import Protocol, Sequence

from src.domain.entities import TenantId


class ChunkRepoPort(Protocol):
    """
    Port for chunk storage with per-tenant deduplication.
    
    Design Decision: Two-table design (chunk_store + document_chunks)
    for proper deduplication across documents.
    
    قرار التصميم: تصميم جدولين لإزالة التكرار بين المستندات
    
    Tables:
    - chunk_store: unique chunks per tenant (id, user_id, chunk_hash, text, tsv)
    - document_chunks: mapping of documents to chunk order (document_id, ord, chunk_id)
    """
    
    def upsert_chunk_store(
        self,
        *,
        tenant_id: TenantId,
        chunk_hash: str,
        text: str,
        parent_id: str | None = None,
        chunk_context: str | None = None,
    ) -> str:
        """
        Upsert a chunk to the store (deduplicated).
        
        Args:
            tenant_id: Owner tenant
            chunk_hash: SHA256 of normalized text
            text: Chunk text content
            
        Returns:
            Chunk ID (new or existing)
            
        Note:
            If chunk with same hash exists for tenant, returns existing ID.
            This enables deduplication across documents.
            
            إذا وُجدت قطعة بنفس التجزئة، يُرجع المعرف الموجود
        """
        ...
    
    def replace_document_chunks(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
        chunk_ids_in_order: Sequence[str],
    ) -> None:
        """
        Replace the chunk mapping for a document.
        
        Args:
            tenant_id: Owner tenant (for validation)
            document_id: Document to update
            chunk_ids_in_order: Ordered list of chunk IDs
            
        Note:
            Deletes existing mappings and creates new ones.
            Preserves order via 'ord' column.
            
            يحذف التعيينات الموجودة وينشئ جديدة
        """
        ...
    
    def get_document_chunk_ids(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> Sequence[str]:
        """
        Get ordered chunk IDs for a document.
        
        Args:
            tenant_id: Owner tenant
            document_id: Document to query
            
        Returns:
            Ordered list of chunk IDs
        """
        ...
    
    def delete_document_chunks(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> int:
        """
        Delete chunk mappings for a document.
        
        Args:
            tenant_id: Owner tenant
            document_id: Document to clean up
            
        Returns:
            Number of mappings deleted
            
        Note:
            Does NOT delete from chunk_store (chunks may be shared).
            Orphan cleanup can be done separately if needed.
            
            لا يحذف من chunk_store (القطع قد تكون مشتركة)
        """
        ...
