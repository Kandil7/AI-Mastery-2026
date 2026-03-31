"""
Chunk Text Reader Port
=======================
Interface for hydrating chunk texts from database.

منفذ قراءة نصوص القطع
"""

from typing import Protocol, Sequence

from src.domain.entities import TenantId


class ChunkTextReaderPort(Protocol):
    """
    Port for reading chunk texts by ID.
    
    Design Decision: Needed because vector store uses minimal payload
    (no text stored in Qdrant). Text is hydrated from Postgres.
    
    قرار التصميم: مطلوب لأن مخزن المتجهات لا يخزن النص
    """
    
    def get_texts_by_ids(
        self,
        *,
        tenant_id: TenantId,
        chunk_ids: Sequence[str],
    ) -> dict[str, str]:
        """
        Get texts for multiple chunk IDs.
        
        Args:
            tenant_id: Owner tenant (for isolation)
            chunk_ids: List of chunk IDs to fetch
            
        Returns:
            Dictionary mapping chunk_id -> text
            
        Note:
            Missing IDs are simply not in the result dict.
            Caller should handle missing chunks gracefully.
            
            المعرفات المفقودة ببساطة لا تكون في النتيجة
        """
        ...
