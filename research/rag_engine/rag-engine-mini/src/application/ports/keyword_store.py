"""
Keyword Store Port
===================
Interface for full-text search operations (BM25-style).

منفذ البحث النصي الكامل
"""

from typing import Protocol, Sequence

from src.domain.entities import Chunk, TenantId


class KeywordStorePort(Protocol):
    """
    Port for keyword-based full-text search.
    
    Implementation: Postgres FTS (tsvector + GIN index)
    
    Design Decision: Using Postgres FTS instead of Elasticsearch
    for simplicity. Good enough for most use cases and keeps the
    stack simpler.
    
    قرار التصميم: استخدام Postgres FTS بدلاً من Elasticsearch للبساطة
    """
    
    def search(
        self,
        *,
        query: str,
        tenant_id: TenantId,
        top_k: int,
        document_id: str | None = None,
    ) -> Sequence[Chunk]:
        """
        Search for chunks matching keywords.
        
        Args:
            query: Search query (natural language)
            tenant_id: Tenant for isolation
            top_k: Maximum results
            document_id: Optional filter for document-specific search
            
        Returns:
            Matching chunks with text (already hydrated)
            
        Note:
            Uses websearch_to_tsquery for natural query parsing.
            Results are ranked by ts_rank_cd.
            
            يستخدم websearch_to_tsquery لتحليل الاستعلام الطبيعي
        """
        ...
