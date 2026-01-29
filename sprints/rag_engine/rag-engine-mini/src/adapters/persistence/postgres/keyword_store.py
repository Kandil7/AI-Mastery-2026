"""
Keyword Store Implementation
=============================
PostgreSQL full-text search using tsvector + GIN index.

تنفيذ البحث النصي الكامل بـ PostgreSQL
"""

from typing import Sequence

from sqlalchemy import text as sql_text

from src.adapters.persistence.postgres.db import SessionLocal
from src.domain.entities import Chunk, TenantId, DocumentId


class PostgresKeywordStore:
    """
    PostgreSQL FTS implementation of KeywordStorePort.
    
    Uses:
    - websearch_to_tsquery for natural query parsing
    - ts_rank_cd for relevance ranking
    - GIN index on tsvector for fast search
    
    تنفيذ البحث النصي الكامل بـ PostgreSQL
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
        
        Uses websearch_to_tsquery for natural language queries.
        Results are ranked by ts_rank_cd.
        """
        with SessionLocal() as db:
            if document_id:
                # Document-filtered search (ChatPDF mode)
                rows = db.execute(
                    sql_text("""
                        SELECT cs.id, cs.text
                        FROM document_chunks dc
                        JOIN chunk_store cs ON cs.id = dc.chunk_id
                        WHERE dc.document_id = :doc_id
                          AND cs.user_id = :user_id
                          AND cs.tsv @@ websearch_to_tsquery('simple', :q)
                        ORDER BY ts_rank_cd(cs.tsv, websearch_to_tsquery('simple', :q)) DESC,
                                 dc.ord ASC
                        LIMIT :k
                    """),
                    {
                        "doc_id": document_id,
                        "user_id": tenant_id.value,
                        "q": query,
                        "k": top_k,
                    },
                ).all()
                
                return [
                    Chunk(
                        id=cid,
                        tenant_id=tenant_id,
                        document_id=DocumentId(document_id),
                        text=txt,
                    )
                    for (cid, txt) in rows
                ]
            
            # Tenant-wide search
            rows = db.execute(
                sql_text("""
                    SELECT cs.id, cs.text
                    FROM chunk_store cs
                    WHERE cs.user_id = :user_id
                      AND cs.tsv @@ websearch_to_tsquery('simple', :q)
                    ORDER BY ts_rank_cd(cs.tsv, websearch_to_tsquery('simple', :q)) DESC
                    LIMIT :k
                """),
                {
                    "user_id": tenant_id.value,
                    "q": query,
                    "k": top_k,
                },
            ).all()
            
            return [
                Chunk(
                    id=cid,
                    tenant_id=tenant_id,
                    document_id=DocumentId(""),  # Not available in tenant-wide search
                    text=txt,
                )
                for (cid, txt) in rows
            ]
