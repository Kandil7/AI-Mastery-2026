"""
Chunk Text Reader Implementation
=================================
PostgreSQL implementation for hydrating chunk texts.

تنفيذ قراءة نصوص القطع من PostgreSQL
"""

from typing import Sequence

from sqlalchemy import text as sql_text

from src.adapters.persistence.postgres.db import SessionLocal
from src.domain.entities import TenantId


class PostgresChunkTextReader:
    """
    PostgreSQL implementation of ChunkTextReaderPort.
    
    Used to hydrate chunk texts after vector search.
    
    تنفيذ قراءة نصوص القطع من PostgreSQL
    """
    
    def get_texts_by_ids(
        self,
        *,
        tenant_id: TenantId,
        chunk_ids: Sequence[str],
    ) -> dict[str, str]:
        """
        Get texts for multiple chunk IDs, fetching parents and adding context.
        """
        if not chunk_ids:
            return {}
        
        with SessionLocal() as db:
            # Stage 2: Join with parent if exists and fetch doc summary context
            query = sql_text("""
                SELECT 
                    c.id, 
                    COALESCE(p.text, c.text) as content,
                    c.chunk_context
                FROM chunk_store c
                LEFT JOIN chunk_store p ON c.parent_id = p.id
                WHERE c.user_id = :user_id
                  AND c.id = ANY(:ids)
            """)
            
            rows = db.execute(
                query,
                {
                    "user_id": tenant_id.value,
                    "ids": list(chunk_ids),
                },
            ).all()
            
            results = {}
            for cid, content, ctx in rows:
                # Prepend document context (Contextual Retrieval)
                final_text = f"[Context: {ctx}]\n\n{content}" if ctx else content
                results[cid] = final_text
                
            return results
