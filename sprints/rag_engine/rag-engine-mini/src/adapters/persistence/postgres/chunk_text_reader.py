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
        Get texts for multiple chunk IDs.
        
        Uses ANY() for efficient batch lookup.
        """
        if not chunk_ids:
            return {}
        
        with SessionLocal() as db:
            rows = db.execute(
                sql_text("""
                    SELECT id, text
                    FROM chunk_store
                    WHERE user_id = :user_id
                      AND id = ANY(:ids)
                """),
                {
                    "user_id": tenant_id.value,
                    "ids": list(chunk_ids),
                },
            ).all()
            
            return {cid: txt for cid, txt in rows}
