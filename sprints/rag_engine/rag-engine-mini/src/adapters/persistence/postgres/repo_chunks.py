"""
Chunk Dedup Repository Implementation
======================================
PostgreSQL implementation for chunk storage with deduplication.

تنفيذ مستودع القطع مع إزالة التكرار
"""

import uuid
from typing import Sequence

from sqlalchemy import select, delete, insert
from sqlalchemy.exc import IntegrityError

from src.adapters.persistence.postgres.db import SessionLocal
from src.adapters.persistence.postgres.models_chunk_store import (
    ChunkStoreRow,
    DocumentChunkRow,
)
from src.domain.entities import TenantId


class PostgresChunkDedupRepo:
    """
    PostgreSQL implementation for chunk deduplication.
    
    Two-table design:
    - chunk_store: unique chunks per tenant (deduped by hash)
    - document_chunks: mapping of document → ordered chunks
    
    تنفيذ PostgreSQL لإزالة تكرار القطع
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
        Upsert a chunk to the store (deduplicated by hash).
        """
        with SessionLocal() as db:
            # Check if exists
            stmt = select(ChunkStoreRow.id).where(
                ChunkStoreRow.user_id == tenant_id.value,
                ChunkStoreRow.chunk_hash == chunk_hash,
            )
            existing = db.execute(stmt).scalar_one_or_none()
            
            if existing:
                # Optional: Update parent/context if they changed?
                # For simplicity, we assume same hash = same chunk.
                return existing
            
            # Create new
            chunk_id = str(uuid.uuid4())
            db.add(
                ChunkStoreRow(
                    id=chunk_id,
                    user_id=tenant_id.value,
                    chunk_hash=chunk_hash,
                    text=text,
                    parent_id=parent_id,
                    chunk_context=chunk_context,
                )
            )
            
            try:
                db.commit()
                return chunk_id
            except IntegrityError:
                # Concurrent insert - fetch existing
                db.rollback()
                existing = db.execute(stmt).scalar_one()
                return existing
    
    def replace_document_chunks(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
        chunk_ids_in_order: Sequence[str],
    ) -> None:
        """
        Replace the chunk mapping for a document.
        
        Deletes existing mappings and creates new ones.
        """
        with SessionLocal() as db:
            # Delete existing mappings
            db.execute(
                delete(DocumentChunkRow).where(
                    DocumentChunkRow.document_id == document_id
                )
            )
            
            # Insert new mappings
            if chunk_ids_in_order:
                rows = [
                    {"document_id": document_id, "ord": i, "chunk_id": cid}
                    for i, cid in enumerate(chunk_ids_in_order)
                ]
                db.execute(insert(DocumentChunkRow), rows)
            
            db.commit()
    
    def get_document_chunk_ids(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> Sequence[str]:
        """Get ordered chunk IDs for a document."""
        with SessionLocal() as db:
            stmt = (
                select(DocumentChunkRow.chunk_id)
                .where(DocumentChunkRow.document_id == document_id)
                .order_by(DocumentChunkRow.ord)
            )
            return list(db.execute(stmt).scalars().all())
    
    def delete_document_chunks(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> int:
        """Delete chunk mappings for a document."""
        with SessionLocal() as db:
            result = db.execute(
                delete(DocumentChunkRow).where(
                    DocumentChunkRow.document_id == document_id
                )
            )
            db.commit()
            return result.rowcount
