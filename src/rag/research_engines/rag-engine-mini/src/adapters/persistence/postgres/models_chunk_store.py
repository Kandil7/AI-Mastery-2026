"""
Chunk Store ORM Models
=======================
SQLAlchemy models for chunk storage with deduplication.

نماذج ORM لتخزين القطع مع إزالة التكرار
"""

from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    func,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TSVECTOR

from src.adapters.persistence.postgres.db import Base


class ChunkStoreRow(Base):
    """
    Deduplicated chunk storage.
    
    Each unique chunk (per tenant) is stored once.
    Documents reference chunks via document_chunks mapping.
    
    Design Decision: Two-table design for proper deduplication:
    - chunk_store: unique chunks with text and tsvector
    - document_chunks: mapping of document → chunk order
    
    تخزين قطع مزالة التكرار
    """
    __tablename__ = "chunk_store"
    
    __table_args__ = (
        UniqueConstraint("user_id", "chunk_hash", name="uq_chunk_store_user_hash"),
    )
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Chunk content
    chunk_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Hierarchical / Contextual retrieval (Stage 2)
    parent_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("chunk_store.id", ondelete="SET NULL"),
        nullable=True,
    )
    chunk_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Full-text search vector (GENERATED in migration)
    # Note: This column is GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
    tsv: Mapped[object] = mapped_column(TSVECTOR, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


# Indexes
Index("ix_chunk_store_user_id", ChunkStoreRow.user_id)
Index("ix_chunk_store_tsv", ChunkStoreRow.tsv, postgresql_using="gin")


class DocumentChunkRow(Base):
    """
    Mapping of documents to chunks with ordering.
    
    Allows:
    - Same chunk to be used by multiple documents
    - Preserving chunk order within document
    
    ربط المستندات بالقطع مع الترتيب
    """
    __tablename__ = "document_chunks"
    
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    ord: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chunk_store.id", ondelete="CASCADE"),
        nullable=False,
    )


# Indexes
Index("ix_document_chunks_document_id", DocumentChunkRow.document_id)
Index("ix_document_chunks_chunk_id", DocumentChunkRow.chunk_id)
