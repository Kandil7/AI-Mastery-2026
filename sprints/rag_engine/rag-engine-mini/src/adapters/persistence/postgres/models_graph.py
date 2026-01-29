"""
Graph RAG ORM Models
====================
SQLAlchemy models for simple knowledge graph storage.

نماذج ORM لتخزين الرسم البياني للمعرفة
"""

from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, func, Index, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from src.adapters.persistence.postgres.db import Base

class GraphTripletRow(Base):
    """
    Storage for extracted knowledge triplets (S, R, O).
    
    تخزين الثلاثيات المعرفية المستخرجة
    """
    __tablename__ = "graph_triplets"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chunk_store.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # The triplet
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    relation: Mapped[str] = mapped_column(String(256), nullable=False)
    obj: Mapped[str] = mapped_column(String(256), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

Index("ix_graph_user_id", GraphTripletRow.user_id)
Index("ix_graph_subject", GraphTripletRow.subject)
Index("ix_graph_object", GraphTripletRow.obj)
Index("ix_graph_document", GraphTripletRow.document_id)
