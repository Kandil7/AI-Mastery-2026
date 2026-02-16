"""
User and Document ORM Models
=============================
SQLAlchemy models for users and documents tables.

نماذج ORM للمستخدمين والمستندات
"""

from datetime import datetime

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, func, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.adapters.persistence.postgres.db import Base


class User(Base):
    """
    User model for multi-tenant isolation.
    
    Each user has:
    - Unique API key for authentication
    - Documents owned by this user
    
    نموذج المستخدم للعزل متعدد المستأجرين
    """
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    api_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    
    # Relationships
    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
    )


# Indexes
Index("ix_users_api_key", User.api_key)
Index("ix_users_email", User.email)


class Document(Base):
    """
    Document model for uploaded files.
    
    Tracks:
    - File metadata (path, type, size)
    - Processing status (queued, indexed, failed)
    - File hash for idempotency
    
    نموذج المستند للملفات المرفوعة
    """
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # File metadata
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Idempotency hash
    file_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    
    # Processing status
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="created",
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    
    # Relationships
    user = relationship("User", back_populates="documents")


# Indexes
Index("ix_documents_user_id", Document.user_id)
Index("ix_documents_status", Document.status)
Index(
    "uq_documents_user_file_sha256",
    Document.user_id,
    Document.file_sha256,
    unique=True,
)
