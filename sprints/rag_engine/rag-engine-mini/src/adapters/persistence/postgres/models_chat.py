"""
Chat Session ORM Models
========================
SQLAlchemy models for chat history persistence.

نماذج ORM لتاريخ المحادثات
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
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY

from src.adapters.persistence.postgres.db import Base


class ChatSessionRow(Base):
    """
    Chat session model.

    A session groups multiple Q&A turns together.

    نموذج جلسة المحادثة
    """

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    topics: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)), nullable=True, default_factory=list
    )
    sentiment: Mapped[str | None] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    turns = relationship(
        "ChatTurnRow",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatTurnRow.created_at",
    )


Index("ix_chat_sessions_user_id", ChatSessionRow.user_id)


class ChatTurnRow(Base):
    """
    Chat turn model.

    A single question-answer exchange with observability fields.

    نموذج دورة المحادثة
    """

    __tablename__ = "chat_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Q&A content
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[str]] = mapped_column(ARRAY(String(36)), nullable=False)

    # Retrieval info
    retrieval_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Observability / Performance tracking
    embed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    session = relationship("ChatSessionRow", back_populates="turns")


Index("ix_chat_turns_session_id", ChatTurnRow.session_id)
Index("ix_chat_turns_user_id", ChatTurnRow.user_id)
