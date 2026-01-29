"""
Chat Repository Implementation
===============================
PostgreSQL implementation of ChatRepoPort.

تنفيذ مستودع المحادثات بـ PostgreSQL
"""

import uuid
from typing import Sequence

from sqlalchemy import select

from src.adapters.persistence.postgres.db import SessionLocal
from src.adapters.persistence.postgres.models_chat import ChatSessionRow, ChatTurnRow
from src.domain.entities import ChatSession, ChatTurn, TenantId


class PostgresChatRepo:
    """
    PostgreSQL implementation of ChatRepoPort.
    
    تنفيذ PostgreSQL لمنفذ مستودع المحادثات
    """
    
    def create_session(
        self,
        *,
        tenant_id: TenantId,
        title: str | None = None,
    ) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        
        with SessionLocal() as db:
            db.add(
                ChatSessionRow(
                    id=session_id,
                    user_id=tenant_id.value,
                    title=title,
                )
            )
            db.commit()
        
        return session_id
    
    def add_turn(
        self,
        *,
        tenant_id: TenantId,
        session_id: str,
        question: str,
        answer: str,
        sources: Sequence[str],
        retrieval_k: int,
        embed_ms: int | None = None,
        search_ms: int | None = None,
        llm_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> str:
        """Add a question-answer turn to a session."""
        turn_id = str(uuid.uuid4())
        
        with SessionLocal() as db:
            db.add(
                ChatTurnRow(
                    id=turn_id,
                    session_id=session_id,
                    user_id=tenant_id.value,
                    question=question,
                    answer=answer,
                    sources=list(sources),
                    retrieval_k=retrieval_k,
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    llm_ms=llm_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
            db.commit()
        
        return turn_id
    
    def get_session_turns(
        self,
        *,
        tenant_id: TenantId,
        session_id: str,
        limit: int = 50,
    ) -> Sequence[ChatTurn]:
        """Get turns for a session."""
        with SessionLocal() as db:
            stmt = (
                select(ChatTurnRow)
                .where(
                    ChatTurnRow.session_id == session_id,
                    ChatTurnRow.user_id == tenant_id.value,
                )
                .order_by(ChatTurnRow.created_at.asc())
                .limit(limit)
            )
            rows = db.execute(stmt).scalars().all()
            
            return [
                ChatTurn(
                    id=row.id,
                    session_id=row.session_id,
                    tenant_id=TenantId(row.user_id),
                    question=row.question,
                    answer=row.answer,
                    sources=row.sources,
                    retrieval_k=row.retrieval_k,
                    embed_ms=row.embed_ms,
                    search_ms=row.search_ms,
                    llm_ms=row.llm_ms,
                    prompt_tokens=row.prompt_tokens,
                    completion_tokens=row.completion_tokens,
                    created_at=row.created_at,
                )
                for row in rows
            ]
    
    def list_sessions(
        self,
        *,
        tenant_id: TenantId,
        limit: int = 50,
    ) -> Sequence[ChatSession]:
        """List chat sessions for a tenant."""
        with SessionLocal() as db:
            stmt = (
                select(ChatSessionRow)
                .where(ChatSessionRow.user_id == tenant_id.value)
                .order_by(ChatSessionRow.created_at.desc())
                .limit(limit)
            )
            rows = db.execute(stmt).scalars().all()
            
            return [
                ChatSession(
                    id=row.id,
                    tenant_id=TenantId(row.user_id),
                    title=row.title,
                    created_at=row.created_at,
                )
                for row in rows
            ]
