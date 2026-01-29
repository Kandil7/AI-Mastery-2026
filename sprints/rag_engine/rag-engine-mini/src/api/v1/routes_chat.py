"""
Chat Routes
=============
Endpoints for chat session management.

نقاط نهاية إدارة جلسات المحادثة
"""

from typing import Sequence

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.v1.deps import get_tenant_id
from src.core.bootstrap import get_container
from src.domain.entities import TenantId

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class CreateSessionRequest(BaseModel):
    """Request to create a chat session."""
    title: str | None = Field(default=None, max_length=256)


class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str


class SessionInfo(BaseModel):
    """Chat session information."""
    session_id: str
    title: str | None
    created_at: str


class TurnInfo(BaseModel):
    """Chat turn information."""
    turn_id: str
    question: str
    answer: str
    sources: list[str]
    created_at: str


class ListSessionsResponse(BaseModel):
    """Response for listing sessions."""
    sessions: list[SessionInfo]


class GetSessionResponse(BaseModel):
    """Response for getting session with turns."""
    session_id: str
    title: str | None
    turns: list[TurnInfo]


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    body: CreateSessionRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> CreateSessionResponse:
    """
    Create a new chat session.
    
    إنشاء جلسة محادثة جديدة
    """
    container = get_container()
    
    # Check if chat_repo is available
    if "chat_repo" not in container:
        # Return placeholder for now
        import uuid
        return CreateSessionResponse(session_id=str(uuid.uuid4()))
    
    chat_repo = container["chat_repo"]
    
    session_id = chat_repo.create_session(
        tenant_id=TenantId(tenant_id),
        title=body.title,
    )
    
    return CreateSessionResponse(session_id=session_id)


@router.get("/sessions", response_model=ListSessionsResponse)
def list_sessions(
    tenant_id: str = Depends(get_tenant_id),
    limit: int = 50,
) -> ListSessionsResponse:
    """
    List chat sessions for the authenticated user.
    
    قائمة جلسات المحادثة للمستخدم
    """
    container = get_container()
    
    if "chat_repo" not in container:
        return ListSessionsResponse(sessions=[])
    
    chat_repo = container["chat_repo"]
    
    sessions = chat_repo.list_sessions(
        tenant_id=TenantId(tenant_id),
        limit=limit,
    )
    
    return ListSessionsResponse(
        sessions=[
            SessionInfo(
                session_id=s.id,
                title=s.title,
                created_at=s.created_at.isoformat() if s.created_at else "",
            )
            for s in sessions
        ]
    )


@router.get("/sessions/{session_id}", response_model=GetSessionResponse)
def get_session(
    session_id: str,
    tenant_id: str = Depends(get_tenant_id),
    limit: int = 50,
) -> GetSessionResponse:
    """
    Get a chat session with its turns.
    
    الحصول على جلسة محادثة مع دوراتها
    """
    container = get_container()
    
    if "chat_repo" not in container:
        return GetSessionResponse(
            session_id=session_id,
            title=None,
            turns=[],
        )
    
    chat_repo = container["chat_repo"]
    
    turns = chat_repo.get_session_turns(
        tenant_id=TenantId(tenant_id),
        session_id=session_id,
        limit=limit,
    )
    
    return GetSessionResponse(
        session_id=session_id,
        title=None,  # TODO: Get from session
        turns=[
            TurnInfo(
                turn_id=t.id,
                question=t.question,
                answer=t.answer,
                sources=list(t.sources),
                created_at=t.created_at.isoformat() if t.created_at else "",
            )
            for t in turns
        ],
    )
