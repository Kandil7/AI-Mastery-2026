"""
Chat Repository Port
=====================
Interface for chat session and turn persistence.

منفذ مستودع المحادثات
"""

from typing import Protocol, Sequence

from src.domain.entities import ChatSession, ChatTurn, TenantId


class ChatRepoPort(Protocol):
    """
    Port for chat history persistence.
    
    Design Decision: Store chat history for:
    - User experience (continue conversations)
    - Observability (track latency, costs, retrieval quality)
    - Evaluation (golden Q&A for testing)
    
    قرار التصميم: تخزين تاريخ المحادثة للتجربة والمراقبة والتقييم
    """
    
    def create_session(
        self,
        *,
        tenant_id: TenantId,
        title: str | None = None,
    ) -> str:
        """
        Create a new chat session.
        
        Args:
            tenant_id: Owner tenant
            title: Optional session title
            
        Returns:
            New session ID
        """
        ...
    
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
        """
        Add a question-answer turn to a session.
        
        Args:
            tenant_id: Owner tenant
            session_id: Session to add to
            question: User question
            answer: Generated answer
            sources: Retrieved chunk IDs
            retrieval_k: Number of chunks retrieved
            embed_ms: Embedding latency (ms)
            search_ms: Search latency (ms)
            llm_ms: LLM latency (ms)
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count
            
        Returns:
            New turn ID
        """
        ...
    
    def get_session_turns(
        self,
        *,
        tenant_id: TenantId,
        session_id: str,
        limit: int = 50,
    ) -> Sequence[ChatTurn]:
        """
        Get turns for a session.
        
        Args:
            tenant_id: Owner tenant
            session_id: Session to query
            limit: Maximum turns to return
            
        Returns:
            List of turns in chronological order
        """
        ...
    
    def list_sessions(
        self,
        *,
        tenant_id: TenantId,
        limit: int = 50,
    ) -> Sequence[ChatSession]:
        """
        List chat sessions for a tenant.
        
        Args:
            tenant_id: Owner tenant
            limit: Maximum sessions to return
            
        Returns:
            List of sessions (most recent first)
        """
        ...
