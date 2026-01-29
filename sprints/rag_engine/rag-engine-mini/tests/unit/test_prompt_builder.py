"""
Prompt Builder Unit Tests
==========================
Tests for RAG prompt construction.
"""

import pytest

from src.application.services.prompt_builder import (
    build_rag_prompt,
    build_chat_messages,
)
from src.domain.entities import Chunk, TenantId, DocumentId


def make_chunk(text: str, chunk_id: str = "chunk_1") -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=chunk_id,
        tenant_id=TenantId("test_tenant"),
        document_id=DocumentId("test_doc"),
        text=text,
    )


class TestBuildRagPrompt:
    """Tests for build_rag_prompt function."""
    
    def test_empty_chunks(self):
        """Empty chunks should produce 'no context' message."""
        prompt = build_rag_prompt(question="What is AI?", chunks=[])
        assert "No relevant context found" in prompt
        assert "What is AI?" in prompt
    
    def test_single_chunk(self):
        """Single chunk should be included in context."""
        chunk = make_chunk("AI stands for Artificial Intelligence.")
        prompt = build_rag_prompt(question="What is AI?", chunks=[chunk])
        
        assert "AI stands for Artificial Intelligence" in prompt
        assert "[Source 1]" in prompt
        assert "What is AI?" in prompt
    
    def test_multiple_chunks(self):
        """Multiple chunks should be numbered."""
        chunks = [
            make_chunk("First piece of information.", "c1"),
            make_chunk("Second piece of information.", "c2"),
        ]
        prompt = build_rag_prompt(question="What?", chunks=chunks)
        
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt
        assert "First piece" in prompt
        assert "Second piece" in prompt
    
    def test_context_budget_respected(self):
        """Context should be truncated at max_context_chars."""
        # Create chunks that exceed budget
        long_text = "x" * 5000
        chunks = [
            make_chunk(long_text, "c1"),
            make_chunk("Should not appear", "c2"),
        ]
        
        prompt = build_rag_prompt(
            question="What?",
            chunks=chunks,
            max_context_chars=4000,
        )
        
        # First chunk should be included
        assert "[Source 1]" in prompt
        # Second chunk should NOT be included (budget exceeded)
        assert "Should not appear" not in prompt
    
    def test_guardrails_present(self):
        """Guardrail instructions should be in prompt."""
        chunk = make_chunk("Some context")
        prompt = build_rag_prompt(question="What?", chunks=[chunk])
        
        assert "Only answer based on" in prompt
        assert "do not make up" in prompt.lower() or "don't have enough" in prompt


class TestBuildChatMessages:
    """Tests for build_chat_messages function."""
    
    def test_returns_list_of_dicts(self):
        """Should return list of message dicts."""
        chunk = make_chunk("Context text")
        messages = build_chat_messages(question="What?", chunks=[chunk])
        
        assert isinstance(messages, list)
        assert len(messages) == 2  # system + user
        assert all("role" in m and "content" in m for m in messages)
    
    def test_user_message_contains_question(self):
        """User message should contain the question."""
        messages = build_chat_messages(question="What is RAG?", chunks=[])
        
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert "What is RAG?" in user_msg["content"]
    
    def test_system_message_contains_context(self):
        """System message should contain the context."""
        chunk = make_chunk("RAG stands for Retrieval-Augmented Generation.")
        messages = build_chat_messages(question="What?", chunks=[chunk])
        
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert "RAG stands for" in system_msg["content"]
