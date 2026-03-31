"""
RAG Prompt Builder
===================
Pure service for constructing prompts with guardrails.

خدمة بناء الأوامر مع الحواجز الأمنية
"""

from typing import Sequence

from src.domain.entities import Chunk


# System prompt template with guardrails
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only answer based on the information in the context below.
2. If the answer is not in the context, say "I don't have enough information to answer this question."
3. Be concise and accurate.
4. Cite the relevant parts of the context when possible.
5. Do not make up information that is not in the context.

Context:
{context}
"""

USER_PROMPT = """Question: {question}

Based on the context provided above, please answer this question."""


def build_rag_prompt(
    *,
    question: str,
    chunks: Sequence[Chunk],
    max_context_chars: int = 12000,
) -> str:
    """
    Build a RAG prompt from question and retrieved chunks.
    
    Args:
        question: User's question
        chunks: Retrieved chunks (already reranked)
        max_context_chars: Maximum context length (for budget control)
    
    Returns:
        Complete prompt for LLM
    
    Design Decision: Guardrails in system prompt to reduce hallucination:
    - Explicit instruction to only use context
    - Explicit fallback for unknown answers
    - Encouragement to cite sources
    
    قرار التصميم: حواجز أمنية لتقليل الهلوسة
    
    Example:
        >>> prompt = build_rag_prompt(
        ...     question="What is the main topic?",
        ...     chunks=[chunk1, chunk2]
        ... )
    """
    # Build context from chunks
    context_parts = []
    total_chars = 0
    
    for i, chunk in enumerate(chunks, start=1):
        # Format each chunk with source reference
        chunk_text = f"[Source {i}] {chunk.text}"
        
        # Check budget
        if total_chars + len(chunk_text) > max_context_chars:
            break
        
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)
    
    # Join context
    context = "\n\n".join(context_parts)
    
    if not context:
        context = "No relevant context found."
    
    # Build full prompt
    system = SYSTEM_PROMPT.format(context=context)
    user = USER_PROMPT.format(question=question)
    
    # Combine into single prompt (for models that don't support roles)
    # For chat models, you might want to return a list of messages instead
    return f"{system}\n\n{user}"


def build_chat_messages(
    *,
    question: str,
    chunks: Sequence[Chunk],
    max_context_chars: int = 12000,
) -> list[dict[str, str]]:
    """
    Build chat messages for models that support message roles.
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    
    بناء رسائل المحادثة للنماذج التي تدعم أدوار الرسائل
    """
    # Build context
    context_parts = []
    total_chars = 0
    
    for i, chunk in enumerate(chunks, start=1):
        chunk_text = f"[Source {i}] {chunk.text}"
        if total_chars + len(chunk_text) > max_context_chars:
            break
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    
    system_content = f"""You are a helpful assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only answer based on the information in the context below.
2. If the answer is not in the context, say "I don't have enough information to answer this question."
3. Be concise and accurate.
4. Cite the relevant parts of the context when possible.
5. Do not make up information that is not in the context.

Context:
{context}"""
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]
