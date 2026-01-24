from __future__ import annotations

from typing import List, Optional

from providers.llm_base import LLMProvider
from rag.retriever import RetrievedChunk


def generate_answer(
    question: str,
    chunks: List[RetrievedChunk],
    provider: LLMProvider,
    max_context_words: Optional[int] = None,
) -> str:
    context = _build_context(chunks, max_context_words)
    prompt = f"Answer the question using the context.\n\nContext:\n{context}\n\nQ: {question}\nA:"
    return provider.generate(prompt)


def generate_answer_strict(
    question: str,
    chunks: List[RetrievedChunk],
    provider: LLMProvider,
    max_context_words: Optional[int] = None,
) -> str:
    context = _build_context(chunks, max_context_words)
    prompt = (
        "Answer using only the provided context. "
        "If the context is insufficient, say you do not have enough information.\n\n"
        f"Context:\n{context}\n\nQ: {question}\nA:"
    )
    return provider.generate(prompt)


def _build_context(chunks: List[RetrievedChunk], max_context_words: Optional[int]) -> str:
    if not max_context_words:
        return "\n\n".join(chunk.text for chunk in chunks)
    words_used = 0
    parts: List[str] = []
    for chunk in chunks:
        chunk_words = chunk.text.split()
        if words_used + len(chunk_words) > max_context_words:
            remaining = max_context_words - words_used
            if remaining <= 0:
                break
            parts.append(" ".join(chunk_words[:remaining]))
            words_used = max_context_words
            break
        parts.append(chunk.text)
        words_used += len(chunk_words)
    return "\n\n".join(parts)
