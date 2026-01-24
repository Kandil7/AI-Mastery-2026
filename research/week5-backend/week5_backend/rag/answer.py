from __future__ import annotations

from typing import List

from providers.llm_base import LLMProvider
from rag.retriever import RetrievedChunk


def generate_answer(question: str, chunks: List[RetrievedChunk], provider: LLMProvider) -> str:
    context = "\n\n".join(chunk.text for chunk in chunks)
    prompt = f"Answer the question using the context.\n\nContext:\n{context}\n\nQ: {question}\nA:"
    return provider.generate(prompt)
