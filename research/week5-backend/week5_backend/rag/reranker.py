from __future__ import annotations

from typing import List, Optional

from providers.llm_base import LLMProvider
from rag.retriever import RetrievedChunk


class Reranker:
    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self._provider = provider

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not self._provider or not chunks:
            return chunks

        prompt = "Rank the following passages for relevance to the question.\n"
        prompt += f"Question: {query}\n"
        for idx, chunk in enumerate(chunks):
            prompt += f"[{idx}] {chunk.text}\n"
        response = self._provider.generate(prompt)
        # Placeholder: keep original order; implement parsing in production.
        _ = response
        return chunks
