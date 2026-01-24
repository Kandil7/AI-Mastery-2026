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
        order = _parse_order(response, len(chunks))
        if not order:
            return chunks
        return [chunks[i] for i in order if 0 <= i < len(chunks)]


def _parse_order(response: str, count: int) -> List[int]:
    response = response.strip()
    digits = [int(tok) for tok in response.replace(",", " ").split() if tok.isdigit()]
    if digits:
        return _unique_in_range(digits, count)
    return []


def _unique_in_range(indices: List[int], count: int) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for idx in indices:
        if 0 <= idx < count and idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered
