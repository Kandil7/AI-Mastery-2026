from __future__ import annotations

from typing import List, Optional
import json

from providers.llm_base import LLMProvider
from rag.retriever import RetrievedChunk


class Reranker:
    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self._provider = provider

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not self._provider or not chunks:
            return chunks

        prompt = (
            "You are a search reranker. Return a JSON object with an 'order' list of indices "
            "ranked from most to least relevant. Only return JSON.\n"
        )
        prompt += f"Question: {query}\n"
        for idx, chunk in enumerate(chunks):
            prompt += f"[{idx}] {_compress_text(chunk.text)}\n"
        response = self._provider.generate(prompt)
        order = _parse_order(response, len(chunks))
        if not order:
            return chunks
        ranked = [chunks[i] for i in order if 0 <= i < len(chunks)]
        if top_k is not None:
            return ranked[:top_k]
        return ranked


def _parse_order(response: str, count: int) -> List[int]:
    response = response.strip()
    if response.startswith("{"):
        try:
            payload = json.loads(response)
            order = payload.get("order", [])
            if isinstance(order, list):
                cleaned = [int(idx) for idx in order if isinstance(idx, int) or str(idx).isdigit()]
                return _unique_in_range(cleaned, count)
        except json.JSONDecodeError:
            pass
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


def _compress_text(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."
