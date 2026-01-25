from __future__ import annotations

from typing import Optional

from providers.llm_base import LLMProvider


class QueryRewriter:
    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self._provider = provider

    def rewrite(self, question: str) -> str:
        if not self._provider:
            return question
        prompt = (
            "Rewrite the question into a concise search query that preserves intent and key entities. "
            "Return only the rewritten query.\n"
            f"Question: {question}\n"
            "Query:"
        )
        rewritten = self._provider.generate(prompt).strip()
        return rewritten or question
