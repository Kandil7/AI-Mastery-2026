from __future__ import annotations

from typing import List

from providers.llm_base import LLMProvider


class Verifier:
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._provider = provider

    def verify(self, answer: str, citations: List[dict]) -> bool:
        if not self._provider:
            return True
        prompt = (
            "Check whether the answer is fully supported by the citations.\n"
            "Reply with YES or NO.\n"
            f"Answer: {answer}\n"
            f"Citations: {citations}\n"
        )
        response = self._provider.generate(prompt).strip().lower()
        return response.startswith("yes")
