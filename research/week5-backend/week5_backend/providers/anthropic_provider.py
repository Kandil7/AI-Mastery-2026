from __future__ import annotations

from typing import Optional

from providers.llm_base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("anthropic package not installed") from exc

        client = Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=512,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""
