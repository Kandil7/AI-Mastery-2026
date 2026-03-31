from __future__ import annotations

from typing import Optional

from providers.llm_base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("openai package not installed") from exc

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
