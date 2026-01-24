from __future__ import annotations

from typing import List, Optional

from providers.embeddings_provider import EmbeddingsProvider


class OpenAIEmbeddings(EmbeddingsProvider):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("openai package not installed") from exc

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        response = client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
