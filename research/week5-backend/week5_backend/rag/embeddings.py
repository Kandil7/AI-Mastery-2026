from __future__ import annotations

from typing import List

from providers.embeddings_provider import EmbeddingsProvider


class EmbeddingService:
    def __init__(self, provider: EmbeddingsProvider) -> None:
        self._provider = provider

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._provider.embed(texts)
