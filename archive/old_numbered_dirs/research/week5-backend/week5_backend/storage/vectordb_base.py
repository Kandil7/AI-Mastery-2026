from __future__ import annotations

from typing import Any, Dict, List, Protocol

from rag.chunking import Chunk
from rag.retriever import RetrievedChunk


class VectorStore(Protocol):
    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        ...

    def query_by_vector(
        self,
        vector: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[RetrievedChunk]:
        ...
