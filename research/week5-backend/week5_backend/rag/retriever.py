from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rag.bm25 import BM25Index
from rag.embeddings import EmbeddingService
from storage.vectordb_base import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingService,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._bm25 = bm25_index

    def retrieve(self, query: str, top_k: int, filters: Dict[str, Any]) -> List[RetrievedChunk]:
        vector = self._embedder.embed([query])[0]
        vector_hits = self._vector_store.query_by_vector(vector=vector, top_k=top_k, filters=filters)
        if not self._bm25:
            return vector_hits

        bm25_hits = self._bm25.query(query, top_k=top_k)
        merged: Dict[str, RetrievedChunk] = {hit.chunk_id: hit for hit in vector_hits}
        for bm in bm25_hits:
            if bm.chunk_id in merged:
                existing = merged[bm.chunk_id]
                merged[bm.chunk_id] = RetrievedChunk(
                    chunk_id=existing.chunk_id,
                    doc_id=existing.doc_id,
                    text=existing.text,
                    score=existing.score + bm.score,
                    metadata=existing.metadata,
                )
            else:
                merged[bm.chunk_id] = RetrievedChunk(
                    chunk_id=bm.chunk_id,
                    doc_id=bm.doc_id,
                    text=bm.text,
                    score=bm.score,
                    metadata={},
                )
        return sorted(merged.values(), key=lambda hit: hit.score, reverse=True)[:top_k]
