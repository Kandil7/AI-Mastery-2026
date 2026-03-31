from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from rag.bm25 import BM25Index, BM25Result
from rag.embeddings import EmbeddingService
from storage.vectordb_base import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class FusionConfig:
    use_rrf: bool = True
    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingService,
        bm25_index: BM25Index | None = None,
        fusion: FusionConfig | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._bm25 = bm25_index
        self._fusion = fusion or FusionConfig()

    def retrieve(self, query: str, top_k: int, filters: Dict[str, Any]) -> List[RetrievedChunk]:
        vector = self._embedder.embed([query])[0]
        vector_hits = self._vector_store.query_by_vector(vector=vector, top_k=top_k, filters=filters)
        if not self._bm25:
            return vector_hits

        bm25_hits = self._bm25.query(query, top_k=top_k)
        if self._fusion.use_rrf:
            fused = _rrf_fuse(
                vector_hits,
                bm25_hits,
                rrf_k=self._fusion.rrf_k,
                vector_weight=self._fusion.vector_weight,
                bm25_weight=self._fusion.bm25_weight,
            )
            return fused[:top_k]

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


def _rrf_fuse(
    vector_hits: List[RetrievedChunk],
    bm25_hits: List[BM25Result],
    rrf_k: int,
    vector_weight: float,
    bm25_weight: float,
) -> List[RetrievedChunk]:
    fused: Dict[str, Tuple[RetrievedChunk, float]] = {}
    for rank, hit in enumerate(vector_hits, start=1):
        score = vector_weight / (rrf_k + rank)
        fused[hit.chunk_id] = (hit, score)
    for rank, hit in enumerate(bm25_hits, start=1):
        score = bm25_weight / (rrf_k + rank)
        if hit.chunk_id in fused:
            existing, existing_score = fused[hit.chunk_id]
            fused[hit.chunk_id] = (
                RetrievedChunk(
                    chunk_id=existing.chunk_id,
                    doc_id=existing.doc_id,
                    text=existing.text,
                    score=existing_score + score,
                    metadata=existing.metadata,
                ),
                existing_score + score,
            )
        else:
            fused[hit.chunk_id] = (
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    score=score,
                    metadata={},
                ),
                score,
            )
    return [entry[0] for entry in sorted(fused.values(), key=lambda item: item[1], reverse=True)]
