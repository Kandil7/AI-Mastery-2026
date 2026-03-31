from __future__ import annotations

from typing import Any, Dict, List

from rag.chunking import Chunk
from rag.retriever import RetrievedChunk
from storage.vectordb_base import VectorStore


class QdrantStore(VectorStore):
    def __init__(self, endpoint: str, collection: str) -> None:
        self._endpoint = endpoint
        self._collection = collection
        self._client = self._create_client()

    def _create_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("qdrant-client package not installed") from exc

        return QdrantClient(url=self._endpoint)

    def _ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.http import models as rest

        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        if not chunks:
            return
        self._ensure_collection(vector_size=len(embeddings[0]))
        points = []
        for idx, chunk in enumerate(chunks):
            points.append(
                {
                    "id": chunk.chunk_id,
                    "vector": embeddings[idx],
                    "payload": {"doc_id": chunk.doc_id, "text": chunk.text, **metadata},
                }
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def query_by_vector(
        self,
        vector: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[RetrievedChunk]:
        from qdrant_client.http import models as rest

        qdrant_filter = None
        if filters:
            conditions = [rest.FieldCondition(key=k, match=rest.MatchValue(value=v)) for k, v in filters.items()]
            qdrant_filter = rest.Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        chunks: List[RetrievedChunk] = []
        for item in results:
            payload = item.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(item.id),
                    doc_id=str(payload.get("doc_id", "")),
                    text=str(payload.get("text", "")),
                    metadata=payload,
                    score=float(item.score),
                )
            )
        return chunks
