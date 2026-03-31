from __future__ import annotations

from typing import Any, Dict, List

from rag.chunking import Chunk
from rag.retriever import RetrievedChunk
from storage.vectordb_base import VectorStore


class WeaviateStore(VectorStore):
    def __init__(self, endpoint: str, index_name: str) -> None:
        self._endpoint = endpoint
        self._index_name = index_name
        self._client = self._create_client()

    def _create_client(self):
        try:
            import weaviate
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("weaviate-client package not installed") from exc

        return weaviate.Client(self._endpoint)

    def _ensure_schema(self, vector_size: int) -> None:
        schema = {
            "class": self._index_name,
            "vectorIndexType": "hnsw",
            "vectorizer": "none",
            "properties": [
                {"name": "doc_id", "dataType": ["text"]},
                {"name": "text", "dataType": ["text"]},
                {"name": "tenant_id", "dataType": ["text"]},
            ],
        }
        existing = self._client.schema.get()
        classes = {c["class"] for c in existing.get("classes", [])}
        if self._index_name not in classes:
            _ = vector_size
            self._client.schema.create_class(schema)

    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        if not chunks:
            return
        self._ensure_schema(vector_size=len(embeddings[0]))
        with self._client.batch as batch:
            for idx, chunk in enumerate(chunks):
                payload = {"doc_id": chunk.doc_id, "text": chunk.text, **metadata}
                batch.add_data_object(
                    data_object=payload,
                    class_name=self._index_name,
                    uuid=chunk.chunk_id,
                    vector=embeddings[idx],
                )

    def query_by_vector(
        self,
        vector: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[RetrievedChunk]:
        query = self._client.query.get(
            self._index_name, ["doc_id", "text", "tenant_id", "_additional {id distance}"]
        ).with_near_vector({"vector": vector})
        if filters:
            where = {"operator": "And", "operands": []}
            for key, value in filters.items():
                where["operands"].append(
                    {"path": [key], "operator": "Equal", "valueText": str(value)}
                )
            query = query.with_where(where)
        result = query.with_limit(top_k).do()
        items = result.get("data", {}).get("Get", {}).get(self._index_name, [])
        chunks: List[RetrievedChunk] = []
        for item in items:
            additional = item.get("_additional", {}) if isinstance(item, dict) else {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(additional.get("id", "")),
                    doc_id=str(item.get("doc_id", "")),
                    text=str(item.get("text", "")),
                    metadata=item,
                    score=float(additional.get("distance", 0.0)),
                )
            )
        return chunks
