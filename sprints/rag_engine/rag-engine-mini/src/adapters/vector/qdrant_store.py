"""
Qdrant Vector Store Adapter
============================
Implementation of VectorStorePort for Qdrant.

محول مخزن متجهات Qdrant
"""

from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.application.ports.vector_store import ScoredChunkResult
from src.domain.entities import TenantId


class QdrantVectorStore:
    """
    Qdrant adapter implementing VectorStorePort.
    
    Design Decision: Minimal payload approach:
    - Only store tenant_id, document_id in payload
    - NO text in payload (saves storage, fetched from Postgres)
    
    قرار التصميم: نهج الحمولة الدنيا - لا نص في Qdrant
    """
    
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        vector_size: int,
    ) -> None:
        """
        Initialize Qdrant adapter.
        
        Args:
            client: Qdrant client instance
            collection: Collection name
            vector_size: Embedding dimension (must match model)
        """
        self._client = client
        self._collection = collection
        self._size = vector_size
    
    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            exists = self._client.collection_exists(self._collection)
            if not exists:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._size,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception:
            # May already exist from concurrent call
            pass
    
    def upsert_points(
        self,
        *,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        tenant_id: str,
        document_id: str,
    ) -> None:
        """
        Upsert vectors with minimal payload.
        
        Only stores:
        - tenant_id (for filtering)
        - document_id (for doc-specific search)
        
        NO text stored in Qdrant.
        """
        points = []
        for point_id, vector in zip(ids, vectors):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "tenant_id": tenant_id,
                        "document_id": document_id,
                    },
                )
            )
        
        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
    
    def search_scored(
        self,
        *,
        query_vector: list[float],
        tenant_id: TenantId,
        top_k: int,
        document_id: str | None = None,
    ) -> Sequence[ScoredChunkResult]:
        """
        Search for similar vectors with tenant isolation.
        
        Returns IDs and scores only (no text - hydrate separately).
        """
        # Build filter conditions
        must = [
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id.value),
            )
        ]
        
        # Optional document filter (ChatPDF mode)
        if document_id:
            must.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )
        
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=Filter(must=must),
            limit=top_k,
        )
        
        output: list[ScoredChunkResult] = []
        for r in results:
            payload = r.payload or {}
            output.append(
                ScoredChunkResult(
                    chunk_id=str(r.id),
                    score=float(r.score),
                    tenant_id=payload.get("tenant_id", tenant_id.value),
                    document_id=payload.get("document_id", document_id or ""),
                )
            )
        
        return output
    
    def delete_by_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> int:
        """Delete all vectors for a document."""
        # Qdrant delete with filter
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id.value),
                    ),
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    ),
                ]
            ),
        )
        # Qdrant doesn't return deleted count easily
        return 0
