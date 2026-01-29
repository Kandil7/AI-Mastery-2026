"""
Vector Store Port
==================
Interface for vector database operations.

منفذ مخزن المتجهات
"""

from typing import Protocol, Sequence

from src.domain.entities import TenantId


class ScoredChunkResult:
    """Result from vector search with score."""
    
    def __init__(self, chunk_id: str, score: float, tenant_id: str, document_id: str) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.tenant_id = tenant_id
        self.document_id = document_id


class VectorStorePort(Protocol):
    """
    Port for vector storage and similarity search.
    
    Implementations: Qdrant, PGVector, Pinecone, etc.
    
    Design Decision: Minimal payload approach - only store IDs and metadata
    in vector store, hydrate text from Postgres for cost/storage efficiency.
    
    قرار التصميم: نهج الحمولة الدنيا - تخزين المعرفات فقط في مخزن المتجهات
    """
    
    def ensure_collection(self) -> None:
        """
        Ensure the collection exists, create if not.
        Called at startup and before upserts.
        """
        ...
    
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
        
        Args:
            ids: Point IDs (chunk IDs)
            vectors: Embedding vectors
            tenant_id: Tenant for isolation
            document_id: Source document ID
            
        Design Note:
            No text in payload - only IDs for reference.
            Text is fetched from Postgres when needed.
        """
        ...
    
    def search_scored(
        self,
        *,
        query_vector: list[float],
        tenant_id: TenantId,
        top_k: int,
        document_id: str | None = None,
    ) -> Sequence[ScoredChunkResult]:
        """
        Search for similar vectors with scores.
        
        Args:
            query_vector: Query embedding
            tenant_id: Tenant for isolation filter
            top_k: Number of results
            document_id: Optional filter for document-specific search
            
        Returns:
            Scored results (IDs and scores, no text)
            
        Note:
            Results need text hydration from database.
            النتائج تحتاج جلب النص من قاعدة البيانات
        """
        ...
    
    def delete_by_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> int:
        """
        Delete all vectors for a document.
        
        Args:
            tenant_id: Tenant for isolation
            document_id: Document to delete
            
        Returns:
            Number of deleted points
        """
        ...
