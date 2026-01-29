"""
Placeholder Repository Implementations
=======================================
In-memory implementations for development/testing.
Replace with real Postgres implementations for production.

تنفيذات مؤقتة في الذاكرة للتطوير/الاختبار
"""

import uuid
from typing import Sequence

from src.domain.entities import (
    TenantId,
    DocumentId,
    DocumentStatus,
    StoredFile,
    Chunk,
)


class PlaceholderDocumentRepo:
    """In-memory document repository."""
    
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}
    
    def create_document(
        self,
        *,
        tenant_id: TenantId,
        stored_file: StoredFile,
        file_sha256: str | None = None,
    ) -> DocumentId:
        doc_id = DocumentId(str(uuid.uuid4()))
        self._docs[doc_id.value] = {
            "tenant_id": tenant_id.value,
            "stored_file": stored_file,
            "file_sha256": file_sha256,
            "status": "created",
            "error": None,
        }
        return doc_id
    
    def set_status(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
        status: str,
        error: str | None = None,
    ) -> None:
        doc = self._docs.get(document_id.value)
        if doc and doc["tenant_id"] == tenant_id.value:
            doc["status"] = status
            doc["error"] = error
    
    def get_status(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> DocumentStatus | None:
        doc = self._docs.get(document_id.value)
        if not doc or doc["tenant_id"] != tenant_id.value:
            return None
        return DocumentStatus(
            document_id=document_id,
            tenant_id=tenant_id,
            filename=doc["stored_file"].filename,
            status=doc["status"],
            error=doc["error"],
        )
    
    def list_documents(
        self,
        *,
        tenant_id: TenantId,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[DocumentStatus]:
        result = []
        for doc_id, doc in self._docs.items():
            if doc["tenant_id"] == tenant_id.value:
                result.append(
                    DocumentStatus(
                        document_id=DocumentId(doc_id),
                        tenant_id=tenant_id,
                        filename=doc["stored_file"].filename,
                        status=doc["status"],
                    )
                )
        return result[offset : offset + limit]
    
    def delete_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> bool:
        doc = self._docs.get(document_id.value)
        if doc and doc["tenant_id"] == tenant_id.value:
            del self._docs[document_id.value]
            return True
        return False


class PlaceholderDocumentIdempotencyRepo:
    """In-memory document idempotency repository."""
    
    def __init__(self) -> None:
        self._by_hash: dict[str, dict] = {}  # key: f"{tenant}:{hash}"
        self._docs: dict[str, dict] = {}
    
    def get_by_file_hash(
        self,
        *,
        tenant_id: TenantId,
        file_sha256: str,
    ) -> DocumentId | None:
        key = f"{tenant_id.value}:{file_sha256}"
        if key in self._by_hash:
            return DocumentId(self._by_hash[key]["doc_id"])
        return None
    
    def create_document_with_hash(
        self,
        *,
        tenant_id: TenantId,
        stored_file: StoredFile,
        file_sha256: str,
    ) -> DocumentId:
        key = f"{tenant_id.value}:{file_sha256}"
        
        # Check if exists
        if key in self._by_hash:
            return DocumentId(self._by_hash[key]["doc_id"])
        
        # Create new
        doc_id = str(uuid.uuid4())
        self._by_hash[key] = {"doc_id": doc_id}
        self._docs[doc_id] = {
            "tenant_id": tenant_id.value,
            "stored_file": stored_file,
            "file_sha256": file_sha256,
            "status": "created",
        }
        return DocumentId(doc_id)


class PlaceholderDocumentReader:
    """In-memory document reader (shares state with repo)."""
    
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}
    
    def get_stored_file(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> StoredFile | None:
        doc = self._docs.get(document_id.value)
        if doc and doc.get("tenant_id") == tenant_id.value:
            return doc.get("stored_file")
        return None


class PlaceholderChunkDedupRepo:
    """In-memory chunk dedup repository."""
    
    def __init__(self) -> None:
        self._chunks: dict[str, dict] = {}  # key: f"{tenant}:{hash}"
        self._doc_chunks: dict[str, list[str]] = {}  # key: doc_id
    
    def upsert_chunk_store(
        self,
        *,
        tenant_id: TenantId,
        chunk_hash: str,
        text: str,
    ) -> str:
        key = f"{tenant_id.value}:{chunk_hash}"
        if key in self._chunks:
            return self._chunks[key]["id"]
        
        chunk_id = str(uuid.uuid4())
        self._chunks[key] = {"id": chunk_id, "text": text}
        return chunk_id
    
    def replace_document_chunks(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
        chunk_ids_in_order: Sequence[str],
    ) -> None:
        self._doc_chunks[document_id] = list(chunk_ids_in_order)
    
    def get_document_chunk_ids(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> Sequence[str]:
        return self._doc_chunks.get(document_id, [])
    
    def delete_document_chunks(
        self,
        *,
        tenant_id: TenantId,
        document_id: str,
    ) -> int:
        if document_id in self._doc_chunks:
            count = len(self._doc_chunks[document_id])
            del self._doc_chunks[document_id]
            return count
        return 0


class PlaceholderChunkTextReader:
    """In-memory chunk text reader."""
    
    def __init__(self) -> None:
        self._texts: dict[str, str] = {}
    
    def get_texts_by_ids(
        self,
        *,
        tenant_id: TenantId,
        chunk_ids: Sequence[str],
    ) -> dict[str, str]:
        return {cid: self._texts.get(cid, "") for cid in chunk_ids if cid in self._texts}


class PlaceholderKeywordStore:
    """In-memory keyword store (no actual FTS)."""
    
    def search(
        self,
        *,
        query: str,
        tenant_id: TenantId,
        top_k: int,
        document_id: str | None = None,
    ) -> Sequence[Chunk]:
        # Placeholder - returns empty in dev mode
        # Real implementation uses Postgres FTS
        return []
