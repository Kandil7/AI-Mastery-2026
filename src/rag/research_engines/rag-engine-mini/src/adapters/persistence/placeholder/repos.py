"""
Placeholder Repository Implementations
=======================================
In-memory implementations for development/testing.
Replace with real Postgres implementations for production.

تنفيذات مؤقتة في الذاكرة للتطوير/الاختبار
"""

import uuid
from datetime import datetime
from typing import Sequence

from src.domain.entities import (
    TenantId,
    DocumentId,
    DocumentStatus,
    StoredFile,
    Chunk,
    ChatSession,
    ChatTurn,
)
from src.adapters.security.password_hasher import verify_password
from src.application.services.document_search import (
    SearchFilter,
    SearchResult,
    SearchResultPagination,
    SortOrder,
)


# Shared placeholder stores (module-level for cross-repo visibility)
_PLACEHOLDER_DOCS: dict[str, dict] = {}
_PLACEHOLDER_CHUNKS: dict[str, dict] = {}
_PLACEHOLDER_DOC_CHUNKS: dict[str, list[str]] = {}
_PLACEHOLDER_CHUNK_TEXTS: dict[str, str] = {}
_PLACEHOLDER_CHAT_SESSIONS: dict[str, dict] = {}
_PLACEHOLDER_CHAT_TURNS: dict[str, list[ChatTurn]] = {}
_PLACEHOLDER_GRAPH: dict[str, list[dict]] = {}
_PLACEHOLDER_USERS: dict[str, dict] = {}


class PlaceholderDocumentRepo:
    """In-memory document repository."""
    
    def __init__(self) -> None:
        self._docs = _PLACEHOLDER_DOCS
    
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
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "chunks_count": 0,
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
            doc["updated_at"] = datetime.utcnow()
    
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

    def count_documents(
        self,
        *,
        tenant_id: TenantId,
        filters: SearchFilter | None = None,
    ) -> int:
        count = 0
        for doc_id, doc in self._docs.items():
            if doc["tenant_id"] != tenant_id.value:
                continue
            if filters and not filters.matches_document(_doc_to_dict(doc_id, doc)):
                continue
            count += 1
        return count

    def get_document(
        self,
        *,
        tenant_id: TenantId,
        document_id: DocumentId,
    ) -> dict | None:
        doc = self._docs.get(document_id.value)
        if not doc or doc["tenant_id"] != tenant_id.value:
            return None
        return _doc_to_dict(document_id.value, doc)

    def search_documents(
        self,
        *,
        tenant_id: TenantId,
        query: str,
        filters: SearchFilter | None = None,
        sort_order: SortOrder | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> SearchResultPagination:
        query_lower = query.lower().strip()
        results: list[SearchResult] = []

        for doc_id, doc in self._docs.items():
            if doc["tenant_id"] != tenant_id.value:
                continue
            doc_dict = _doc_to_dict(doc_id, doc)
            if query_lower and query_lower not in doc_dict["filename"].lower():
                continue
            if filters and not filters.matches_document(doc_dict):
                continue
            results.append(
                SearchResult(
                    document_id=doc_dict["id"],
                    filename=doc_dict["filename"],
                    status=doc_dict["status"],
                    size_bytes=doc_dict["size_bytes"],
                    content_type=doc_dict["content_type"],
                    created_at=doc_dict["created_at"].isoformat()
                    if doc_dict["created_at"]
                    else "",
                    chunks_count=doc_dict["chunks_count"],
                    matches_filters=[],
                )
            )

        sort_order = sort_order or SortOrder.CREATED_DESC
        if sort_order == SortOrder.CREATED_ASC:
            results.sort(key=lambda r: r.created_at)
        elif sort_order == SortOrder.CREATED_DESC:
            results.sort(key=lambda r: r.created_at, reverse=True)
        elif sort_order == SortOrder.FILENAME_ASC:
            results.sort(key=lambda r: r.filename.lower())
        elif sort_order == SortOrder.FILENAME_DESC:
            results.sort(key=lambda r: r.filename.lower(), reverse=True)
        elif sort_order == SortOrder.SIZE_ASC:
            results.sort(key=lambda r: r.size_bytes)
        elif sort_order == SortOrder.SIZE_DESC:
            results.sort(key=lambda r: r.size_bytes, reverse=True)

        total = len(results)
        paged = results[offset : offset + limit]
        return SearchResultPagination(
            results=paged,
            total=total,
            offset=offset,
            limit=limit,
            has_next=(offset + limit) < total,
            has_prev=offset > 0,
        )


class PlaceholderDocumentIdempotencyRepo:
    """In-memory document idempotency repository."""
    
    def __init__(self) -> None:
        self._by_hash: dict[str, dict] = {}  # key: f"{tenant}:{hash}"
        self._docs = _PLACEHOLDER_DOCS
    
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
            "error": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "chunks_count": 0,
        }
        return DocumentId(doc_id)


class PlaceholderDocumentReader:
    """In-memory document reader (shares state with repo)."""
    
    def __init__(self) -> None:
        self._docs = _PLACEHOLDER_DOCS
    
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
        self._chunks = _PLACEHOLDER_CHUNKS  # key: f"{tenant}:{hash}"
        self._doc_chunks = _PLACEHOLDER_DOC_CHUNKS  # key: doc_id
    
    def upsert_chunk_store(
        self,
        *,
        tenant_id: TenantId,
        chunk_hash: str,
        text: str,
        parent_id: str | None = None,
        chunk_context: str | None = None,
    ) -> str:
        key = f"{tenant_id.value}:{chunk_hash}"
        if key in self._chunks:
            return self._chunks[key]["id"]

        chunk_id = str(uuid.uuid4())
        self._chunks[key] = {
            "id": chunk_id,
            "text": text,
            "tenant_id": tenant_id.value,
            "parent_id": parent_id,
            "chunk_context": chunk_context,
        }
        _PLACEHOLDER_CHUNK_TEXTS[chunk_id] = text
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
        self._texts = _PLACEHOLDER_CHUNK_TEXTS
    
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
        results: list[Chunk] = []
        query_lower = query.lower()
        for chunk_id, meta in _PLACEHOLDER_CHUNKS.items():
            if meta.get("tenant_id") != tenant_id.value:
                continue
            text = meta.get("text", "")
            if query_lower in text.lower():
                results.append(
                    Chunk(
                        id=meta["id"],
                        tenant_id=tenant_id,
                        document_id=DocumentId(document_id or ""),
                        text=text,
                        parent_id=meta.get("parent_id"),
                        chunk_context=meta.get("chunk_context"),
                    )
                )
                if len(results) >= top_k:
                    break
        return results


def _doc_to_dict(doc_id: str, doc: dict) -> dict:
    stored = doc["stored_file"]
    return {
        "id": doc_id,
        "user_id": doc["tenant_id"],
        "filename": stored.filename,
        "content_type": stored.content_type,
        "file_path": stored.path,
        "size_bytes": stored.size_bytes,
        "file_sha256": doc.get("file_sha256"),
        "status": doc.get("status"),
        "error": doc.get("error"),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "chunks_count": doc.get("chunks_count", 0),
    }


class PlaceholderChatRepo:
    """In-memory chat repository."""

    def create_session(
        self,
        *,
        tenant_id: TenantId,
        title: str | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        _PLACEHOLDER_CHAT_SESSIONS[session_id] = {
            "id": session_id,
            "tenant_id": tenant_id.value,
            "title": title,
            "created_at": datetime.utcnow(),
        }
        _PLACEHOLDER_CHAT_TURNS[session_id] = []
        return session_id

    def add_turn(
        self,
        *,
        tenant_id: TenantId,
        session_id: str,
        question: str,
        answer: str,
        sources: Sequence[str],
        retrieval_k: int,
        embed_ms: int | None = None,
        search_ms: int | None = None,
        llm_ms: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> str:
        turn_id = str(uuid.uuid4())
        turn = ChatTurn(
            id=turn_id,
            session_id=session_id,
            tenant_id=tenant_id,
            question=question,
            answer=answer,
            sources=sources,
            retrieval_k=retrieval_k,
            embed_ms=embed_ms,
            search_ms=search_ms,
            llm_ms=llm_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            created_at=datetime.utcnow(),
        )
        _PLACEHOLDER_CHAT_TURNS.setdefault(session_id, []).append(turn)
        return turn_id

    def get_session_turns(
        self,
        *,
        tenant_id: TenantId,
        session_id: str,
        limit: int = 50,
    ) -> Sequence[ChatTurn]:
        turns = _PLACEHOLDER_CHAT_TURNS.get(session_id, [])
        return turns[-limit:]

    def list_sessions(
        self,
        *,
        tenant_id: TenantId,
        limit: int = 50,
    ) -> Sequence[ChatSession]:
        sessions = [
            ChatSession(
                id=s["id"],
                tenant_id=TenantId(s["tenant_id"]),
                title=s.get("title"),
                created_at=s.get("created_at"),
            )
            for s in _PLACEHOLDER_CHAT_SESSIONS.values()
            if s["tenant_id"] == tenant_id.value
        ]
        return sessions[:limit]


class PlaceholderGraphRepo:
    """In-memory graph repository."""

    def save_triplets(
        self,
        tenant_id: TenantId,
        document_id: str,
        chunk_id: str,
        triplets: list[dict],
    ) -> None:
        if not triplets:
            return
        key = f"{tenant_id.value}:{document_id}"
        stored = _PLACEHOLDER_GRAPH.setdefault(key, [])
        for t in triplets:
            stored.append(
                {
                    "subject": t.get("subject"),
                    "relation": t.get("relation"),
                    "obj": t.get("obj"),
                }
            )

    def get_triplets_by_entity(
        self,
        tenant_id: TenantId,
        entity_name: str,
    ) -> list[dict]:
        result: list[dict] = []
        needle = entity_name.lower()
        for key, triplets in _PLACEHOLDER_GRAPH.items():
            if not key.startswith(f"{tenant_id.value}:"):
                continue
            for t in triplets:
                if needle in str(t.get("subject", "")).lower() or needle in str(
                    t.get("obj", "")
                ).lower():
                    result.append(t)
        return result

    def delete_by_document(self, tenant_id: TenantId, document_id: str) -> int:
        key = f"{tenant_id.value}:{document_id}"
        if key not in _PLACEHOLDER_GRAPH:
            return 0
        count = len(_PLACEHOLDER_GRAPH[key])
        del _PLACEHOLDER_GRAPH[key]
        return count


class PlaceholderUserRepo:
    """In-memory user repository for dev/testing."""

    def create_user(self, *, email: str, hashed_password: str) -> str:
        user_id = str(uuid.uuid4())
        _PLACEHOLDER_USERS[user_id] = {
            "id": user_id,
            "email": email,
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow(),
        }
        return user_id

    def get_user_by_email(self, *, email: str) -> dict | None:
        for user in _PLACEHOLDER_USERS.values():
            if user["email"].lower() == email.lower():
                return user
        return None

    def get_user_by_id(self, *, user_id: str) -> dict | None:
        return _PLACEHOLDER_USERS.get(user_id)

    def email_exists(self, *, email: str) -> bool:
        return self.get_user_by_email(email=email) is not None

    def update_password(self, *, user_id: str, hashed_password: str) -> None:
        if user_id in _PLACEHOLDER_USERS:
            _PLACEHOLDER_USERS[user_id]["hashed_password"] = hashed_password

    def verify_password(self, *, email: str, plain_password: str) -> bool:
        user = self.get_user_by_email(email=email)
        if not user:
            return False
        return verify_password(plain_password, user["hashed_password"])
