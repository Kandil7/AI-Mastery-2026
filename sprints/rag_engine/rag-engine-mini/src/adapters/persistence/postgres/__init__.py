"""Postgres persistence adapters package."""

from src.adapters.persistence.postgres.db import Base, engine, SessionLocal
from src.adapters.persistence.postgres.models import User, Document
from src.adapters.persistence.postgres.models_chunk_store import ChunkStoreRow, DocumentChunkRow
from src.adapters.persistence.postgres.repo_documents import PostgresDocumentRepo
from src.adapters.persistence.postgres.repo_users import UserLookupRepo
from src.adapters.persistence.postgres.repo_chunks import PostgresChunkDedupRepo
from src.adapters.persistence.postgres.keyword_store import PostgresKeywordStore
from src.adapters.persistence.postgres.chunk_text_reader import PostgresChunkTextReader

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "User",
    "Document",
    "ChunkStoreRow",
    "DocumentChunkRow",
    "PostgresDocumentRepo",
    "UserLookupRepo",
    "PostgresChunkDedupRepo",
    "PostgresKeywordStore",
    "PostgresChunkTextReader",
]
