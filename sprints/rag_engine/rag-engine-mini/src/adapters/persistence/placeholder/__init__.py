"""Persistence adapters placeholder package."""

from src.adapters.persistence.placeholder.repos import (
    PlaceholderDocumentRepo,
    PlaceholderDocumentIdempotencyRepo,
    PlaceholderDocumentReader,
    PlaceholderChunkDedupRepo,
    PlaceholderChunkTextReader,
    PlaceholderKeywordStore,
)

__all__ = [
    "PlaceholderDocumentRepo",
    "PlaceholderDocumentIdempotencyRepo",
    "PlaceholderDocumentReader",
    "PlaceholderChunkDedupRepo",
    "PlaceholderChunkTextReader",
    "PlaceholderKeywordStore",
]
