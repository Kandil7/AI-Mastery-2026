"""
Embeddings Module

Provides unified interfaces for text and image embeddings with caching support.
"""

from .embeddings import (
    TextEmbedder,
    ImageEmbedder,
    MultiModalEmbedder,
    EmbeddingCache,
    EmbeddingConfig,
)

__all__ = [
    "TextEmbedder",
    "ImageEmbedder", 
    "MultiModalEmbedder",
    "EmbeddingCache",
    "EmbeddingConfig",
]
