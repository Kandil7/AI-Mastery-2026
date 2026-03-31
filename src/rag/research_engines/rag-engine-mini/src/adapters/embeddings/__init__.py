"""Embeddings adapters package."""

from src.adapters.embeddings.openai_embeddings import OpenAIEmbeddings

try:
    from src.adapters.embeddings.local_embeddings import LocalEmbeddings
except ModuleNotFoundError:
    LocalEmbeddings = None

__all__ = ["OpenAIEmbeddings", "LocalEmbeddings"]
