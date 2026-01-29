"""Embeddings adapters package."""

from src.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
from src.adapters.embeddings.local_embeddings import LocalEmbeddings

__all__ = ["OpenAIEmbeddings", "LocalEmbeddings"]
