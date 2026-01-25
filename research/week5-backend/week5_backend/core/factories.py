from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from agents.policies import RoutingPolicy
from core.settings import Settings
from providers.anthropic_provider import AnthropicProvider
from providers.embeddings_provider import EmbeddingsProvider
from providers.llm_base import LLMProvider
from providers.local_embeddings import LocalEmbeddings
from providers.local_vllm_provider import LocalVLLMProvider
from providers.openai_embeddings import OpenAIEmbeddings
from providers.openai_provider import OpenAIProvider
from storage.pgvector_store import PgVectorStore
from storage.qdrant_store import QdrantStore
from storage.vectordb_base import VectorStore
from storage.weaviate_store import WeaviateStore
from rag.bm25_store import load_bm25_index


def _provider_config(settings: Settings, name: str) -> Dict[str, Any]:
    return (settings.raw.get("providers") or {}).get(name, {})


def _vector_config(settings: Settings, name: str) -> Dict[str, Any]:
    return (settings.raw.get("vectorstores") or {}).get(name, {})


def create_llm_provider(settings: Settings, name: str | None = None) -> LLMProvider:
    provider_name = name or settings.default_provider
    config = _provider_config(settings, provider_name)
    if provider_name == "openai":
        return OpenAIProvider(
            model=str(config.get("model", "gpt-4o-mini")),
            api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=config.get("base_url"),
        )
    if provider_name == "anthropic":
        return AnthropicProvider(
            model=str(config.get("model", "claude-3-5-sonnet")),
            api_key=config.get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
        )
    if provider_name == "local_vllm":
        return LocalVLLMProvider(
            model=str(config.get("model", "meta-llama/Llama-3-8B-Instruct")),
            base_url=str(config.get("base_url", "http://localhost:8000/v1")),
            api_key=config.get("api_key") or os.getenv("LOCAL_VLLM_API_KEY") or "local",
        )
    raise ValueError(f"Unsupported provider: {provider_name}")


def create_embeddings_provider(
    settings: Settings,
    name: str | None = None,
) -> EmbeddingsProvider:
    provider_name = name or settings.default_provider
    config = _provider_config(settings, provider_name)
    if provider_name == "openai":
        return OpenAIEmbeddings(
            model=str(config.get("embeddings_model", "text-embedding-3-large")),
            api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=config.get("base_url"),
        )
    if provider_name == "local_vllm":
        return LocalEmbeddings(
            model=str(config.get("embeddings_model", "bge-small-en")),
            base_url=str(config.get("base_url", "http://localhost:8000/v1")),
            api_key=config.get("api_key") or os.getenv("LOCAL_VLLM_API_KEY") or "local",
        )
    raise ValueError(f"Unsupported embeddings provider: {provider_name}")


def create_vector_store(settings: Settings, name: str | None = None) -> VectorStore:
    store_name = name or settings.default_vector_store
    config = _vector_config(settings, store_name)
    if store_name == "pgvector":
        return PgVectorStore(
            dsn=str(config.get("dsn", "postgresql://localhost/rag")),
            table=str(config.get("table", "rag_chunks")),
            embedding_dim=int(config.get("embedding_dim", 1536)),
        )
    if store_name == "qdrant":
        return QdrantStore(
            endpoint=str(config.get("endpoint", "http://localhost:6333")),
            collection=str(config.get("collection", "rag_chunks")),
        )
    if store_name == "weaviate":
        return WeaviateStore(
            endpoint=str(config.get("endpoint", "http://localhost:8080")),
            index_name=str(config.get("index_name", "RagChunk")),
        )
    raise ValueError(f"Unsupported vector store: {store_name}")


def create_bm25_index(settings: Settings):
    path = settings.raw.get("bm25_index_path", "data/bm25_index.jsonl")
    bm25_path = Path(path)
    if not bm25_path.is_absolute():
        bm25_path = Path(__file__).resolve().parents[1] / bm25_path
    return load_bm25_index(bm25_path)


def create_routing_policy(settings: Settings) -> RoutingPolicy:
    routing = settings.raw.get("routing") or {}
    return RoutingPolicy(
        default_provider=str(routing.get("default_provider", settings.default_provider)),
        fallback_provider=str(routing.get("fallback_provider", settings.default_provider)),
    )
