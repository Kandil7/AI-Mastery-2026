from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from core.factories import create_embeddings_provider, create_vector_store
from core.settings import load_settings
from rag.chunking import Chunk, simple_chunk
from rag.embeddings import EmbeddingService
from storage.vectordb_base import VectorStore


def ingest_document(
    tenant_id: str,
    source_type: str,
    uri: str,
    metadata: Dict[str, Any],
) -> str:
    settings = load_settings()
    embedder = EmbeddingService(create_embeddings_provider(settings))
    store = create_vector_store(settings)
    doc_id = str(uuid.uuid4())

    text = _load_source_text(source_type, uri)
    index_text(
        doc_id=doc_id,
        text=text,
        embedder=embedder,
        vector_store=store,
        metadata={"tenant_id": tenant_id, **metadata},
    )
    return doc_id


def index_text(
    doc_id: str,
    text: str,
    embedder: EmbeddingService,
    vector_store: VectorStore,
    metadata: Dict[str, Any],
    chunks: Optional[List[Chunk]] = None,
) -> None:
    resolved_chunks = chunks or simple_chunk(text=text, doc_id=doc_id)
    embeddings = embedder.embed([chunk.text for chunk in resolved_chunks])
    vector_store.upsert(resolved_chunks, embeddings, metadata=metadata)


def _load_source_text(source_type: str, uri: str) -> str:
    if source_type == "web":
        response = requests.get(uri, timeout=20)
        response.raise_for_status()
        return response.text
    if source_type == "file":
        return Path(uri).read_text(encoding="utf-8")
    if source_type == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("pypdf package not installed") from exc

        reader = PdfReader(uri)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported source_type: {source_type}")
