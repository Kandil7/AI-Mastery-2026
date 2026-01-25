# src/chunking/base.py
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from src.retrieval import Document  # your existing model

from .config import ChunkingConfig
from .spans import ChunkSpan, TextSpan


class BaseChunker(ABC):
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Document]:
        raise NotImplementedError

    def _stable_chunk_id(self, original_id: str, chunk_index: int, span: TextSpan, content: str) -> str:
        # Deterministic ID for idempotency across runs
        h = hashlib.sha1()
        h.update(original_id.encode("utf-8"))
        h.update(f":{chunk_index}:{span.start}:{span.end}".encode("utf-8"))
        # hash prefix avoids gigantic IDs while still reducing collision risk
        h.update(content[:256].encode("utf-8", errors="ignore"))
        digest = h.hexdigest()[:12]
        return f"{original_id}_chunk_{chunk_index}_{digest}"

    def _create_chunk_document(self, original_doc: Document, chunk: ChunkSpan, chunk_index: int) -> Document:
        content = chunk.text
        span = chunk.span

        chunk_id = self._stable_chunk_id(original_doc.id, chunk_index, span, content)

        chunk_metadata = {
            **(original_doc.metadata or {}),
            "chunk_index": chunk_index,
            "chunk_start": span.start,
            "chunk_end": span.end,
            "original_id": original_doc.id,
            "chunk_char_len": len(content),
            "chunk_strategy": self.__class__.__name__,
        }

        return Document(
            id=chunk_id,
            content=content,
            source=original_doc.source,
            doc_type=f"{original_doc.doc_type}_chunk",
            metadata=chunk_metadata,
            created_at=original_doc.created_at,
            updated_at=original_doc.updated_at,
            access_control=original_doc.access_control,
            page_number=original_doc.page_number,
            section_title=original_doc.section_title,
        )