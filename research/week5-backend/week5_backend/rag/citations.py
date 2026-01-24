from __future__ import annotations

from typing import List

from rag.retriever import RetrievedChunk


def format_citations(chunks: List[RetrievedChunk]) -> List[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "score": chunk.score,
            "snippet": chunk.text[:180],
        }
        for chunk in chunks
    ]
