from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def simple_chunk(text: str, doc_id: str, max_tokens: int = 400) -> List[Chunk]:
    # Naive word-based chunker for teaching purposes.
    words = text.split()
    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_id = f"{doc_id}:{chunk_index}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text))
        chunk_index += 1
        start = end
    return chunks
