from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple


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


def structured_chunk(
    text: str,
    doc_id: str,
    max_tokens: int = 400,
    overlap: int = 40,
) -> List[Chunk]:
    # Chunk by section headings to preserve document structure.
    sections = _split_sections(text)
    chunks: List[Chunk] = []
    chunk_index = 0
    safe_overlap = max(0, min(overlap, max_tokens - 1))
    for heading, body in sections:
        words = body.split()
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            payload = " ".join(words[start:end])
            if heading:
                payload = f"{heading}\n{payload}"
            chunk_id = f"{doc_id}:{chunk_index}"
            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=payload))
            chunk_index += 1
            if end == len(words):
                break
            start = max(0, end - safe_overlap)
    return chunks


def build_chunker(
    mode: str,
    max_tokens: int = 400,
    overlap: int = 40,
) -> Callable[[str, str], List[Chunk]]:
    mode_lower = mode.lower()
    if mode_lower == "structured":
        return lambda text, doc_id: structured_chunk(
            text=text,
            doc_id=doc_id,
            max_tokens=max_tokens,
            overlap=overlap,
        )
    return lambda text, doc_id: simple_chunk(text=text, doc_id=doc_id, max_tokens=max_tokens)


def _split_sections(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_heading = ""
    current_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = stripped
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))
    return [(heading, "\n".join(lines)) for heading, lines in sections]
