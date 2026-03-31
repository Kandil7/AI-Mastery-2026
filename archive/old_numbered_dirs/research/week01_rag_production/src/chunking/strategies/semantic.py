# src/chunking/strategies/semantic.py
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from src.retrieval import Document

from ..base import BaseChunker
from ..config import ChunkingConfig
from ..spans import ChunkSpan, TextSpan
from ..tokenizer import TokenCounter, build_counter
from ..sanitize import sanitize_text


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that splits text at paragraph and sentence boundaries.

    Uses token-aware counting (tiktoken when available, else characters) and
    preserves exact spans into the original, sanitized document.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.sentence_endings = re.compile(r"[.!?]+")
        self.counter: TokenCounter = build_counter(prefer_tiktoken=True)

    def chunk_document(self, document: Document) -> List[Document]:
        sanitized_content = sanitize_text(
            document.content,
            max_chars=self.config.max_document_chars,
            strip_control_chars=self.config.strip_control_chars,
            normalize_newlines=self.config.normalize_newlines,
        )

        paragraphs = list(self._paragraphs_with_spans(sanitized_content))

        chunks: List[Document] = []
        chunk_index = 0

        for para_text, para_start, para_end in paragraphs:
            if self.counter.count(para_text) <= self.config.chunk_size:
                chunk_span = ChunkSpan(text=para_text, span=TextSpan(para_start, para_end))
                chunks.append(self._create_chunk_document(document, chunk_span, chunk_index))
                chunk_index += 1
                continue

            sentences = list(self._sentences_with_spans(para_text, base_offset=para_start))
            current_start: int | None = None
            current_end: int | None = None

            for sentence_text, s_start, s_end in sentences:
                if current_start is None:
                    current_start = s_start
                    current_end = s_end
                    continue

                candidate_end = s_end
                candidate_text = sanitized_content[current_start:candidate_end]

                if self.counter.count(candidate_text) <= self.config.chunk_size:
                    current_end = candidate_end
                    continue

                # flush accumulated chunk
                if current_end is not None and current_end > current_start:
                    chunk_span = ChunkSpan(
                        text=sanitized_content[current_start:current_end],
                        span=TextSpan(current_start, current_end),
                    )
                    chunks.append(self._create_chunk_document(document, chunk_span, chunk_index))
                    chunk_index += 1

                # handle oversized sentence
                if self.counter.count(sentence_text) > self.config.chunk_size:
                    for sub_text, sub_start, sub_end in self._split_large_sentence(sentence_text, s_start):
                        chunk_span = ChunkSpan(text=sub_text, span=TextSpan(sub_start, sub_end))
                        chunks.append(self._create_chunk_document(document, chunk_span, chunk_index))
                        chunk_index += 1
                    current_start = None
                    current_end = None
                else:
                    current_start = s_start
                    current_end = s_end

            if current_start is not None and current_end is not None and current_end > current_start:
                chunk_span = ChunkSpan(
                    text=sanitized_content[current_start:current_end],
                    span=TextSpan(current_start, current_end),
                )
                chunks.append(self._create_chunk_document(document, chunk_span, chunk_index))
                chunk_index += 1

        return self._apply_overlap_to_semantic_chunks(chunks)

    def _paragraphs_with_spans(self, text: str) -> Iterable[Tuple[str, int, int]]:
        """Yield paragraphs with absolute spans using double newlines as boundaries."""
        for match in re.finditer(r"(.*?)(\n\n|$)", text, flags=re.DOTALL):
            para = match.group(1)
            if not para or not para.strip():
                continue
            start = match.start(1)
            end = match.end(1)
            yield para, start, end

    def _sentences_with_spans(self, text: str, base_offset: int) -> Iterable[Tuple[str, int, int]]:
        """Yield sentences within a paragraph with absolute spans."""
        last = 0
        for match in self.sentence_endings.finditer(text):
            end = match.end()
            sentence = text[last:end]
            if sentence.strip():
                yield sentence, base_offset + last, base_offset + end
            last = end
        if last < len(text):
            tail = text[last:]
            if tail.strip():
                yield tail, base_offset + last, base_offset + len(text)

    def _split_large_sentence(self, sentence: str, base_offset: int) -> List[Tuple[str, int, int]]:
        """Fallback to character-based splitting for oversized sentences."""
        chunks: List[Tuple[str, int, int]] = []
        start = 0
        prev_start = -1
        while start < len(sentence):
            end = min(start + self.config.chunk_size, len(sentence))
            chunk_text = sentence[start:end]
            chunk_start = base_offset + start
            chunk_end = base_offset + end
            chunks.append((chunk_text, chunk_start, chunk_end))
            prev_start = start
            start = end - self.config.chunk_overlap
            if start <= prev_start:
                start = prev_start + 1
        return chunks

    def _apply_overlap_to_semantic_chunks(self, chunks: List[Document]) -> List[Document]:
        """Apply character-based overlap while keeping original spans."""
        if not chunks or self.config.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = result[-1]

            prev_content = previous_chunk.content
            overlap_size = min(self.config.chunk_overlap, len(prev_content))

            new_content = (
                prev_content[-overlap_size:] + current_chunk.content if overlap_size > 0 else current_chunk.content
            )

            new_metadata = {
                **(getattr(current_chunk, "metadata", {}) or {}),
                "has_overlap": True,
                "overlap_from": getattr(previous_chunk, "id", None),
                "overlap_size": overlap_size,
                "chunk_char_len": len(new_content),
            }

            ctor_kwargs = {
                "id": getattr(current_chunk, "id", None),
                "content": new_content,
                "metadata": new_metadata,
            }
            if hasattr(current_chunk, "embedding"):
                ctor_kwargs["embedding"] = getattr(current_chunk, "embedding")

            new_doc = Document(**ctor_kwargs)

            for attr in (
                "source",
                "doc_type",
                "created_at",
                "updated_at",
                "access_control",
                "page_number",
                "section_title",
            ):
                if hasattr(current_chunk, attr):
                    setattr(new_doc, attr, getattr(current_chunk, attr))

            result.append(new_doc)

        return result
