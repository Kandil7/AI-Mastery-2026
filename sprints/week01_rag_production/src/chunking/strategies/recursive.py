# src/chunking/strategies/recursive.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

from src.retrieval import Document

from ..base import BaseChunker
from ..config import ChunkingConfig
from ..spans import ChunkSpan, TextSpan
from ..tokenizer import TokenCounter, CharCounter
from ..sanitize import sanitize_text


@dataclass(frozen=True)
class _Piece:
    text: str
    span: TextSpan


def _split_with_separator(text: str, sep: str, base_offset: int) -> List[_Piece]:
    """
    Split text by separator while preserving exact spans.
    The separator is not included in pieces (consistent with typical split),
    but spans remain correct within original.
    """
    if sep == "":
        # fallback: no natural split
        return [_Piece(text=text, span=TextSpan(base_offset, base_offset + len(text)))]

    pieces: List[_Piece] = []
    i = 0
    while True:
        j = text.find(sep, i)
        if j == -1:
            tail = text[i:]
            if tail:
                pieces.append(_Piece(tail, TextSpan(base_offset + i, base_offset + len(text))))
            break
        part = text[i:j]
        if part:
            pieces.append(_Piece(part, TextSpan(base_offset + i, base_offset + j)))
        i = j + len(sep)
    return pieces


def _merge_pieces(
    pieces: List[_Piece],
    sep: str,
    chunk_size_units: int,
    min_chunk_units: int,
    counter: TokenCounter,
) -> List[_Piece]:
    """
    Greedy merge pieces into chunks with max chunk_size_units.
    Units are given by counter (tokens if available, else chars).
    """
    merged: List[_Piece] = []
    cur_text = ""
    cur_start = None
    cur_end = None

    for p in pieces:
        candidate = (cur_text + sep + p.text) if cur_text else p.text
        if counter.count(candidate) <= chunk_size_units:
            if cur_start is None:
                cur_start = p.span.start
            cur_text = candidate
            cur_end = p.span.end
            continue

        # flush current
        if cur_text and cur_start is not None and cur_end is not None:
            if counter.count(cur_text) >= min_chunk_units:
                merged.append(_Piece(cur_text, TextSpan(cur_start, cur_end)))
            else:
                # too small: still keep if nothing else exists
                merged.append(_Piece(cur_text, TextSpan(cur_start, cur_end)))

        # start new with p
        cur_text = p.text
        cur_start = p.span.start
        cur_end = p.span.end

    if cur_text and cur_start is not None and cur_end is not None:
        merged.append(_Piece(cur_text, TextSpan(cur_start, cur_end)))

    return merged


def _apply_overlap_units(
    chunks: List[_Piece],
    counter: TokenCounter,
    overlap_units: int,
    chunk_size_units: int,
) -> List[_Piece]:
    """
    Apply overlap at final stage, without losing exact spans.
    Overlap is implemented by expanding chunk i to include a suffix of chunk i-1.
    """
    if not chunks or overlap_units <= 0:
        return chunks

    result = []
    for i, chunk in enumerate(chunks):
        # For first chunk, no previous to overlap with
        if i == 0:
            result.append(chunk)
            continue

        # Calculate how much of previous chunk to include as overlap
        prev_chunk_text = chunks[i-1].text
        prev_chunk_size = counter.count(prev_chunk_text)
        
        # Determine overlap size in units
        overlap_size_units = min(overlap_units, prev_chunk_size)
        
        if overlap_size_units <= 0:
            result.append(chunk)
            continue

        # Get the overlapping portion from the previous chunk
        overlap_text = counter.slice_by_units(prev_chunk_text, prev_chunk_size - overlap_size_units, prev_chunk_size)
        
        # Create new chunk with overlap prepended
        new_text = overlap_text + chunk.text
        new_span = TextSpan(chunks[i-1].span.start, chunk.span.end)
        
        result.append(_Piece(new_text, new_span))

    return result


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursive character chunker that splits text by separators in order of preference.

    This chunker attempts to split text using a hierarchy of separators:
    1. Double newlines (\n\n)
    2. Single newlines (\n)
    3. Spaces ( )
    4. Characters (none)

    It recursively applies separators until chunks fit within the specified size.
    """

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk document using recursive character splitting.

        Args:
            document: Document to chunk

        Returns:
            List of chunked documents
        """
        # Sanitize the input text
        sanitized_content = sanitize_text(
            document.content,
            max_chars=self.config.max_document_chars,
            strip_control_chars=self.config.strip_control_chars,
            normalize_newlines=self.config.normalize_newlines
        )

        # Use character counter for this implementation
        counter = CharCounter()
        
        chunks = []
        texts = [_Piece(text=sanitized_content, span=TextSpan(0, len(sanitized_content)))]

        # Process each separator in order of preference
        for separator in self.config.separators:
            new_texts = []

            for text_piece in texts:
                text = text_piece.text
                
                # If text is already small enough, keep it as is
                if counter.count(text) <= self.config.chunk_size:
                    new_texts.append(text_piece)
                    continue

                # Split the text using the current separator
                if separator == "":
                    # Character-level splitting
                    splits = [_Piece(text=text[i:i+self.config.chunk_size], 
                                    span=TextSpan(i, min(i+self.config.chunk_size, len(text)))) 
                             for i in range(0, len(text), self.config.chunk_size)]
                else:
                    # Separator-based splitting using our helper function
                    splits = _split_with_separator(text, separator, text_piece.span.start)

                # Process splits to ensure they fit within chunk size
                current_chunk = ""
                current_start = None
                current_end = None
                
                for split_piece in splits:
                    split_text = split_piece.text
                    
                    # Check if adding this split would exceed chunk size
                    test_chunk = current_chunk + separator + split_text if current_chunk else split_text

                    if counter.count(test_chunk) <= self.config.chunk_size:
                        if current_start is None:
                            current_start = split_piece.span.start
                        current_chunk = test_chunk
                        current_end = split_piece.span.end
                    else:
                        # If current chunk has content, save it
                        if current_chunk:
                            chunks.append(_Piece(current_chunk, TextSpan(current_start, current_end)))

                        # If the split itself is larger than chunk size,
                        # we need to further split it
                        if counter.count(split_text) > self.config.chunk_size:
                            # Recursively process this oversized split
                            sub_splits = self._split_large_text(split_text, split_piece.span.start)
                            new_texts.extend(sub_splits)
                        else:
                            current_chunk = split_text
                            current_start = split_piece.span.start
                            current_end = split_piece.span.end

                # Add any remaining content as a chunk
                if current_chunk:
                    new_texts.append(_Piece(current_chunk, TextSpan(current_start, current_end)))

            texts = new_texts

        # Apply overlap to the chunks
        texts_with_overlap = _apply_overlap_units(texts, counter, self.config.chunk_overlap, self.config.chunk_size)

        # Create document chunks
        for i, text_piece in enumerate(texts_with_overlap):
            chunk_span = ChunkSpan(text=text_piece.text, span=text_piece.span)
            chunk_doc = self._create_chunk_document(document, chunk_span, i)
            chunks.append(chunk_doc)

        return chunks

    def _split_large_text(self, text: str, base_offset: int) -> List[_Piece]:
        """
        Split a text that is larger than the chunk size.

        Args:
            text: Text to split
            base_offset: Starting offset in the original document

        Returns:
            List of text chunks with spans
        """
        chunks = []
        start = 0
        counter = CharCounter()

        while start < len(text):
            # Find the end position for this chunk
            end = start + self.config.chunk_size
            
            # Adjust end position to not exceed text length
            if end > len(text):
                end = len(text)
            
            chunk_text = text[start:end]
            chunk_start = base_offset + start
            chunk_end = base_offset + end
            
            chunks.append(_Piece(chunk_text, TextSpan(chunk_start, chunk_end)))
            
            # Move start position forward, accounting for overlap
            start = end - self.config.chunk_overlap
            if start < end and self.config.chunk_overlap > 0:
                # Ensure we don't get stuck in an infinite loop
                start = max(start, start + 1)

        return chunks