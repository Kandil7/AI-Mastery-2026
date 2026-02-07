# src/chunking/strategies/code.py
from __future__ import annotations

import re
from typing import List

from src.retrieval import Document

from ..base import BaseChunker
from ..config import ChunkingConfig
from ..spans import ChunkSpan, TextSpan
from ..tokenizer import CharCounter
from ..sanitize import sanitize_text


class CodeChunker(BaseChunker):
    """
    Code-specific chunker that respects code structure and syntax.

    This chunker is designed for programming documents and attempts
    to maintain syntactic validity and logical code units.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        # Define code-specific separators
        self.code_separators = [
            r'\n\s*def\s+\w+',  # Function definitions
            r'\n\s*class\s+\w+',  # Class definitions
            r'\n\s*@',  # Decorators
            r'\n\s*if\s+.*:',  # If statements
            r'\n\s*for\s+.*:',  # For loops
            r'\n\s*while\s+.*:',  # While loops
            r'\n\s*try:',  # Try blocks
            r'\n\s*except\s*',  # Except blocks
            r'\n\s*else:',  # Else blocks
            r'\n\s*elif\s+.*:',  # Elif statements
            r'\n\s*import\s+',  # Import statements
            r'\n\s*from\s+\w+\s+import',  # From import statements
            r'\n\s*"""',  # Triple quotes (docstrings)
            r"\n\s*'''",  # Triple single quotes (docstrings)
            r'\n\s*#',  # Comments
            r'\n\s*\n',  # Blank lines
            r'\n',  # Newlines
        ]
        self.counter = CharCounter()

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk code document respecting code structure.

        Args:
            document: Code document to chunk

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
        
        content = sanitized_content
        chunks = []
        chunk_index = 0

        # Try to split by code structure
        current_position = 0
        current_chunk = ""
        current_start_pos = 0

        while current_position < len(content):
            # Find the next code structure boundary
            next_boundary = self._find_next_boundary(content, current_position)

            if next_boundary == -1:
                # No more boundaries, add remaining content
                remaining = content[current_position:]
                if self.counter.count(current_chunk) + self.counter.count(remaining) <= self.config.chunk_size:
                    current_chunk += remaining
                    if current_chunk.strip():
                        chunk_span = ChunkSpan(
                            text=current_chunk, 
                            span=TextSpan(current_start_pos, len(content))
                        )
                        chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index)
                        chunks.append(chunk_doc)
                    break
                else:
                    # Need to force split
                    forced_chunks = self._force_split(current_chunk + remaining, current_start_pos)
                    for i, forced_chunk_data in enumerate(forced_chunks):
                        chunk_span = ChunkSpan(
                            text=forced_chunk_data["text"],
                            span=forced_chunk_data["span"]
                        )
                        chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index + i)
                        chunks.append(chunk_doc)
                    break

            # Get the segment to add
            segment = content[current_position:next_boundary]

            # Check if adding this segment would exceed chunk size
            if self.counter.count(current_chunk) + self.counter.count(segment) <= self.config.chunk_size:
                if not current_chunk:  # First segment in this chunk
                    current_start_pos = current_position
                current_chunk += segment
                current_position = next_boundary
            else:
                # Current chunk is full, save it
                if current_chunk.strip():
                    chunk_span = ChunkSpan(
                        text=current_chunk, 
                        span=TextSpan(current_start_pos, current_position)
                    )
                    chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index)
                    chunks.append(chunk_doc)
                    chunk_index += 1

                # Start new chunk with the segment
                current_chunk = segment
                current_start_pos = current_position
                current_position = next_boundary

        # Add any remaining content
        if current_chunk.strip():
            chunk_span = ChunkSpan(
                text=current_chunk,
                span=TextSpan(current_start_pos, len(content))
            )
            chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index)
            chunks.append(chunk_doc)

        # Apply overlap to the chunks
        chunks = self._apply_overlap_to_code_chunks(chunks)

        return chunks

    def _find_next_boundary(self, content: str, start_pos: int) -> int:
        """Find the next code structure boundary."""
        min_pos = float('inf')

        for pattern in self.code_separators:
            match = re.search(pattern, content[start_pos:])
            if match:
                abs_pos = start_pos + match.start()
                if abs_pos < min_pos:
                    min_pos = abs_pos

        return int(min_pos) if min_pos != float('inf') else -1

    def _force_split(self, content: str, start_pos: int) -> List[dict]:
        """Force split content that doesn't have natural boundaries."""
        chunks = []
        current_pos = start_pos
        counter = CharCounter()

        while current_pos < start_pos + len(content):
            prev_pos = current_pos
            # Calculate how much we can take for this chunk
            remaining_content = content[current_pos - start_pos:]
            chunk_size = min(self.config.chunk_size, len(remaining_content))
            
            # Take the chunk
            chunk_text = remaining_content[:chunk_size]
            chunk_start = current_pos
            chunk_end = current_pos + len(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "span": TextSpan(chunk_start, chunk_end)
            })
            
            # Move to next position, accounting for overlap
            next_pos = chunk_end - self.config.chunk_overlap
            if next_pos <= prev_pos:
                next_pos = prev_pos + 1
            current_pos = next_pos

        return chunks
    
    def _apply_overlap_to_code_chunks(self, chunks: List[Document]) -> List[Document]:
        """Apply overlap to code chunks."""
        if not chunks or self.config.chunk_overlap <= 0:
            return chunks
        
        result = [chunks[0]]  # First chunk remains the same
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = result[-1]
            
            prev_content = previous_chunk.content
            overlap_size = min(self.config.chunk_overlap, len(prev_content))
            
            new_content = (
                prev_content[-overlap_size:] + current_chunk.content
                if overlap_size > 0
                else current_chunk.content
            )
            
            new_metadata = {
                **(getattr(current_chunk, "metadata", {}) or {}),
                "has_overlap": True,
                "overlap_from": getattr(previous_chunk, "id", None),
                "overlap_strategy": "code",
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
