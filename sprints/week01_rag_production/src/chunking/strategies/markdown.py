# src/chunking/strategies/markdown.py
from __future__ import annotations

import re
from typing import List

from src.retrieval import Document

from ..base import BaseChunker
from ..config import ChunkingConfig
from ..spans import ChunkSpan, TextSpan
from ..tokenizer import CharCounter
from ..sanitize import sanitize_text


class MarkdownChunker(BaseChunker):
    """
    Markdown-aware chunker that preserves document structure.

    This chunker understands markdown syntax and attempts to maintain
    structural integrity while chunking.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        # Define markdown-specific separators
        self.md_separators = [
            r'\n#{1,6}\s',  # Headers (h1-h6)
            r'\n\s*-\s',  # Unordered list items
            r'\n\s*\d+\.\s',  # Ordered list items
            r'\n\s*>',  # Blockquotes
            r'\n\s*```',  # Code blocks
            r'\n\s*~~~',  # Alternative code blocks
            r'\n\s*\|\s',  # Table rows
            r'\n\s*---',  # Horizontal rules
            r'\n\s*\*\*\*',  # Horizontal rules with asterisks
            r'\n\s*___',  # Horizontal rules with underscores
            r'\n\s*\n',  # Paragraph breaks
            r'\n',  # Line breaks
        ]
        self.counter = CharCounter()

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk markdown document preserving structure.

        Args:
            document: Markdown document to chunk

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

        # Split by markdown structure
        current_position = 0
        current_chunk = ""
        current_start_pos = 0

        while current_position < len(content):
            # Find the next markdown structure boundary
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
        chunks = self._apply_overlap_to_md_chunks(chunks)

        return chunks

    def _find_next_boundary(self, content: str, start_pos: int) -> int:
        """Find the next markdown structure boundary."""
        min_pos = float('inf')

        for pattern in self.md_separators:
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
            current_pos = chunk_end - self.config.chunk_overlap
            if current_pos <= chunk_end and self.config.chunk_overlap > 0:
                # Ensure we don't get stuck in an infinite loop
                current_pos = max(current_pos, chunk_end + 1)

        return chunks
    
    def _apply_overlap_to_md_chunks(self, chunks: List[Document]) -> List[Document]:
        """Apply overlap to markdown chunks."""
        if not chunks or self.config.chunk_overlap <= 0:
            return chunks
        
        result = [chunks[0]]  # First chunk remains the same
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = result[-1]  # Use the last chunk in result (which may have overlap)
            
            # Extract the overlap from the previous chunk
            prev_content = previous_chunk.content
            prev_tokens = len(prev_content)  # Using character count as tokens for simplicity
            overlap_size = min(self.config.chunk_overlap, prev_tokens)
            
            if overlap_size > 0:
                overlap_text = prev_content[-overlap_size:]
                new_content = overlap_text + current_chunk.content
            else:
                new_content = current_chunk.content
            
            # Create a new document with the overlapped content
            new_doc = Document(
                id=current_chunk.id,
                content=new_content,
                source=current_chunk.source,
                doc_type=current_chunk.doc_type,
                metadata={
                    **current_chunk.metadata,
                    "has_overlap": True,
                    "overlap_from": previous_chunk.id,
                    "overlap_strategy": "markdown"
                },
                created_at=current_chunk.created_at,
                updated_at=current_chunk.updated_at,
                access_control=current_chunk.access_control,
                page_number=current_chunk.page_number,
                section_title=current_chunk.section_title
            )
            
            result.append(new_doc)
        
        return result