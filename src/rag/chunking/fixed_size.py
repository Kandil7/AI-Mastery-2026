"""
Fixed-Size Chunking Strategy

Implements simple, fast, predictable fixed-size chunking.

Best for:
- Speed-critical applications
- Documents with uniform structure
- When token budget predictability is essential

Limitations:
- May break sentences and semantic units
- No structure awareness
- Can split mid-context

Example:
    >>> from src.rag.chunking import FixedSizeChunker, ChunkingConfig
    >>> config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
    >>> chunker = FixedSizeChunker(config)
    >>> chunks = chunker.chunk({"id": "doc1", "content": "Long text here..."})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseChunker, Chunk, ChunkingConfig

logger = logging.getLogger(__name__)


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking implementation.

    Splits text into chunks of approximately equal size based on
    token count (or character count if tokenizer unavailable).

    This is the simplest and fastest chunking strategy, making it
    ideal for applications where speed and predictability are more
    important than semantic coherence.

    Attributes:
        config: Chunking configuration

    Example:
        >>> config = ChunkingConfig(
        ...     chunk_size=512,
        ...     chunk_overlap=50,
        ... )
        >>> chunker = FixedSizeChunker(config)
        >>> doc = {"id": "doc_001", "content": "Your text here..."}
        >>> chunks = chunker.chunk(doc)
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        """
        Initialize the fixed-size chunker.

        Args:
            config: Chunking configuration. If not provided, uses defaults.
        """
        super().__init__(config)
        self._logger = logging.getLogger(self.__class__.__name__)

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Split document into fixed-size chunks.

        Algorithm:
        1. Tokenize the document content
        2. Create chunks of fixed token size
        3. Apply overlap between consecutive chunks
        4. Decode tokens back to text

        Args:
            document: Document dictionary with 'id' and 'content' fields

        Returns:
            List of Chunk objects with consistent sizes

        Raises:
            ValueError: If document is invalid
        """
        self._validate_document(document)

        content = self._clean_text(document.get("content", ""))
        doc_id = document.get("id", "unknown")

        # Tokenize content
        tokens = self.token_counter.encode(content)

        if not tokens:
            self._logger.warning(f"Document {doc_id} has no tokens after encoding")
            return []

        # Calculate chunking parameters
        chunk_size = self.config.chunk_size
        overlap = min(self.config.chunk_overlap, chunk_size - 1)

        # Ensure minimum chunk size
        if chunk_size < self.config.min_chunk_size:
            self._logger.warning(
                f"Chunk size {chunk_size} below minimum {self.config.min_chunk_size}, "
                f"adjusting to minimum"
            )
            chunk_size = self.config.min_chunk_size

        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(tokens):
            # Calculate end position
            end_idx = min(start_idx + chunk_size, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode to text
            chunk_text = self.token_counter.decode(chunk_tokens)

            if chunk_text.strip():
                # Calculate character indices for metadata
                char_start = self._token_to_char_index(content, tokens, start_idx)
                char_end = self._token_to_char_index(content, tokens, end_idx)

                chunk = self._create_chunk(
                    content=chunk_text,
                    document=document,
                    start_index=char_start,
                    end_index=char_end,
                    extra_metadata={
                        "token_count": len(chunk_tokens),
                        "chunk_index": chunk_index,
                        "is_first": chunk_index == 0,
                        "is_last": end_idx >= len(tokens),
                    },
                )

                chunks.append(chunk)
                chunk_index += 1

            # Move to next chunk with overlap
            if end_idx >= len(tokens):
                break

            start_idx = end_idx - overlap

        self._logger.info(
            f"Created {len(chunks)} fixed-size chunks from document {doc_id} "
            f"({len(tokens)} tokens total)"
        )

        return chunks

    def _token_to_char_index(
        self,
        content: str,
        tokens: List[int],
        token_index: int,
    ) -> int:
        """
        Convert token index to character index in original text.

        This is an approximation since token boundaries don't always
        align with character boundaries.

        Args:
            content: Original text content
            tokens: List of token IDs
            token_index: Index of token to convert

        Returns:
            Approximate character index
        """
        if token_index <= 0:
            return 0

        if token_index >= len(tokens):
            return len(content)

        # Decode tokens up to the target index
        partial_tokens = tokens[:token_index]
        partial_text = self.token_counter.decode(partial_tokens)

        # Find position in original content
        pos = content.find(partial_text)
        if pos >= 0:
            return pos + len(partial_text)

        # Fallback: estimate based on ratio
        ratio = token_index / max(len(tokens), 1)
        return int(len(content) * ratio)


def create_fixed_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
    tokenizer_name: str = "cl100k_base",
) -> FixedSizeChunker:
    """
    Factory function to create a FixedSizeChunker.

    Args:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chunk_size: Minimum acceptable chunk size
        max_chunk_size: Maximum acceptable chunk size
        tokenizer_name: Name of tokenizer encoding to use

    Returns:
        Configured FixedSizeChunker instance

    Example:
        >>> chunker = create_fixed_chunker(chunk_size=256, chunk_overlap=25)
        >>> chunks = chunker.chunk({"id": "doc1", "content": "Text..."})
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        tokenizer_name=tokenizer_name,
    )

    return FixedSizeChunker(config)
