"""
Token-Aware Chunking Strategy

Implements precise token-based chunking using tiktoken.

Best for:
- When exact token counts are critical
- LLM context window management
- Cost estimation and optimization
- Multi-byte character handling (Unicode)

How it works:
1. Tokenize entire document using tiktoken
2. Create chunks at exact token boundaries
3. Attempt to end at sentence boundaries when possible
4. Decode tokens back to text

Example:
    >>> from src.rag.chunking import TokenAwareChunker, ChunkingConfig
    >>> config = ChunkingConfig(
    ...     chunk_size=512,  # Exact tokens
    ...     chunk_overlap=50,
    ...     tokenizer_name="cl100k_base",
    ... )
    >>> chunker = TokenAwareChunker(config)
    >>> chunks = chunker.chunk({"id": "doc1", "content": "Text..."})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseChunker, Chunk, ChunkingConfig, TokenCounter

logger = logging.getLogger(__name__)


class TokenAwareChunker(BaseChunker):
    """
    Token-accurate chunking using tiktoken.

    This strategy provides precise token counting and chunking,
    essential for:
    - Managing LLM context windows
    - Accurate cost estimation
    - Handling multi-byte characters correctly

    Attributes:
        config: Chunking configuration
        token_counter: Token counter utility

    Example:
        >>> chunker = TokenAwareChunker(
        ...     ChunkingConfig(
        ...         chunk_size=512,
        ...         tokenizer_name="cl100k_base"
        ...     )
        ... )
        >>> chunks = chunker.chunk({"id": "doc1", "content": "Text..."})
        >>> print(f"Each chunk has ~{chunker.config.chunk_size} tokens")
    """

    # Sentence ending token patterns (approximate)
    SENTENCE_ENDINGS = [". ", "! ", "? ", "\n\n", "\n"]

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
    ) -> None:
        """
        Initialize the token-aware chunker.

        Args:
            config: Chunking configuration
        """
        super().__init__(config)
        self._logger = logging.getLogger(self.__class__.__name__)

        # Ensure we have a tokenizer
        if not self.token_counter.is_available:
            self._logger.warning(
                "tiktoken not available. Token counts will be estimates."
            )

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Split document into token-accurate chunks.

        Args:
            document: Document dictionary with 'id' and 'content' fields

        Returns:
            List of Chunk objects with precise token counts

        Raises:
            ValueError: If document is invalid
        """
        self._validate_document(document)

        content = self._clean_text(document.get("content", ""))
        doc_id = document.get("id", "unknown")

        # Tokenize content
        tokens = self.token_counter.encode(content)

        if not tokens:
            self._logger.warning(
                f"Document {doc_id} has no tokens after encoding"
            )
            return []

        total_tokens = len(tokens)
        self._logger.debug(
            f"Tokenized document {doc_id}: {total_tokens} tokens"
        )

        # Calculate chunking parameters
        chunk_size = self.config.chunk_size
        overlap = min(self.config.chunk_overlap, chunk_size - 1)

        # Ensure minimum chunk size
        if chunk_size < self.config.min_chunk_size:
            self._logger.warning(
                f"Chunk size {chunk_size} below minimum, adjusting to "
                f"{self.config.min_chunk_size}"
            )
            chunk_size = self.config.min_chunk_size

        # Create chunks
        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < total_tokens:
            # Calculate end position
            end_idx = min(start_idx + chunk_size, total_tokens)

            # Try to end at sentence boundary
            if end_idx < total_tokens:
                adjusted_end = self._find_sentence_boundary(
                    tokens,
                    start_idx,
                    end_idx,
                )
                if adjusted_end > start_idx:
                    end_idx = adjusted_end

            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode to text
            chunk_text = self.token_counter.decode(chunk_tokens)

            if chunk_text.strip():
                # Calculate character indices
                char_start = self._estimate_char_position(
                    content,
                    tokens,
                    start_idx,
                )
                char_end = self._estimate_char_position(
                    content,
                    tokens,
                    end_idx,
                )

                chunk = self._create_chunk(
                    content=chunk_text,
                    document=document,
                    start_index=char_start,
                    end_index=char_end,
                    extra_metadata={
                        "token_count": len(chunk_tokens),
                        "chunk_index": chunk_index,
                        "is_first": chunk_index == 0,
                        "is_last": end_idx >= total_tokens,
                        "chunk_method_detail": "token_aware",
                        "tokenizer": self.config.tokenizer_name,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move to next chunk with overlap
            if end_idx >= total_tokens:
                break

            start_idx = end_idx - overlap

        self._logger.info(
            f"Created {len(chunks)} token-aware chunks from document {doc_id} "
            f"({total_tokens} total tokens)"
        )

        return chunks

    def _find_sentence_boundary(
        self,
        tokens: List[int],
        start_idx: int,
        end_idx: int,
    ) -> int:
        """
        Try to find a sentence boundary near the end position.

        Looks backwards from end_idx to find a natural break point.

        Args:
            tokens: Full token list
            start_idx: Start of chunk
            end_idx: Proposed end of chunk

        Returns:
            Adjusted end index at sentence boundary
        """
        # Decode a window around the end to find sentence endings
        window_start = max(start_idx, end_idx - 50)
        window_tokens = tokens[window_start:end_idx]
        window_text = self.token_counter.decode(window_tokens)

        # Look for sentence endings
        best_adjustment = 0

        for ending in self.SENTENCE_ENDINGS:
            pos = window_text.rfind(ending)
            if pos > len(window_text) // 2:  # In second half of window
                # Calculate token adjustment
                text_before_ending = window_text[:pos + len(ending)]
                adjustment_tokens = self.token_counter.count(text_before_ending)

                if adjustment_tokens > best_adjustment:
                    best_adjustment = adjustment_tokens

        if best_adjustment > 0:
            return window_start + best_adjustment

        return end_idx

    def _estimate_char_position(
        self,
        content: str,
        tokens: List[int],
        token_idx: int,
    ) -> int:
        """
        Estimate character position from token index.

        This is an approximation since token boundaries don't
        always align with character boundaries.

        Args:
            content: Original content
            tokens: Token list
            token_idx: Token index

        Returns:
            Estimated character position
        """
        if token_idx <= 0:
            return 0

        if token_idx >= len(tokens):
            return len(content)

        # Decode tokens up to position
        partial_tokens = tokens[:token_idx]
        partial_text = self.token_counter.decode(partial_tokens)

        # Find in original content
        pos = content.find(partial_text)
        if pos >= 0:
            return pos + len(partial_text)

        # Fallback: proportional estimate
        ratio = token_idx / max(len(tokens), 1)
        return int(len(content) * ratio)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Convenience method using this chunker's tokenizer.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        return self.token_counter.count(text)

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "...",
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        return self.token_counter.truncate(text, max_tokens, suffix)


def create_token_aware_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
    tokenizer_name: str = "cl100k_base",
    model_name: Optional[str] = None,
) -> TokenAwareChunker:
    """
    Factory function to create a TokenAwareChunker.

    Args:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chunk_size: Minimum acceptable chunk size
        max_chunk_size: Maximum acceptable chunk size
        tokenizer_name: tiktoken encoding name
        model_name: Model name for automatic encoding selection

    Returns:
        Configured TokenAwareChunker instance

    Example:
        >>> chunker = create_token_aware_chunker(
        ...     chunk_size=256,
        ...     tokenizer_name="p50k_base",  # For code
        ... )
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        tokenizer_name=tokenizer_name,
        model_name=model_name,
    )

    return TokenAwareChunker(config)


# Utility functions


def count_tokens(
    text: str,
    encoding_name: str = "cl100k_base",
) -> int:
    """
    Count tokens in text using specified encoding.

    Args:
        text: Text to count
        encoding_name: tiktoken encoding name

    Returns:
        Token count

    Example:
        >>> count = count_tokens("Hello, world!", "cl100k_base")
        >>> print(f"Token count: {count}")
    """
    counter = TokenCounter(tokenizer_name=encoding_name)
    return counter.count(text)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
    suffix: str = "...",
) -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        encoding_name: tiktoken encoding name
        suffix: Suffix to add if truncated

    Returns:
        Truncated text

    Example:
        >>> truncated = truncate_to_tokens(long_text, 512)
    """
    counter = TokenCounter(tokenizer_name=encoding_name)
    return counter.truncate(text, max_tokens, suffix)


def split_by_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
    overlap: int = 0,
) -> List[str]:
    """
    Split text into token-sized chunks.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        encoding_name: tiktoken encoding name
        overlap: Overlap between chunks

    Returns:
        List of text chunks

    Example:
        >>> chunks = split_by_tokens(long_text, 512, overlap=50)
    """
    counter = TokenCounter(tokenizer_name=encoding_name)
    tokens = counter.encode(text)

    if not tokens:
        return []

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = counter.decode(chunk_tokens)

        if chunk_text.strip():
            chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks
