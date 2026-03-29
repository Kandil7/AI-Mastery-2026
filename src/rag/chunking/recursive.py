"""
Recursive Chunking Strategy

Implements structure-preserving recursive character splitting.

Best for:
- Most general-purpose use cases (recommended default)
- Documents with clear hierarchical structure
- When preserving paragraphs and sentences is important

How it works:
1. Try splitting by largest separator (e.g., section breaks)
2. If chunks still too large, recurse with smaller separators
3. Merge small adjacent chunks
4. Split oversized chunks at word/character level

Example:
    >>> from src.rag.chunking import RecursiveChunker, ChunkingConfig
    >>> config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
    >>> chunker = RecursiveChunker(config)
    >>> chunks = chunker.chunk({"id": "doc1", "content": "Text with\n\nparagraphs..."})
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .base import BaseChunker, Chunk, ChunkingConfig, is_arabic_text

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """
    Recursive character-based chunking with hierarchical separators.

    This strategy splits text by trying increasingly fine-grained
    separators, preserving document structure where possible.

    Separator hierarchy (tried in order):
    1. Section breaks (\\n\\n\\n)
    2. Paragraph breaks (\\n\\n)
    3. Line breaks (\\n)
    4. Sentence endings (. ! ?)
    5. Word boundaries (space)
    6. Character level (last resort)

    Attributes:
        config: Chunking configuration
        separators: List of separators to try (auto-selected based on content)

    Example:
        >>> chunker = RecursiveChunker(
        ...     ChunkingConfig(chunk_size=512, chunk_overlap=50)
        ... )
        >>> doc = {"id": "doc_001", "content": "Long document..."}
        >>> chunks = chunker.chunk(doc)
    """

    # Default separator hierarchy for Latin scripts
    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Section breaks
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentence endings
        "! ",
        "? ",
        "; ",
        "  ",      # Double space
        " ",       # Word boundaries
        "",        # Character level (last resort)
    ]

    # Arabic-optimized separators
    ARABIC_SEPARATORS = [
        "\n\n\n",
        "\n\n",
        "\n",
        "۔ ",     # Arabic full stop
        "؛ ",     # Arabic semicolon
        "؟ ",     # Arabic question mark
        "﴿",     # Quranic verse start (preserved)
        "﴾",      # Quranic verse end
        " ",
        "",
    ]

    # Code-specific separators
    CODE_SEPARATORS = {
        "python": [
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\n@decorator",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "javascript": [
            "\nfunction ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            "\nclass ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "java": [
            "\npublic class ",
            "\nprivate class ",
            "\npublic ",
            "\nprivate ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    }

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the recursive chunker.

        Args:
            config: Chunking configuration
            separators: Custom separator list (auto-detected if not provided)
        """
        super().__init__(config)
        self._separators = separators
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_separators(self, content: str) -> List[str]:
        """
        Get appropriate separators based on content type.

        Args:
            content: Text content to analyze

        Returns:
            List of separators to use
        """
        if self._separators:
            return self._separators

        # Check for code
        if self.config.language and self.config.language.lower() != "auto":
            code_separators = self.CODE_SEPARATORS.get(self.config.language.lower())
            if code_separators:
                return code_separators

        # Check for Arabic text
        if is_arabic_text(content):
            return self.ARABIC_SEPARATORS

        return self.DEFAULT_SEPARATORS

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Split document using recursive character-based chunking.

        Args:
            document: Document dictionary with 'id' and 'content' fields

        Returns:
            List of Chunk objects preserving document structure

        Raises:
            ValueError: If document is invalid
        """
        self._validate_document(document)

        content = self._clean_text(document.get("content", ""))
        doc_id = document.get("id", "unknown")

        # Get appropriate separators
        separators = self._get_separators(content)

        self._logger.debug(
            f"Starting recursive chunking for {doc_id} "
            f"with {len(separators)} separator levels"
        )

        # Perform recursive splitting
        chunk_texts = self._recursive_split(
            text=content,
            separators=separators,
            start_offset=0,
        )

        # Convert to Chunk objects
        chunks = []
        current_offset = 0

        for chunk_text in chunk_texts:
            chunk = self._create_chunk(
                content=chunk_text,
                document=document,
                start_index=current_offset,
                end_index=current_offset + len(chunk_text),
                extra_metadata={
                    "chunk_method_detail": "recursive",
                    "separator_count": len(separators),
                },
            )
            chunks.append(chunk)
            current_offset += len(chunk_text)

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        self._logger.info(
            f"Created {len(chunks)} recursive chunks from document {doc_id} "
            f"({len(content)} characters)"
        )

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        start_offset: int,
    ) -> List[str]:
        """
        Recursively split text by separator hierarchy.

        Algorithm:
        1. Check if text fits in target chunk size
        2. Try splitting by current separator
        3. If chunks still too large, recurse with next separator
        4. Merge small adjacent chunks

        Args:
            text: Text to split
            separators: Remaining separators to try
            start_offset: Character offset in original document

        Returns:
            List of chunk texts
        """
        if not text.strip():
            return []

        # Check if text fits in one chunk
        text_tokens = self.token_counter.count(text)

        if text_tokens <= self.config.chunk_size:
            return [text]

        # No more separators, force split
        if not separators:
            return self._force_split(text, self.config.chunk_size)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        if separator:
            parts = self._split_keep_separator(text, separator)
        else:
            parts = [text]

        # If only one part, try next separator
        if len(parts) <= 1:
            if remaining_separators:
                return self._recursive_split(
                    text=text,
                    separators=remaining_separators,
                    start_offset=start_offset,
                )
            else:
                # No more separators, force split
                return self._force_split(text, self.config.chunk_size)

        # Process each part
        result = []

        for part in parts:
            if not part.strip():
                continue

            part_tokens = self.token_counter.count(part)

            # Recurse if still too large
            if part_tokens > self.config.chunk_size and remaining_separators:
                sub_chunks = self._recursive_split(
                    text=part,
                    separators=remaining_separators,
                    start_offset=start_offset,
                )
                result.extend(sub_chunks)
            else:
                result.append(part)

        return result

    def _split_keep_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text but keep separator with each chunk.

        This preserves important structural markers like newlines
        and punctuation.

        Args:
            text: Text to split
            separator: Separator string

        Returns:
            List of text parts with separators preserved
        """
        if not separator:
            return [text]

        parts = text.split(separator)
        result = []

        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                # Keep separator with this part
                result.append(part + separator)
            else:
                # Last part, no trailing separator
                result.append(part)

        return result

    def _force_split(self, text: str, max_tokens: int) -> List[str]:
        """
        Force split text when no separators work.

        Splits at exact token boundary, potentially mid-word.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk

        Returns:
            List of forced chunks
        """
        tokens = self.token_counter.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.token_counter.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text)

            start = end

        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge adjacent chunks that are too small.

        Combines chunks below min_chunk_size to avoid fragmentation.

        Args:
            chunks: List of chunks to potentially merge

        Returns:
            List of chunks with small ones merged
        """
        if len(chunks) <= 1:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            combined_tokens = (
                self.token_counter.count(current.content)
                + self.token_counter.count(next_chunk.content)
            )

            if combined_tokens <= self.config.chunk_size:
                # Merge chunks
                merged_content = current.content + "\n\n" + next_chunk.content

                current = self._create_chunk(
                    content=merged_content,
                    document={"id": current.document_id, "metadata": {}},
                    start_index=current.start_index,
                    end_index=next_chunk.end_index,
                    extra_metadata=current.metadata,
                )
            else:
                # Keep separate
                merged.append(current)
                current = next_chunk

        merged.append(current)

        if len(merged) != len(chunks):
            self._logger.debug(
                f"Merged {len(chunks)} chunks into {len(merged)} "
                f"(removed {len(chunks) - len(merged)} small chunks)"
            )

        return merged


def create_recursive_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
    separators: Optional[List[str]] = None,
    language: str = "auto",
) -> RecursiveChunker:
    """
    Factory function to create a RecursiveChunker.

    Args:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chunk_size: Minimum acceptable chunk size
        max_chunk_size: Maximum acceptable chunk size
        separators: Custom separator list (auto-detected if None)
        language: Programming language for code (or 'auto' for detection)

    Returns:
        Configured RecursiveChunker instance

    Example:
        >>> chunker = create_recursive_chunker(
        ...     chunk_size=256,
        ...     separators=["\n\n", "\n", ". ", " "]
        ... )
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        language=language,
    )

    return RecursiveChunker(config, separators=separators)
