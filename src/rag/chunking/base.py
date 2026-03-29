"""
Base Module for Chunking Strategies

This module provides the foundational abstractions for all chunking strategies:
- BaseChunker: Abstract base class defining the chunking interface
- Chunk: Data class representing a text chunk with metadata
- ChunkingConfig: Configuration dataclass for chunking parameters
- TokenCounter: Utility for accurate token counting

Following production best practices:
- Comprehensive type hints
- Google-style docstrings
- Robust error handling
- Logging throughout
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class ChunkingStrategy(Enum):
    """
    Available chunking strategies.

    Attributes:
        FIXED: Fast, predictable fixed-size chunking
        RECURSIVE: Structure-preserving recursive chunking (recommended default)
        SEMANTIC: Context-aware embedding-based chunking
        HIERARCHICAL: Parent-child relationship chunking
        CODE: Language-aware code splitting
        TOKEN_AWARE: Token-accurate chunking with tiktoken
    """

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    CODE = "code"
    TOKEN_AWARE = "token_aware"


# ==================== Data Classes ====================


@dataclass
class Chunk:
    """
    Represents a chunk of text with comprehensive metadata.

    This dataclass encapsulates a single text chunk along with all
    necessary metadata for retrieval, ranking, and citation purposes.

    Attributes:
        content: The actual text content of the chunk
        chunk_id: Unique identifier for this chunk (auto-generated if not provided)
        document_id: ID of the source document
        start_index: Start character index in original document
        end_index: End character index in original document
        metadata: Additional metadata (source, page numbers, etc.)
        parent_id: Optional ID of parent chunk (for hierarchical chunking)
        tokens: Optional pre-computed token IDs
        embedding: Optional pre-computed embedding vector

    Example:
        >>> chunk = Chunk(
        ...     content="Machine learning is a subset of AI.",
        ...     document_id="doc_001",
        ...     start_index=0,
        ...     end_index=42,
        ...     metadata={"source": "intro.pdf", "page": 1}
        ... )
        >>> print(chunk.chunk_id)
        'doc_001_chunk_a1b2c3d4'
        >>> print(chunk.word_count)
        8
    """

    content: str
    document_id: str
    start_index: int = 0
    end_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    parent_id: Optional[str] = None
    tokens: Optional[List[int]] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

        # Ensure end_index is set
        if self.end_index == 0 and self.content:
            self.end_index = self.start_index + len(self.content)

    def _generate_id(self) -> str:
        """
        Generate a unique chunk ID based on content hash.

        Returns:
            A unique chunk ID in format '{document_id}_chunk_{hash}'
        """
        content_hash = hashlib.md5(self.content.encode("utf-8")).hexdigest()[:8]
        return f"{self.document_id}_chunk_{content_hash}"

    @property
    def word_count(self) -> int:
        """Get the number of words in the chunk."""
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Get the number of characters in the chunk."""
        return len(self.content)

    @property
    def is_empty(self) -> bool:
        """Check if chunk content is empty or whitespace only."""
        return not self.content.strip()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary representation.

        Returns:
            Dictionary with all chunk attributes
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """
        Create a Chunk instance from dictionary.

        Args:
            data: Dictionary containing chunk data

        Returns:
            Chunk instance
        """
        return cls(
            content=data.get("content", ""),
            document_id=data.get("document_id", ""),
            start_index=data.get("start_index", 0),
            end_index=data.get("end_index", 0),
            metadata=data.get("metadata", {}),
            chunk_id=data.get("chunk_id", ""),
            parent_id=data.get("parent_id"),
        )

    def __len__(self) -> int:
        """Return the character count of the chunk."""
        return self.char_count

    def __str__(self) -> str:
        """Return a string representation of the chunk."""
        preview = self.content[:50].replace("\n", " ")
        if len(self.content) > 50:
            preview += "..."
        return f"Chunk(id={self.chunk_id}, preview='{preview}')"

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"Chunk(chunk_id='{self.chunk_id}', "
            f"document_id='{self.document_id}', "
            f"word_count={self.word_count})"
        )


@dataclass
class ChunkingConfig:
    """
    Configuration for chunking operations.

    This dataclass centralizes all configuration parameters for
    chunking strategies, providing sensible defaults while allowing
    fine-grained customization.

    Attributes:
        strategy: The chunking strategy to use
        chunk_size: Target chunk size (tokens or characters depending on strategy)
        chunk_overlap: Overlap between consecutive chunks
        min_chunk_size: Minimum acceptable chunk size
        max_chunk_size: Maximum acceptable chunk size
        keep_separator: Whether to keep separators in chunks
        add_start_index: Whether to track start indices in original text

        # Token-specific
        tokenizer_name: Name of tokenizer to use (e.g., 'cl100k_base')
        model_name: Model name for tokenizer inference

        # Semantic-specific
        embedding_model: Model name for semantic chunking
        similarity_threshold: Threshold for semantic boundary detection
        buffer_size: Number of sentences to buffer for semantic analysis

        # Hierarchical-specific
        parent_chunk_size: Size of parent chunks in hierarchical chunking
        child_chunk_size: Size of child chunks in hierarchical chunking

        # Code-specific
        language: Programming language for code chunking

        # Arabic/Islamic text
        preserve_arabic_marks: Whether to preserve Arabic diacritics
        preserve_verses: Whether to preserve Quranic verse boundaries
        preserve_hadith: Whether to preserve Hadith unit boundaries

    Example:
        >>> config = ChunkingConfig(
        ...     strategy=ChunkingStrategy.RECURSIVE,
        ...     chunk_size=512,
        ...     chunk_overlap=50,
        ... )
        >>> print(config)
        ChunkingConfig(strategy=recursive, chunk_size=512, overlap=50)
    """

    # Core settings
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

    # Behavior flags
    keep_separator: bool = False
    add_start_index: bool = True

    # Tokenizer settings
    tokenizer_name: str = "cl100k_base"
    model_name: str = "gpt-4"

    # Semantic chunking
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    similarity_threshold: float = 0.5
    buffer_size: int = 1

    # Hierarchical chunking
    parent_chunk_size: int = 2000
    child_chunk_size: int = 500

    # Code chunking
    language: str = "python"

    # Arabic/Islamic text
    preserve_arabic_marks: bool = True
    preserve_verses: bool = True
    preserve_hadith: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "strategy": self.strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "keep_separator": self.keep_separator,
            "tokenizer_name": self.tokenizer_name,
            "embedding_model": self.embedding_model,
            "similarity_threshold": self.similarity_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingConfig":
        """Create config from dictionary."""
        if "strategy" in data and isinstance(data["strategy"], str):
            data["strategy"] = ChunkingStrategy(data["strategy"])
        return cls(**data)


# ==================== Token Counter ====================


class TokenCounter:
    """
    Utility class for accurate token counting.

    Provides tokenization support using tiktoken when available,
    with fallback to character/word-based estimation.

    Attributes:
        tokenizer_name: Name of the tiktoken encoding to use
        model_name: Model name for automatic encoding selection

    Example:
        >>> counter = TokenCounter(model_name="gpt-4")
        >>> text = "Hello, world!"
        >>> count = counter.count(text)
        >>> print(f"Token count: {count}")
        Token count: 4
    """

    def __init__(
        self,
        tokenizer_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the token counter.

        Args:
            tokenizer_name: Specific tiktoken encoding name
            model_name: Model name for automatic encoding selection
        """
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self._tokenizer = None
        self._load_error: Optional[str] = None

    def _load_tokenizer(self) -> None:
        """
        Lazily load the tokenizer.

        Attempts to load tiktoken with the specified encoding or model.
        Sets _load_error if loading fails.
        """
        if self._tokenizer is not None:
            return

        try:
            import tiktoken

            if self.tokenizer_name:
                self._tokenizer = tiktoken.get_encoding(self.tokenizer_name)
            elif self.model_name:
                self._tokenizer = tiktoken.encoding_for_model(self.model_name)
            else:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

            logger.debug(f"Loaded tokenizer: {self.tokenizer_name or self.model_name}")

        except ImportError:
            self._load_error = "tiktoken not installed"
            logger.warning("tiktoken not installed. Using character-based estimation.")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Failed to load tokenizer: {e}. Using fallback.")

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens (or character estimate if tokenizer unavailable)
        """
        self._load_tokenizer()

        if self._tokenizer:
            return len(self._tokenizer.encode(text))

        # Fallback: word-based estimation
        return len(text.split())

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs (or character ordinals if tokenizer unavailable)

        Raises:
            ValueError: If text is empty
        """
        if not text:
            raise ValueError("Cannot encode empty text")

        self._load_tokenizer()

        if self._tokenizer:
            return self._tokenizer.encode(text)

        # Fallback: character ordinals
        return [ord(c) for c in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        self._load_tokenizer()

        if self._tokenizer:
            return self._tokenizer.decode(tokens)

        # Fallback: character ordinals
        return "".join(chr(t) for t in tokens if 0 <= t <= 0x10FFFF)

    def truncate(
        self,
        text: str,
        max_tokens: int,
        suffix: str = "...",
    ) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            suffix: Suffix to add if truncation occurs

        Returns:
            Truncated text
        """
        self._load_tokenizer()

        if not text:
            return text

        if self._tokenizer:
            tokens = self._tokenizer.encode(text)

            if len(tokens) <= max_tokens:
                return text

            # Reserve space for suffix
            suffix_tokens = self._tokenizer.encode(suffix)
            available_tokens = max_tokens - len(suffix_tokens)

            truncated_tokens = tokens[:available_tokens]
            return self._tokenizer.decode(truncated_tokens) + suffix

        # Fallback: character-based truncation
        if len(text) <= max_tokens * 4:  # Rough estimate
            return text

        return text[: max_tokens * 4 - len(suffix)] + suffix

    @property
    def is_available(self) -> bool:
        """Check if tiktoken tokenizer is available."""
        self._load_tokenizer()
        return self._tokenizer is not None


# ==================== Base Chunker ====================


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.

    This class defines the interface that all chunker implementations
    must follow. It provides common utilities and enforces consistent
    behavior across different strategies.

    Attributes:
        config: Chunking configuration
        token_counter: Token counter utility

    Example:
        >>> class MyChunker(BaseChunker):
        ...     def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        ...         # Implementation here
        ...         pass
        ...
        >>> config = ChunkingConfig(chunk_size=512)
        >>> chunker = MyChunker(config)
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        """
        Initialize the base chunker.

        Args:
            config: Chunking configuration (uses defaults if not provided)
        """
        self.config = config or ChunkingConfig()
        self.token_counter = TokenCounter(
            tokenizer_name=self.config.tokenizer_name,
            model_name=self.config.model_name,
        )
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Split a document into chunks.

        This is the main entry point for chunking. Implementations
        should split the document content according to their strategy
        and return a list of Chunk objects with appropriate metadata.

        Args:
            document: Document dictionary containing:
                - id: Document identifier
                - content: Text content to chunk
                - metadata: Optional document metadata

        Returns:
            List of Chunk objects

        Raises:
            ValueError: If document is invalid or content is empty
        """
        pass

    def chunk_texts(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[List[Chunk]]:
        """
        Chunk multiple texts.

        Convenience method for processing multiple documents.

        Args:
            texts: List of text strings to chunk
            doc_ids: Optional list of document IDs (auto-generated if not provided)
            base_metadata: Optional base metadata to apply to all documents

        Returns:
            List of lists, where each inner list contains chunks for one text
        """
        all_chunks = []

        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i:04d}"

            document = {
                "id": doc_id,
                "content": text,
                "metadata": base_metadata or {},
            }

            chunks = self.chunk(document)
            all_chunks.append(chunks)

        self._logger.info(f"Chunked {len(texts)} texts into {sum(len(c) for c in all_chunks)} total chunks")
        return all_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split text into string chunks (without metadata).

        This is a convenience method for simple text splitting without
        the full Chunk object overhead. Calls chunk() internally and
        extracts just the content.

        Args:
            text: Text to split

        Returns:
            List of text strings
        """
        document = {
            "id": "temp",
            "content": text,
            "metadata": {},
        }

        chunks = self.chunk(document)
        return [chunk.content for chunk in chunks]

    def _create_chunk(
        self,
        content: str,
        document: Dict[str, Any],
        start_index: int,
        end_index: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> Chunk:
        """
        Create a Chunk object with standardized metadata.

        This helper method ensures consistent metadata structure
        across all chunker implementations.

        Args:
            content: The chunk's text content
            document: Source document dictionary
            start_index: Start index in original document
            end_index: End index in original document
            extra_metadata: Additional metadata to include
            parent_id: Optional parent chunk ID

        Returns:
            Fully populated Chunk object
        """
        # Build base metadata
        metadata = {
            "document_id": document.get("id", "unknown"),
            "source": document.get("source", "unknown"),
            "source_type": document.get("source_type", "unknown"),
            "chunk_method": self.config.strategy.value,
            "word_count": len(content.split()),
            "char_count": len(content),
            "token_count": self.token_counter.count(content),
            **(extra_metadata or {}),
        }

        # Merge document metadata with prefix
        doc_metadata = document.get("metadata", {})
        if doc_metadata:
            for key, value in doc_metadata.items():
                metadata[f"doc_{key}"] = value

        return Chunk(
            content=content,
            chunk_id="",  # Auto-generated
            document_id=document.get("id", "unknown"),
            start_index=start_index,
            end_index=end_index,
            metadata=metadata,
            parent_id=parent_id,
        )

    def _validate_document(self, document: Dict[str, Any]) -> None:
        """
        Validate document structure.

        Args:
            document: Document to validate

        Raises:
            ValueError: If document is invalid
        """
        if not isinstance(document, dict):
            raise ValueError("Document must be a dictionary")

        if "content" not in document:
            raise ValueError("Document must have 'content' field")

        content = document.get("content", "")
        if not content or not isinstance(content, str):
            raise ValueError("Document content must be a non-empty string")

        if not content.strip():
            raise ValueError("Document content cannot be empty or whitespace only")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text before chunking.

        Performs common text normalization:
        - Remove excessive whitespace
        - Normalize line endings
        - Remove control characters

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace (3+ newlines -> 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove control characters (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Strip leading/trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def _merge_overlaps(
        self,
        chunks: List[str],
        overlap_size: int,
    ) -> List[str]:
        """
        Add overlap between consecutive chunks.

        Args:
            chunks: List of chunk texts
            overlap_size: Number of characters to overlap

        Returns:
            List of chunks with overlaps added
        """
        if not chunks or overlap_size <= 0:
            return chunks

        result = []

        for i, chunk in enumerate(chunks):
            if i > 0 and result:
                # Get overlap from previous chunk
                prev_chunk = result[-1]
                overlap_text = prev_chunk[-overlap_size:]

                # Find where overlap ends in current chunk
                if overlap_text in chunk:
                    overlap_end = chunk.find(overlap_text) + len(overlap_text)
                    chunk = chunk[overlap_end:]

            result.append(chunk)

        return result


# ==================== Utility Functions ====================


def generate_chunk_id(document_id: str, content: str, index: int = 0) -> str:
    """
    Generate a unique chunk ID.

    Args:
        document_id: Parent document ID
        content: Chunk content for hashing
        index: Optional index for uniqueness

    Returns:
        Unique chunk ID
    """
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    return f"{document_id}_chunk_{index:04d}_{content_hash}"


def estimate_tokens_from_chars(char_count: int) -> int:
    """
    Estimate token count from character count.

    Rough approximation: ~4 characters per token for English text.

    Args:
        char_count: Number of characters

    Returns:
        Estimated token count
    """
    return max(1, char_count // 4)


def is_arabic_text(text: str) -> bool:
    """
    Check if text is primarily Arabic.

    Args:
        text: Text to analyze

    Returns:
        True if text is primarily Arabic (>50% Arabic characters)
    """
    if not text:
        return False

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    return arabic_chars / max(len(text), 1) > 0.5
