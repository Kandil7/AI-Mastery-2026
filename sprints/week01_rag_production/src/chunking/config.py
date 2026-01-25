# src/chunking/config.py
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional


class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CODE = "code"
    MARKDOWN = "markdown"
    CHARACTER = "character"
    AUTO = "auto"


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Production chunking config.

    Notes:
      - chunk_size and overlap are interpreted in "units" of the tokenizer/counter in use.
      - For character fallback, these become characters.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 50

    # Recursive splitting hierarchy (best -> worst boundaries)
    separators: Optional[List[str]] = None

    # Default strategy preference (AUTO recommended)
    strategy: ChunkingStrategy = ChunkingStrategy.AUTO

    # Safety/limits
    max_document_chars: int = 2_000_000  # cap to protect memory
    strip_control_chars: bool = True
    normalize_newlines: bool = True

    # Structure hints
    preserve_structure: bool = True

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.max_document_chars <= 0:
            raise ValueError("max_document_chars must be positive")

        if self.separators is None:
            object.__setattr__(self, "separators", ["\n\n", "\n", " ", ""])

    def copy_with(self, **kwargs) -> "ChunkingConfig":
        return replace(self, **kwargs)