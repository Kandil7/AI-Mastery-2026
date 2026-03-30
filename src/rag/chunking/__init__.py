"""
Chunking Module for RAG Systems

Production-ready text chunking strategies for Retrieval-Augmented Generation.

This module provides unified, well-tested chunking implementations:
- Fixed-size chunking (fast, predictable)
- Recursive chunking (structure-preserving, recommended default)
- Semantic chunking (embedding-based, high quality)
- Hierarchical chunking (parent-child relationships)
- Code-aware chunking (language-specific)
- Token-aware chunking (precise token counting)

Quick Start:
    >>> from src.rag.chunking import create_chunker, ChunkingStrategy
    >>>
    >>> # Simple usage
    >>> chunker = create_chunker("recursive", chunk_size=512)
    >>> chunks = chunker.chunk({
    ...     "id": "doc1",
    ...     "content": "Your document text here..."
    ... })
    >>>
    >>> # Using strategy enum
    >>> from src.rag.chunking import ChunkingStrategy
    >>> chunker = create_chunker(ChunkingStrategy.SEMANTIC)
    >>>
    >>> # Custom configuration
    >>> from src.rag.chunking import ChunkingConfig
    >>> config = ChunkingConfig(
    ...     strategy=ChunkingStrategy.RECURSIVE,
    ...     chunk_size=256,
    ...     chunk_overlap=25,
    ... )
    >>> chunker = create_chunker(config.strategy, config=config)

Module Structure:
    src/rag/chunking/
    ├── __init__.py      # This file - public API
    ├── base.py          # Base classes, Chunk, ChunkingConfig
    ├── fixed_size.py    # Fixed-size chunking
    ├── recursive.py     # Recursive character chunking
    ├── semantic.py      # Semantic/embedding chunking
    ├── hierarchical.py  # Parent-child chunking
    ├── code.py          # Code-aware chunking
    ├── token_aware.py   # Token-precise chunking
    └── factory.py       # ChunkerFactory

For detailed documentation, see README.md in this directory.
"""

from __future__ import annotations

# Base classes and types
from .base import (
    BaseChunker,
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    TokenCounter,
    generate_chunk_id,
    estimate_tokens_from_chars,
    is_arabic_text,
)

# Chunker implementations
from .fixed_size import FixedSizeChunker, create_fixed_chunker
from .recursive import RecursiveChunker, create_recursive_chunker
from .semantic import SemanticChunker, create_semantic_chunker
from .hierarchical import HierarchicalChunker, HierarchicalChunkResult, create_hierarchical_chunker
from .code import CodeChunker, create_code_chunker
from .token_aware import TokenAwareChunker, create_token_aware_chunker

# Factory
from .factory import (
    ChunkerFactory,
    create_chunker,
    get_recommended_config,
)

# Token utilities
from .token_aware import (
    count_tokens,
    truncate_to_tokens,
    split_by_tokens,
)

__all__ = [
    # Base classes
    "BaseChunker",
    "Chunk",
    "ChunkingConfig",
    "ChunkingStrategy",
    "TokenCounter",
    
    # Chunker implementations
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "HierarchicalChunkResult",
    "CodeChunker",
    "TokenAwareChunker",
    
    # Factory functions
    "ChunkerFactory",
    "create_chunker",
    "create_fixed_chunker",
    "create_recursive_chunker",
    "create_semantic_chunker",
    "create_hierarchical_chunker",
    "create_code_chunker",
    "create_token_aware_chunker",
    "get_recommended_config",
    
    # Utility functions
    "count_tokens",
    "truncate_to_tokens",
    "split_by_tokens",
    "generate_chunk_id",
    "estimate_tokens_from_chars",
    "is_arabic_text",
]

__version__ = "1.0.0"
__author__ = "AI-Mastery-2026 Team"
