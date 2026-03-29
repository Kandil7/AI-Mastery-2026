"""
Chunker Factory Module

Provides unified factory for creating chunker instances.

Example:
    >>> from src.rag.chunking import ChunkerFactory, ChunkingStrategy
    >>> chunker = ChunkerFactory.create(ChunkingStrategy.RECURSIVE)
    >>> chunks = chunker.chunk({"id": "doc1", "content": "Text..."})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class ChunkerFactory:
    """
    Factory for creating chunker instances.

    Provides a centralized way to create chunkers with consistent
    configuration and proper initialization.

    Example:
        >>> # Create using strategy enum
        >>> chunker = ChunkerFactory.create(ChunkingStrategy.RECURSIVE)
        >>>
        >>> # Create using string name
        >>> chunker = ChunkerFactory.create("semantic")
        >>>
        >>> # Create with custom config
        >>> config = ChunkingConfig(chunk_size=256)
        >>> chunker = ChunkerFactory.create("recursive", config=config)
    """

    # Registry of available chunkers
    _chunker_registry: Dict[str, Type[BaseChunker]] = {}

    @classmethod
    def register_chunker(
        cls,
        strategy: str,
        chunker_class: Type[BaseChunker],
    ) -> None:
        """
        Register a new chunker type.

        Args:
            strategy: Strategy name
            chunker_class: Chunker class to register
        """
        cls._chunker_registry[strategy.lower()] = chunker_class
        logger.debug(f"Registered chunker: {strategy}")

    @classmethod
    def create(
        cls,
        strategy: str | ChunkingStrategy,
        config: Optional[ChunkingConfig] = None,
        **kwargs: Any,
    ) -> BaseChunker:
        """
        Create a chunker instance.

        Args:
            strategy: Chunking strategy (enum or string)
            config: Optional configuration (created from kwargs if not provided)
            **kwargs: Additional configuration parameters

        Returns:
            Configured chunker instance

        Raises:
            ValueError: If strategy is unknown

        Example:
            >>> chunker = ChunkerFactory.create(
            ...     "recursive",
            ...     chunk_size=512,
            ...     chunk_overlap=50
            ... )
        """
        # Convert strategy to string
        if isinstance(strategy, ChunkingStrategy):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy).lower()

        # Create config if not provided
        if config is None:
            config = ChunkingConfig(**kwargs)

        # Get chunker class
        chunker_class = cls._get_chunker_class(strategy_name)

        logger.info(f"Creating {strategy_name} chunker")

        return chunker_class(config, **kwargs)

    @classmethod
    def _get_chunker_class(
        cls,
        strategy_name: str,
    ) -> Type[BaseChunker]:
        """
        Get chunker class for strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Chunker class

        Raises:
            ValueError: If strategy is unknown
        """
        # Lazy load chunkers on first use
        if not cls._chunker_registry:
            cls._load_default_chunkers()

        chunker_class = cls._chunker_registry.get(strategy_name)

        if not chunker_class:
            available = list(cls._chunker_registry.keys())
            raise ValueError(
                f"Unknown chunking strategy: {strategy_name}. "
                f"Available strategies: {available}"
            )

        return chunker_class

    @classmethod
    def _load_default_chunkers(cls) -> None:
        """Load default chunker implementations."""
        from .fixed_size import FixedSizeChunker
        from .recursive import RecursiveChunker
        from .semantic import SemanticChunker
        from .hierarchical import HierarchicalChunker
        from .code import CodeChunker
        from .token_aware import TokenAwareChunker

        cls._chunker_registry = {
            ChunkingStrategy.FIXED.value: FixedSizeChunker,
            ChunkingStrategy.RECURSIVE.value: RecursiveChunker,
            ChunkingStrategy.SEMANTIC.value: SemanticChunker,
            ChunkingStrategy.HIERARCHICAL.value: HierarchicalChunker,
            ChunkingStrategy.CODE.value: CodeChunker,
            ChunkingStrategy.TOKEN_AWARE.value: TokenAwareChunker,
        }

        logger.debug("Loaded default chunker implementations")

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """
        Get list of available chunking strategies.

        Returns:
            List of strategy names
        """
        if not cls._chunker_registry:
            cls._load_default_chunkers()

        return list(cls._chunker_registry.keys())

    @classmethod
    def get_recommended_strategy(cls, content_type: str) -> ChunkingStrategy:
        """
        Get recommended chunking strategy for content type.

        Args:
            content_type: Type of content (e.g., 'code', 'documentation', 'legal')

        Returns:
            Recommended strategy

        Example:
            >>> strategy = ChunkerFactory.get_recommended_strategy("code")
            >>> print(strategy)
            ChunkingStrategy.CODE
        """
        recommendations = {
            "code": ChunkingStrategy.CODE,
            "programming": ChunkingStrategy.CODE,
            "documentation": ChunkingStrategy.RECURSIVE,
            "articles": ChunkingStrategy.RECURSIVE,
            "legal": ChunkingStrategy.SEMANTIC,
            "contracts": ChunkingStrategy.SEMANTIC,
            "research": ChunkingStrategy.SEMANTIC,
            "books": ChunkingStrategy.HIERARCHICAL,
            "long-form": ChunkingStrategy.HIERARCHICAL,
            "api-docs": ChunkingStrategy.RECURSIVE,
            "technical": ChunkingStrategy.RECURSIVE,
        }

        content_type_lower = content_type.lower()

        for key, strategy in recommendations.items():
            if key in content_type_lower:
                return strategy

        # Default to recursive for unknown types
        return ChunkingStrategy.RECURSIVE


def create_chunker(
    strategy: str | ChunkingStrategy = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs: Any,
) -> BaseChunker:
    """
    Convenience function to create a chunker.

    Args:
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        **kwargs: Additional configuration

    Returns:
        Configured chunker instance

    Example:
        >>> chunker = create_chunker("semantic", chunk_size=768)
    """
    return ChunkerFactory.create(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )


def get_recommended_config(content_type: str, **overrides: Any) -> ChunkingConfig:
    """
    Get recommended configuration for content type.

    Args:
        content_type: Type of content
        **overrides: Configuration overrides

    Returns:
        Recommended configuration

    Example:
        >>> config = get_recommended_config("code", chunk_size=1000)
    """
    strategy = ChunkerFactory.get_recommended_strategy(content_type)

    # Default configs by type
    defaults = {
        ChunkingStrategy.CODE: {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "language": "auto",
        },
        ChunkingStrategy.SEMANTIC: {
            "chunk_size": 768,
            "chunk_overlap": 50,
            "similarity_threshold": 0.5,
        },
        ChunkingStrategy.HIERARCHICAL: {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
        },
        ChunkingStrategy.RECURSIVE: {
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
        ChunkingStrategy.FIXED: {
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
        ChunkingStrategy.TOKEN_AWARE: {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "tokenizer_name": "cl100k_base",
        },
    }

    config_dict = defaults.get(strategy, {})
    config_dict.update(overrides)
    config_dict["strategy"] = strategy

    return ChunkingConfig(**config_dict)
