# src/chunking/factory.py
from .config import ChunkingConfig, ChunkingStrategy
from .base import BaseChunker
from .analyze import choose_strategy
from .strategies.recursive import RecursiveCharacterChunker
from .strategies.semantic import SemanticChunker
from .strategies.code import CodeChunker
from .strategies.markdown import MarkdownChunker
from .advanced import AdvancedChunker


class ChunkerFactory:
    """Factory for creating appropriate chunkers based on strategy."""

    @staticmethod
    def create_chunker(strategy: ChunkingStrategy, config: ChunkingConfig) -> BaseChunker:
        """
        Create a chunker instance based on the specified strategy.

        Args:
            strategy: Chunking strategy to use
            config: Chunking configuration

        Returns:
            Appropriate chunker instance
        """
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterChunker(config)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(config)
        elif strategy == ChunkingStrategy.CODE:
            return CodeChunker(config)
        elif strategy == ChunkingStrategy.MARKDOWN:
            return MarkdownChunker(config)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            # For paragraph chunking, we'll use recursive with paragraph separator
            new_config = config.copy_with(separators=["\n\n", "\n", " ", ""])
            return RecursiveCharacterChunker(new_config)
        elif strategy == ChunkingStrategy.SENTENCE:
            # For sentence chunking, we'll use recursive with sentence separators
            new_config = config.copy_with(separators=[". ", "! ", "? ", "\n", " ", ""])
            return RecursiveCharacterChunker(new_config)
        elif strategy == ChunkingStrategy.CHARACTER:
            # For character chunking, we'll use recursive with empty separator
            new_config = config.copy_with(separators=[""])
            return RecursiveCharacterChunker(new_config)
        elif strategy == ChunkingStrategy.AUTO:
            # For AUTO, we'll analyze the document to determine the best strategy
            return AdvancedChunker(config)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")