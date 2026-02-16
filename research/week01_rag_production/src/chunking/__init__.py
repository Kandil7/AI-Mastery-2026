"""
Advanced Text Chunking Strategies for Production RAG System

This module implements various text chunking strategies for the RAG system,
allowing for optimal document segmentation based on content type and retrieval needs.
Different chunking strategies are available to handle various document types and
preserving semantic coherence while enabling effective retrieval.

The chunking strategies follow production best practices:
- Multiple chunking algorithms for different content types
- Semantic boundary preservation
- Configurable chunk sizes and overlaps
- Metadata preservation during chunking
- Content-aware splitting
- Performance optimization for large documents

Key Features:
- Recursive chunking with configurable separators
- Semantic chunking based on sentence boundaries
- Code-specific chunking for programming documents
- Markdown-aware chunking preserving structure
- Paragraph-aware chunking
- Overlap management to preserve context
- Metadata inheritance in chunks

Security Considerations:
- Content sanitization during chunking
- Boundary validation to prevent injection
- Proper encoding handling
- Memory-safe processing of large documents
"""

from .config import ChunkingStrategy, ChunkingConfig
from .base import BaseChunker
from .strategies.recursive import RecursiveCharacterChunker
from .strategies.semantic import SemanticChunker
from .strategies.code import CodeChunker
from .strategies.markdown import MarkdownChunker
from .factory import ChunkerFactory
from .advanced import AdvancedChunker
from .sanitize import sanitize_text
from .tokenizer import CharCounter, TiktokenCounter, build_counter
from .spans import TextSpan, ChunkSpan


def chunk_document(document, config=None):
    """
    Convenience function to chunk a document with default or provided configuration.

    Args:
        document: Document to chunk
        config: Optional chunking configuration (uses defaults if not provided)

    Returns:
        List of chunked documents
    """
    if config is None:
        config = ChunkingConfig()

    # If config strategy is AUTO, use AdvancedChunker, otherwise use appropriate chunker
    if config.strategy == ChunkingStrategy.AUTO:
        chunker = AdvancedChunker(config)
    else:
        chunker = ChunkerFactory.create_chunker(config.strategy, config)

    return chunker.chunk_document(document)


__all__ = [
    "ChunkingStrategy", "ChunkingConfig", "BaseChunker",
    "RecursiveCharacterChunker", "SemanticChunker", "CodeChunker",
    "MarkdownChunker", "ChunkerFactory", "AdvancedChunker",
    "chunk_document", "sanitize_text", "CharCounter", "TiktokenCounter",
    "build_counter", "TextSpan", "ChunkSpan"
]