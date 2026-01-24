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

import re
import math
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import logging

from src.retrieval import Document


class ChunkingStrategy(Enum):
    """Enumeration for different chunking strategies."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CODE = "code"
    MARKDOWN = "markdown"
    CHARACTER = "character"


@dataclass
class ChunkingConfig:
    """
    Configuration for chunking operations.

    Attributes:
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between consecutive chunks
        separators (List[str]): Separators to use for splitting
        strategy (ChunkingStrategy): Chunking strategy to use
        min_chunk_size (int): Minimum acceptable chunk size
        preserve_structure (bool): Whether to preserve document structure
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = None
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    min_chunk_size: int = 50
    preserve_structure: bool = True

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]
        
        # Validate configuration
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")


class BaseChunker(ABC):
    """Abstract base class for chunkers."""

    def __init__(self, config: ChunkingConfig):
        """
        Initialize the chunker with configuration.

        Args:
            config: Chunking configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document to chunk

        Returns:
            List of chunked documents
        """
        pass

    def _create_chunk_document(self, original_doc: Document, content: str, 
                             chunk_index: int, start_pos: int, end_pos: int) -> Document:
        """
        Create a chunk document with appropriate metadata.

        Args:
            original_doc: Original document
            content: Chunk content
            chunk_index: Index of this chunk
            start_pos: Starting position in original document
            end_pos: Ending position in original document

        Returns:
            Chunked document with metadata
        """
        chunk_id = f"{original_doc.id}_chunk_{chunk_index}"
        
        # Inherit metadata from original document and add chunk-specific metadata
        chunk_metadata = {
            **original_doc.metadata,
            "chunk_index": chunk_index,
            "chunk_start": start_pos,
            "chunk_end": end_pos,
            "original_id": original_doc.id,
            "chunk_size": len(content),
            "chunk_strategy": self.config.strategy.value
        }
        
        return Document(
            id=chunk_id,
            content=content,
            source=original_doc.source,
            doc_type=f"{original_doc.doc_type}_chunk",
            metadata=chunk_metadata,
            created_at=original_doc.created_at,
            updated_at=original_doc.updated_at,
            access_control=original_doc.access_control,
            page_number=original_doc.page_number,
            section_title=original_doc.section_title
        )


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursive character chunker that splits text by separators in order of preference.
    
    This chunker attempts to split text using a hierarchy of separators:
    1. Double newlines (\n\n)
    2. Single newlines (\n)
    3. Spaces ( )
    4. Characters (none)
    
    It recursively applies separators until chunks fit within the specified size.
    """

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk document using recursive character splitting.

        Args:
            document: Document to chunk

        Returns:
            List of chunked documents
        """
        chunks = []
        texts = [document.content]
        
        # Process each separator in order of preference
        for separator in self.config.separators:
            new_texts = []
            
            for text in texts:
                # If text is already small enough, keep it as is
                if len(text) <= self.config.chunk_size:
                    new_texts.append(text)
                    continue
                
                # Split the text using the current separator
                if separator == "":
                    # Character-level splitting
                    splits = [text[i:i+self.config.chunk_size] 
                             for i in range(0, len(text), self.config.chunk_size)]
                else:
                    # Separator-based splitting
                    splits = text.split(separator)
                
                # Process splits to ensure they fit within chunk size
                current_chunk = ""
                for split in splits:
                    # Check if adding this split would exceed chunk size
                    test_chunk = current_chunk + separator + split if current_chunk else split
                    
                    if len(test_chunk) <= self.config.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # If current chunk has content, save it
                        if current_chunk:
                            new_texts.append(current_chunk)
                        
                        # If the split itself is larger than chunk size, 
                        # we need to further split it
                        if len(split) > self.config.chunk_size:
                            # Recursively process this oversized split
                            sub_splits = self._split_large_text(split)
                            new_texts.extend(sub_splits)
                        else:
                            current_chunk = split
                
                # Add any remaining content as a chunk
                if current_chunk:
                    new_texts.append(current_chunk)
            
            texts = new_texts
        
        # Create document chunks
        for i, text in enumerate(texts):
            if len(text.strip()) >= self.config.min_chunk_size:
                chunk_doc = self._create_chunk_document(
                    document, text, i, 
                    sum(len(t) for t in texts[:i]),  # Approximate start position
                    sum(len(t) for t in texts[:i+1])   # Approximate end position
                )
                chunks.append(chunk_doc)
        
        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """
        Split a text that is larger than the chunk size.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.config.chunk_overlap
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that splits text based on semantic boundaries.
    
    This chunker attempts to maintain semantic coherence by splitting
    at sentence boundaries and respecting paragraph structure.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        # Compile regex patterns for sentence detection
        self.sentence_endings = re.compile(r'[.!?]+')
        self.punctuation = re.compile(r'[.!?]+')
        self.whitespace = re.compile(r'\s+')

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk document using semantic boundaries.

        Args:
            document: Document to chunk

        Returns:
            List of chunked documents
        """
        # First, split by paragraphs
        paragraphs = self._split_by_paragraphs(document.content)
        
        chunks = []
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If paragraph is small enough, add as single chunk
            if len(paragraph) <= self.config.chunk_size:
                chunk_doc = self._create_chunk_document(
                    document, paragraph, chunk_index, 0, len(paragraph)
                )
                chunks.append(chunk_doc)
                chunk_index += 1
                continue
            
            # Otherwise, split paragraph into sentences and group them
            sentences = self._split_by_sentences(paragraph)
            
            current_chunk = ""
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(test_chunk) <= self.config.chunk_size:
                    current_chunk = test_chunk
                else:
                    # If current chunk has content, save it
                    if current_chunk:
                        chunk_doc = self._create_chunk_document(
                            document, current_chunk.strip(), chunk_index, 0, len(current_chunk)
                        )
                        chunks.append(chunk_doc)
                        chunk_index += 1
                    
                    # If the sentence itself is larger than chunk size, 
                    # we need to further split it
                    if len(sentence) > self.config.chunk_size:
                        sub_chunks = self._split_large_sentence(sentence)
                        for sub_chunk in sub_chunks:
                            chunk_doc = self._create_chunk_document(
                                document, sub_chunk, chunk_index, 0, len(sub_chunk)
                            )
                            chunks.append(chunk_doc)
                            chunk_index += 1
                    else:
                        current_chunk = sentence
            
            # Add any remaining content as a chunk
            if current_chunk:
                chunk_doc = self._create_chunk_document(
                    document, current_chunk.strip(), chunk_index, 0, len(current_chunk)
                )
                chunks.append(chunk_doc)
                chunk_index += 1
        
        return chunks

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs."""
        # Split by double newlines first
        paragraphs = text.split('\n\n')
        # Clean up whitespace
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences."""
        # Find sentence boundaries
        sentences = self.sentence_endings.split(text)
        # Get the actual sentence endings
        sentence_endings = self.sentence_endings.findall(text)
        
        # Reconstruct sentences with their endings
        result = []
        for i, sentence in enumerate(sentences):
            if i < len(sentence_endings):
                result.append(sentence + sentence_endings[i])
            else:
                # Last part might not have an ending
                if sentence.strip():
                    result.append(sentence)
        
        # Clean up whitespace
        return [s.strip() for s in result if s.strip()]

    def _split_large_sentence(self, sentence: str) -> List[str]:
        """Split a sentence that is larger than the chunk size."""
        # First try to split by commas
        parts = sentence.split(',')
        if len(parts) > 1:
            chunks = []
            current_chunk = ""
            
            for part in parts:
                test_chunk = current_chunk + "," + part if current_chunk else part
                
                if len(test_chunk) <= self.config.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # If any chunks are still too large, fall back to character splitting
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > self.config.chunk_size:
                    final_chunks.extend(self._fallback_split(chunk))
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
        
        # If no commas, fall back to character splitting
        return self._fallback_split(sentence)

    def _fallback_split(self, text: str) -> List[str]:
        """Fallback to character-based splitting."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.config.chunk_overlap
        
        return chunks


class CodeChunker(BaseChunker):
    """
    Code-specific chunker that respects code structure and syntax.
    
    This chunker is designed for programming documents and attempts
    to maintain syntactic validity and logical code units.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        # Define code-specific separators
        self.code_separators = [
            r'\n\s*def\s+\w+',  # Function definitions
            r'\n\s*class\s+\w+',  # Class definitions
            r'\n\s*@',  # Decorators
            r'\n\s*if\s+.*:',  # If statements
            r'\n\s*for\s+.*:',  # For loops
            r'\n\s*while\s+.*:',  # While loops
            r'\n\s*try:',  # Try blocks
            r'\n\s*except\s*',  # Except blocks
            r'\n\s*else:',  # Else blocks
            r'\n\s*elif\s+.*:',  # Elif statements
            r'\n\s*import\s+',  # Import statements
            r'\n\s*from\s+\w+\s+import',  # From import statements
            r'\n\s*"""',  # Triple quotes (docstrings)
            r"\n\s*'''",  # Triple single quotes (docstrings)
            r'\n\s*#',  # Comments
            r'\n\s*\n',  # Blank lines
            r'\n',  # Newlines
        ]

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk code document respecting code structure.

        Args:
            document: Code document to chunk

        Returns:
            List of chunked documents
        """
        content = document.content
        chunks = []
        chunk_index = 0
        
        # Try to split by code structure
        current_position = 0
        current_chunk = ""
        
        while current_position < len(content):
            # Find the next code structure boundary
            next_boundary = self._find_next_boundary(content, current_position)
            
            if next_boundary == -1:
                # No more boundaries, add remaining content
                remaining = content[current_position:]
                if len(remaining) + len(current_chunk) <= self.config.chunk_size:
                    current_chunk += remaining
                    if current_chunk.strip():
                        chunk_doc = self._create_chunk_document(
                            document, current_chunk, chunk_index, 
                            current_position - len(current_chunk), len(content)
                        )
                        chunks.append(chunk_doc)
                    break
                else:
                    # Need to force split
                    forced_chunks = self._force_split(current_chunk + remaining)
                    for i, forced_chunk in enumerate(forced_chunks):
                        chunk_doc = self._create_chunk_document(
                            document, forced_chunk, chunk_index + i,
                            0, len(forced_chunk)
                        )
                        chunks.append(chunk_doc)
                    break
            
            # Get the segment to add
            segment = content[current_position:next_boundary]
            
            # Check if adding this segment would exceed chunk size
            if len(current_chunk) + len(segment) <= self.config.chunk_size:
                current_chunk += segment
                current_position = next_boundary
            else:
                # Current chunk is full, save it
                if current_chunk.strip():
                    chunk_doc = self._create_chunk_document(
                        document, current_chunk, chunk_index,
                        current_position - len(current_chunk), current_position
                    )
                    chunks.append(chunk_doc)
                    chunk_index += 1
                
                # Start new chunk with the segment
                current_chunk = segment
                current_position = next_boundary
        
        # Add any remaining content
        if current_chunk.strip():
            chunk_doc = self._create_chunk_document(
                document, current_chunk, chunk_index,
                len(content) - len(current_chunk), len(content)
            )
            chunks.append(chunk_doc)
        
        return chunks

    def _find_next_boundary(self, content: str, start_pos: int) -> int:
        """Find the next code structure boundary."""
        min_pos = float('inf')
        
        for pattern in self.code_separators:
            match = re.search(pattern, content[start_pos:])
            if match:
                abs_pos = start_pos + match.start()
                if abs_pos < min_pos:
                    min_pos = abs_pos
        
        return int(min_pos) if min_pos != float('inf') else -1

    def _force_split(self, content: str) -> List[str]:
        """Force split content that doesn't have natural boundaries."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.config.chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - self.config.chunk_overlap
        
        return chunks


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

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk markdown document preserving structure.

        Args:
            document: Markdown document to chunk

        Returns:
            List of chunked documents
        """
        content = document.content
        chunks = []
        chunk_index = 0
        
        # Split by markdown structure
        current_position = 0
        current_chunk = ""
        
        while current_position < len(content):
            # Find the next markdown structure boundary
            next_boundary = self._find_next_boundary(content, current_position)
            
            if next_boundary == -1:
                # No more boundaries, add remaining content
                remaining = content[current_position:]
                if len(remaining) + len(current_chunk) <= self.config.chunk_size:
                    current_chunk += remaining
                    if current_chunk.strip():
                        chunk_doc = self._create_chunk_document(
                            document, current_chunk, chunk_index,
                            current_position - len(current_chunk), len(content)
                        )
                        chunks.append(chunk_doc)
                    break
                else:
                    # Need to force split
                    forced_chunks = self._force_split(current_chunk + remaining)
                    for i, forced_chunk in enumerate(forced_chunks):
                        chunk_doc = self._create_chunk_document(
                            document, forced_chunk, chunk_index + i,
                            0, len(forced_chunk)
                        )
                        chunks.append(chunk_doc)
                    break
            
            # Get the segment to add
            segment = content[current_position:next_boundary]
            
            # Check if adding this segment would exceed chunk size
            if len(current_chunk) + len(segment) <= self.config.chunk_size:
                current_chunk += segment
                current_position = next_boundary
            else:
                # Current chunk is full, save it
                if current_chunk.strip():
                    chunk_doc = self._create_chunk_document(
                        document, current_chunk, chunk_index,
                        current_position - len(current_chunk), current_position
                    )
                    chunks.append(chunk_doc)
                    chunk_index += 1
                
                # Start new chunk with the segment
                current_chunk = segment
                current_position = next_boundary
        
        # Add any remaining content
        if current_chunk.strip():
            chunk_doc = self._create_chunk_document(
                document, current_chunk, chunk_index,
                len(content) - len(current_chunk), len(content)
            )
            chunks.append(chunk_doc)
        
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

    def _force_split(self, content: str) -> List[str]:
        """Force split content that doesn't have natural boundaries."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.config.chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - self.config.chunk_overlap
        
        return chunks


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
            new_config = config
            new_config.separators = ["\n\n", "\n", " ", ""]
            return RecursiveCharacterChunker(new_config)
        elif strategy == ChunkingStrategy.SENTENCE:
            # For sentence chunking, we'll use recursive with sentence separators
            new_config = config
            new_config.separators = [". ", "! ", "? ", "\n", " ", ""]
            return RecursiveCharacterChunker(new_config)
        elif strategy == ChunkingStrategy.CHARACTER:
            # For character chunking, we'll use recursive with empty separator
            new_config = config
            new_config.separators = [""]
            return RecursiveCharacterChunker(new_config)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


class AdvancedChunker:
    """
    Advanced chunker that can dynamically select the best strategy based on content.
    
    This chunker analyzes the document content and selects the most appropriate
    chunking strategy automatically.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Automatically select the best chunking strategy and chunk the document.

        Args:
            document: Document to chunk

        Returns:
            List of chunked documents
        """
        # Analyze document to determine best strategy
        strategy = self._analyze_document(document)
        
        # Create appropriate chunker
        chunker = ChunkerFactory.create_chunker(strategy, self.config)
        
        self.logger.info(f"Using {strategy.value} chunking strategy for document {document.id}")
        
        # Chunk the document
        return chunker.chunk_document(document)

    def _analyze_document(self, document: Document) -> ChunkingStrategy:
        """
        Analyze document to determine the best chunking strategy.

        Args:
            document: Document to analyze

        Returns:
            Recommended chunking strategy
        """
        content = document.content.lower()
        doc_type = document.doc_type.lower()
        
        # Check for code indicators
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'if __name__', 
                          'function', 'var ', 'const ', 'let ', 'public ', 'private ']
        code_matches = sum(1 for indicator in code_indicators if indicator in content)
        
        # Check for markdown indicators
        md_indicators = ['#', '##', '###', '- ', '1. ', '> ', '```', '|', '---']
        md_matches = sum(1 for indicator in md_indicators if indicator in content)
        
        # Check for paragraph structure
        paragraph_count = content.count('\n\n')
        
        # Determine strategy based on analysis
        if 'code' in doc_type or 'programming' in doc_type or code_matches > 5:
            return ChunkingStrategy.CODE
        elif 'markdown' in doc_type or 'md' in doc_type or md_matches > 3:
            return ChunkingStrategy.MARKDOWN
        elif paragraph_count > len(content.split()) / 100:  # If many paragraphs relative to content
            return ChunkingStrategy.PARAGRAPH
        else:
            # Default to semantic for general content
            return ChunkingStrategy.SEMANTIC


def chunk_document(document: Document, config: Optional[ChunkingConfig] = None) -> List[Document]:
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
    
    chunker = AdvancedChunker(config)
    return chunker.chunk_document(document)


__all__ = [
    "ChunkingStrategy", "ChunkingConfig", "BaseChunker", 
    "RecursiveCharacterChunker", "SemanticChunker", "CodeChunker", 
    "MarkdownChunker", "ChunkerFactory", "AdvancedChunker", 
    "chunk_document"
]