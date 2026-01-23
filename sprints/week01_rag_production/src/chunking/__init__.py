"""
Advanced Chunking Strategies for Production RAG Systems

This module implements various chunking strategies for breaking down documents
into smaller, more manageable pieces suitable for embedding and retrieval.
Different strategies are appropriate for different types of content and use cases.

The module includes:
- Fixed-size chunking for uniform processing
- Semantic chunking that respects document structure
- Agentic chunking that uses LLMs for intelligent splitting
- Validation and quality assurance for chunks

Key Features:
- Multiple chunking strategies for different use cases
- Overlap support to maintain context across chunks
- Content-aware splitting that respects document structure
- Quality validation for generated chunks
- Performance optimization for large documents

Strategies Implemented:
- FixedSizeChunker: Splits documents at regular intervals
- SemanticChunker: Respects paragraph/sentence boundaries
- AgenticChunker: Uses LLMs to determine optimal split points
- HierarchicalChunker: Maintains document hierarchy during chunking
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from src.retrieval import Document


@dataclass
class ChunkingConfig:
    """
    Configuration for chunking strategies.
    
    Attributes:
        chunk_size (int): Target size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        min_chunk_size (int): Minimum acceptable chunk size
        max_chunk_size (int): Maximum acceptable chunk size
        sentence_aware (bool): Whether to respect sentence boundaries
        paragraph_aware (bool): Whether to respect paragraph boundaries
    """
    chunk_size: int = 512
    overlap: int = 100
    min_chunk_size: int = 50
    max_chunk_size: int = 1024
    sentence_aware: bool = True
    paragraph_aware: bool = True


class ChunkingStrategy(ABC):
    """
    Abstract base class for different chunking strategies.
    
    Defines the interface for various chunking approaches that can be used
    to split documents into smaller, more manageable pieces while preserving
    semantic coherence and context.
    """
    
    @abstractmethod
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """
        Split a document into chunks according to the strategy.
        
        Args:
            document (Document): The document to chunk
            config (ChunkingConfig): Configuration for chunking
            
        Returns:
            List[Document]: List of chunked documents
        """
        pass
    
    def validate_chunks(self, chunks: List[Document], config: ChunkingConfig) -> List[Document]:
        """
        Validate and filter chunks based on size and quality criteria.
        
        Args:
            chunks (List[Document]): List of chunks to validate
            config (ChunkingConfig): Configuration for validation criteria
            
        Returns:
            List[Document]: Validated and filtered chunks
        """
        validated_chunks = []
        for chunk in chunks:
            if len(chunk.content) >= config.min_chunk_size and len(chunk.content) <= config.max_chunk_size:
                validated_chunks.append(chunk)
        
        return validated_chunks


class FixedSizeChunker(ChunkingStrategy):
    """
    Fixed-size chunking strategy that splits documents at regular intervals.
    
    This approach divides documents into fixed-length segments regardless
    of semantic boundaries. It's simple and predictable but may split
    in the middle of meaningful text units.
    
    Pros:
    - Consistent chunk sizes
    - Predictable processing times
    - Simple implementation
    
    Cons:
    - May break semantic coherence
    - Context can be lost at boundaries
    """
    
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """
        Split a document into fixed-size chunks.
        
        Args:
            document (Document): The document to chunk
            config (ChunkingConfig): Configuration for chunking
            
        Returns:
            List[Document]: List of chunked documents
        """
        content = document.content
        chunks = []
        
        start = 0
        while start < len(content):
            end = start + config.chunk_size
            chunk_content = content[start:end]
            
            # Create a new document for this chunk
            chunk_doc = Document(
                id=f"{document.id}_chunk_{len(chunks) + 1}",
                content=chunk_content,
                source=document.source,
                doc_type=f"{document.doc_type}_chunk",
                created_at=document.created_at,
                updated_at=document.updated_at,
                access_control=document.access_control,
                page_number=document.page_number,
                section_title=document.section_title,
                metadata={
                    **document.metadata,
                    "chunk_index": len(chunks),
                    "original_id": document.id,
                    "chunk_method": "fixed_size",
                    "start_pos": start,
                    "end_pos": end
                }
            )
            
            chunks.append(chunk_doc)
            
            # Move to the next chunk position with overlap
            start = end - config.overlap if config.overlap < config.chunk_size else start + config.chunk_size
        
        return self.validate_chunks(chunks, config)


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking strategy that respects document structure.
    
    This approach attempts to split documents at natural boundaries
    like paragraph breaks, headings, or sentences to maintain context
    and meaning within each chunk.
    
    Pros:
    - Preserves semantic coherence
    - Better context retention
    - More natural reading experience
    
    Cons:
    - Variable chunk sizes
    - More complex implementation
    - May require more processing
    """
    
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """
        Split a document into semantically coherent chunks.
        
        Args:
            document (Document): The document to chunk
            config (ChunkingConfig): Configuration for chunking
            
        Returns:
            List[Document]: List of chunked documents
        """
        # Split by paragraphs first
        paragraphs = [p.strip() for p in document.content.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > config.chunk_size and current_chunk:
                # Save the current chunk
                chunk_doc = Document(
                    id=f"{document.id}_chunk_{chunk_idx + 1}",
                    content=current_chunk.strip(),
                    source=document.source,
                    doc_type=f"{document.doc_type}_chunk",
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                    access_control=document.access_control,
                    page_number=document.page_number,
                    section_title=document.section_title,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_idx,
                        "original_id": document.id,
                        "chunk_method": "semantic",
                        "boundary_type": "paragraph"
                    }
                )
                chunks.append(chunk_doc)
                
                # Start a new chunk with potential overlap
                if config.overlap > 0:
                    # Take the end portion of the previous chunk for overlap
                    overlap_text = current_chunk[-config.overlap:] if len(current_chunk) > config.overlap else current_chunk
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
                chunk_idx += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk_doc = Document(
                id=f"{document.id}_chunk_{chunk_idx + 1}",
                content=current_chunk.strip(),
                source=document.source,
                doc_type=f"{document.doc_type}_chunk",
                created_at=document.created_at,
                updated_at=document.updated_at,
                access_control=document.access_control,
                page_number=document.page_number,
                section_title=document.section_title,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_idx,
                    "original_id": document.id,
                    "chunk_method": "semantic",
                    "boundary_type": "paragraph"
                }
            )
            chunks.append(chunk_doc)
        
        return self.validate_chunks(chunks, config)
    
    def chunk_sentences(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """
        Alternative method to chunk based on sentences.
        
        Args:
            document (Document): The document to chunk
            config (ChunkingConfig): Configuration for chunking
            
        Returns:
            List[Document]: List of chunked documents
        """
        # Split by sentences using regex
        sentences = re.split(r'[.!?]+\s+', document.content)
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > config.chunk_size and current_chunk:
                # Save the current chunk
                chunk_doc = Document(
                    id=f"{document.id}_chunk_{chunk_idx + 1}",
                    content=current_chunk.strip(),
                    source=document.source,
                    doc_type=f"{document.doc_type}_chunk",
                    created_at=document.created_at,
                    updated_at=document.updated_at,
                    access_control=document.access_control,
                    page_number=document.page_number,
                    section_title=document.section_title,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_idx,
                        "original_id": document.id,
                        "chunk_method": "semantic_sentence",
                        "boundary_type": "sentence"
                    }
                )
                chunks.append(chunk_doc)
                
                # Start a new chunk with potential overlap
                if config.overlap > 0:
                    # Take the end portion of the previous chunk for overlap
                    overlap_text = current_chunk[-config.overlap:] if len(current_chunk) > config.overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                chunk_idx += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk_doc = Document(
                id=f"{document.id}_chunk_{chunk_idx + 1}",
                content=current_chunk.strip(),
                source=document.source,
                doc_type=f"{document.doc_type}_chunk",
                created_at=document.created_at,
                updated_at=document.updated_at,
                access_control=document.access_control,
                page_number=document.page_number,
                section_title=document.section_title,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_idx,
                    "original_id": document.id,
                    "chunk_method": "semantic_sentence",
                    "boundary_type": "sentence"
                }
            )
            chunks.append(chunk_doc)
        
        return self.validate_chunks(chunks, config)


class AgenticChunker(ChunkingStrategy):
    """
    Agentic chunking strategy that uses LLMs for intelligent splitting.
    
    This approach leverages language models to determine optimal split points
    based on semantic coherence and topic shifts. It's more computationally
    expensive but produces higher quality chunks.
    
    Note: This is a simplified implementation. A full implementation would
    involve calling an LLM to determine optimal split points.
    
    Pros:
    - Highest quality chunks
    - Topic-aware splitting
    - Contextually appropriate boundaries
    
    Cons:
    - High computational cost
    - Requires LLM access
    - Slower processing
    """
    
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """
        Split a document using LLM-guided chunking.
        
        Args:
            document (Document): The document to chunk
            config (ChunkingConfig): Configuration for chunking
            
        Returns:
            List[Document]: List of chunked documents
        """
        # For this implementation, we'll simulate agentic chunking by using
        # a combination of semantic and size-based approaches
        # In a real implementation, this would involve calling an LLM
        
        # First, try semantic chunking
        semantic_chunker = SemanticChunker()
        chunks = semantic_chunker.chunk(document, config)
        
        # If any chunks are still too large, further subdivide them
        final_chunks = []
        for chunk in chunks:
            if len(chunk.content) <= config.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Further subdivide large chunks using fixed-size approach
                sub_config = ChunkingConfig(
                    chunk_size=config.max_chunk_size // 2,
                    overlap=config.overlap // 2,
                    min_chunk_size=config.min_chunk_size,
                    max_chunk_size=config.max_chunk_size
                )
                sub_chunks = FixedSizeChunker().chunk(chunk, sub_config)
                final_chunks.extend(sub_chunks)
        
        return self.validate_chunks(final_chunks, config)


class HierarchicalChunker(ChunkingStrategy):
    """
    Hierarchical chunking strategy that maintains document structure.
    
    This approach preserves the hierarchical nature of documents (sections,
    subsections, paragraphs) while creating appropriately sized chunks.
    Useful for structured documents like manuals, reports, or academic papers.
    
    Pros:
    - Preserves document hierarchy
    - Maintains structural context
    - Good for navigable documents
    
    Cons:
    - More complex implementation
    - May create uneven chunk sizes
    - Requires structured input
    """
    
    def chunk(self, document: Document, config: ChunkingConfig) -> List[Document]:
        """
        Split a document while preserving its hierarchical structure.
        
        Args:
            document (Document): The document to chunk
            config (ChunkingConfig): Configuration for chunking
            
        Returns:
            List[Document]: List of chunked documents
        """
        # Identify document structure (headings, sections)
        lines = document.content.split('\n')
        chunks = []
        current_section = ""
        current_heading = ""
        chunk_idx = 0
        
        for line in lines:
            # Check if line is a heading (simple heuristic)
            if self._is_heading(line):
                # If we have accumulated content, save it as a chunk
                if current_section.strip():
                    chunk_doc = self._create_hierarchical_chunk(
                        document, current_section, current_heading, chunk_idx
                    )
                    chunks.append(chunk_doc)
                    chunk_idx += 1
                
                # Start new section with this heading
                current_heading = line.strip()
                current_section = line + "\n"
            else:
                # Add line to current section
                current_section += line + "\n"
                
                # If section becomes too large, create a chunk
                if len(current_section) > config.chunk_size:
                    chunk_doc = self._create_hierarchical_chunk(
                        document, current_section, current_heading, chunk_idx
                    )
                    chunks.append(chunk_doc)
                    chunk_idx += 1
                    
                    # Reset section but keep the heading context
                    current_section = current_heading + "\n" if current_heading else ""
        
        # Add the last section if it has content
        if current_section.strip():
            chunk_doc = self._create_hierarchical_chunk(
                document, current_section, current_heading, chunk_idx
            )
            chunks.append(chunk_doc)
        
        return self.validate_chunks(chunks, config)
    
    def _is_heading(self, line: str) -> bool:
        """
        Simple heuristic to identify if a line is a heading.
        
        Args:
            line (str): Line to check
            
        Returns:
            bool: True if line appears to be a heading
        """
        # Check for common heading patterns
        line = line.strip()
        if not line:
            return False
            
        # Check for markdown-style headings
        if line.startswith('#'):
            return True
            
        # Check for all caps titles (common in documents)
        if line.isupper() and len(line) < 100:
            return True
            
        # Check for numbered sections (e.g., "1. Introduction")
        if re.match(r'^\d+(\.\d+)*\s+\w+', line):
            return True
            
        return False
    
    def _create_hierarchical_chunk(
        self, 
        original_doc: Document, 
        content: str, 
        heading: str, 
        chunk_idx: int
    ) -> Document:
        """
        Create a document chunk with hierarchical metadata.
        
        Args:
            original_doc (Document): Original document
            content (str): Content for the chunk
            heading (str): Heading associated with the chunk
            chunk_idx (int): Index of this chunk
            
        Returns:
            Document: Chunked document with hierarchical metadata
        """
        return Document(
            id=f"{original_doc.id}_hchunk_{chunk_idx + 1}",
            content=content.strip(),
            source=original_doc.source,
            doc_type=f"{original_doc.doc_type}_hchunk",
            created_at=original_doc.created_at,
            updated_at=original_doc.updated_at,
            access_control=original_doc.access_control,
            page_number=original_doc.page_number,
            section_title=heading if heading else original_doc.section_title,
            metadata={
                **original_doc.metadata,
                "chunk_index": chunk_idx,
                "original_id": original_doc.id,
                "chunk_method": "hierarchical",
                "hierarchy_level": self._get_hierarchy_level(heading),
                "section_title": heading
            }
        )
    
    def _get_hierarchy_level(self, heading: str) -> int:
        """
        Determine the hierarchy level based on heading format.
        
        Args:
            heading (str): Heading text
            
        Returns:
            int: Hierarchy level (1 for highest level)
        """
        if heading.startswith('###'):
            return 3
        elif heading.startswith('##'):
            return 2
        elif heading.startswith('#'):
            return 1
        else:
            # Try to detect from numbered sections
            match = re.match(r'^(\d+)\.?', heading)
            if match:
                level = len(match.group(1).split('.'))
                return level
            return 0


def get_chunker(strategy_name: str) -> ChunkingStrategy:
    """
    Factory function to get a chunking strategy by name.
    
    Args:
        strategy_name (str): Name of the strategy to use
            
    Returns:
        ChunkingStrategy: Instance of the requested strategy
    """
    strategies = {
        'fixed': FixedSizeChunker(),
        'semantic': SemanticChunker(),
        'agentic': AgenticChunker(),
        'hierarchical': HierarchicalChunker()
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}. "
                         f"Available: {list(strategies.keys())}")
    
    return strategies[strategy_name]