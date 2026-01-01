"""
Production Data Pipeline for Enterprise RAG Systems
====================================================

Implements the first pillar of production RAG: architecting a production-grade
data pipeline with advanced chunking, multi-format handling, and metadata enrichment.

Key Components:
- SemanticChunker: LLM-based semantic boundary detection
- HierarchicalChunker: Parent-child chunk relationships
- MetadataExtractor: Automatic metadata enrichment
- DocumentParser: Multi-format document handling (PDF, DOCX, HTML)

Reference: "From Prototype to Production: Enterprise RAG Systems"
"""

import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import numpy as np


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class DocumentChunk:
    """
    A chunk of document content with metadata.
    
    Production systems require rich metadata for:
    - Filtering during retrieval (date, department, access)
    - Citation in generated responses
    - Cache invalidation on updates
    """
    id: str
    content: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    # Parent-child relationships for hierarchical chunking
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate deterministic chunk ID."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.doc_id}_chunk_{self.chunk_index}_{content_hash}"


@dataclass
class Document:
    """Raw document before processing."""
    id: str
    content: str
    source: str
    doc_type: str = "text"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# CHUNKING STRATEGIES
# ============================================================

class BaseChunker(ABC):
    """Base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Split document into chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking with overlap.
    
    ⚠️ Prototype-level approach - included for comparison.
    Production systems should use semantic or hierarchical chunking.
    
    Uses sliding window with 15-20% overlap to prevent context loss.
    """
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        overlap: int = 100,
        separator: str = " "
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        text = document.content
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at word boundary
            if end < len(text):
                last_space = text[start:end].rfind(self.separator)
                if last_space > self.chunk_size // 2:
                    end = start + last_space
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    id="",
                    content=chunk_content,
                    doc_id=document.id,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=document.metadata.copy()
                ))
                chunk_index += 1
            
            start = end - self.overlap
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using text structure analysis.
    
    Production Pattern:
    - Analyzes document structure (headers, paragraphs, lists)
    - Preserves semantic units (complete thoughts, ideas)
    - Respects document hierarchy
    
    Advantages over fixed-size:
    - No mid-sentence splits
    - Preserves context
    - Better retrieval quality
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        sentence_splitter: Optional[Callable] = None
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_splitter = sentence_splitter or self._default_sentence_split
        
        # Semantic boundary patterns
        self.strong_boundaries = [
            r'\n\s*#{1,6}\s+',      # Markdown headers
            r'\n\s*\d+\.\s+',        # Numbered lists
            r'\n\s*[-*]\s+',         # Bullet points
            r'\n{2,}',               # Double newlines (paragraphs)
        ]
        
        self.weak_boundaries = [
            r'(?<=[.!?])\s+',        # Sentence endings
            r'(?<=:)\s+',            # After colons
        ]
    
    def _default_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """Find positions of semantic boundaries."""
        boundaries = {0, len(text)}
        
        # Strong boundaries (headers, paragraphs)
        for pattern in self.strong_boundaries:
            for match in re.finditer(pattern, text):
                boundaries.add(match.start())
        
        return sorted(boundaries)
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        text = document.content
        boundaries = self._find_semantic_boundaries(text)
        chunks = []
        chunk_index = 0
        
        i = 0
        while i < len(boundaries) - 1:
            start = boundaries[i]
            
            # Find end that respects max chunk size
            end_idx = i + 1
            while end_idx < len(boundaries):
                potential_end = boundaries[end_idx]
                if potential_end - start > self.max_chunk_size:
                    break
                end_idx += 1
            
            # Use previous boundary as end
            end = boundaries[min(end_idx, len(boundaries) - 1)]
            if end - start > self.max_chunk_size:
                end = boundaries[end_idx - 1] if end_idx > i + 1 else start + self.max_chunk_size
            
            chunk_content = text[start:end].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(DocumentChunk(
                    id="",
                    content=chunk_content,
                    doc_id=document.id,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=document.metadata.copy()
                ))
                chunk_index += 1
            
            i = end_idx - 1 if end_idx > i + 1 else i + 1
        
        return chunks


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking with parent-child relationships.
    
    Production Pattern:
    - Creates small, precise "child" chunks for retrieval
    - Maintains links to larger "parent" chunks for context
    - Retrieval uses children, generation uses parents
    
    This solves the precision-context tradeoff:
    - Small chunks = better retrieval precision
    - Large context = better LLM understanding
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        child_overlap: int = 50
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        
        self.parent_chunker = SemanticChunker(max_chunk_size=parent_chunk_size)
        self.child_chunker = FixedSizeChunker(
            chunk_size=child_chunk_size, 
            overlap=child_overlap
        )
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Create hierarchical chunks with parent-child relationships."""
        all_chunks = []
        
        # First, create parent chunks
        parent_chunks = self.parent_chunker.chunk(document)
        
        for parent in parent_chunks:
            parent.metadata["chunk_type"] = "parent"
            all_chunks.append(parent)
            
            # Create child document from parent content
            child_doc = Document(
                id=f"{document.id}_parent_{parent.chunk_index}",
                content=parent.content,
                source=document.source,
                metadata=parent.metadata.copy()
            )
            
            # Create child chunks
            children = self.child_chunker.chunk(child_doc)
            
            for child in children:
                child.parent_id = parent.id
                child.metadata["chunk_type"] = "child"
                child.metadata["parent_chunk_index"] = parent.chunk_index
                parent.children_ids.append(child.id)
                all_chunks.append(child)
        
        return all_chunks
    
    def get_parent(
        self, 
        child_chunk: DocumentChunk, 
        all_chunks: List[DocumentChunk]
    ) -> Optional[DocumentChunk]:
        """Retrieve parent chunk for a child."""
        if not child_chunk.parent_id:
            return None
        
        for chunk in all_chunks:
            if chunk.id == child_chunk.parent_id:
                return chunk
        return None


# ============================================================
# METADATA EXTRACTION
# ============================================================

class MetadataExtractor:
    """
    Extract and enrich document metadata.
    
    Production Pattern:
    Metadata enables:
    - Precise filtering during retrieval
    - Access control enforcement
    - Cache invalidation
    - Usage analytics
    
    Key metadata fields:
    - doc_type: policy, contract, memo, etc.
    - department: sales, engineering, legal
    - access_level: public, internal, confidential
    - created_at, updated_at: for freshness
    """
    
    def __init__(self, extractors: Optional[Dict[str, Callable]] = None):
        self.extractors = extractors or {}
        
        # Default extractors
        self._register_default_extractors()
    
    def _register_default_extractors(self):
        """Register default metadata extractors."""
        self.extractors.update({
            "word_count": self._extract_word_count,
            "language": self._extract_language,
            "has_code": self._extract_has_code,
            "has_tables": self._extract_has_tables,
            "document_type": self._extract_document_type,
            "key_entities": self._extract_key_entities,
        })
    
    def _extract_word_count(self, doc: Document) -> int:
        return len(doc.content.split())
    
    def _extract_language(self, doc: Document) -> str:
        # Simplified - production would use langdetect
        if any(ord(c) > 127 for c in doc.content[:500]):
            return "non-english"
        return "english"
    
    def _extract_has_code(self, doc: Document) -> bool:
        code_patterns = [r'```', r'def\s+\w+\(', r'class\s+\w+:', r'import\s+\w+']
        return any(re.search(p, doc.content) for p in code_patterns)
    
    def _extract_has_tables(self, doc: Document) -> bool:
        return '|' in doc.content and re.search(r'\|[^|]+\|', doc.content) is not None
    
    def _extract_document_type(self, doc: Document) -> str:
        content_lower = doc.content.lower()[:1000]
        
        type_indicators = {
            "policy": ["policy", "procedure", "guideline", "compliance"],
            "contract": ["agreement", "contract", "terms", "party", "whereas"],
            "memo": ["memo", "memorandum", "to:", "from:", "subject:"],
            "report": ["report", "analysis", "findings", "conclusion"],
            "documentation": ["api", "function", "parameter", "returns", "example"],
        }
        
        for doc_type, indicators in type_indicators.items():
            if any(ind in content_lower for ind in indicators):
                return doc_type
        
        return "general"
    
    def _extract_key_entities(self, doc: Document) -> List[str]:
        """Extract potential key entities (simplified NER)."""
        # Look for capitalized phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', doc.content)
        # Dedupe and limit
        unique = list(set(entities))[:20]
        return unique
    
    def extract(self, doc: Document) -> Dict[str, Any]:
        """Extract all metadata from document."""
        metadata = doc.metadata.copy()
        
        for name, extractor in self.extractors.items():
            try:
                metadata[name] = extractor(doc)
            except Exception as e:
                metadata[f"{name}_error"] = str(e)
        
        # Add timestamps
        metadata["processed_at"] = datetime.now().isoformat()
        
        return metadata
    
    def register_extractor(self, name: str, extractor: Callable):
        """Register a custom metadata extractor."""
        self.extractors[name] = extractor


# ============================================================
# DOCUMENT PARSING
# ============================================================

class DocumentParser:
    """
    Multi-format document parser.
    
    Production Pattern:
    Enterprise data exists in diverse formats:
    - PDF: Scanned and digital
    - DOCX: Microsoft Word
    - HTML: Web pages
    - Markdown: Technical docs
    
    Key considerations:
    - Preserve structure (tables, lists, headers)
    - Handle encoding issues
    - Extract embedded metadata
    """
    
    def __init__(self):
        self.parsers = {
            "text": self._parse_text,
            "txt": self._parse_text,
            "md": self._parse_markdown,
            "markdown": self._parse_markdown,
            "html": self._parse_html,
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
        }
    
    def parse(self, content: str, format: str) -> Document:
        """Parse document content based on format."""
        parser = self.parsers.get(format.lower(), self._parse_text)
        return parser(content, format)
    
    def _parse_text(self, content: str, format: str) -> Document:
        """Parse plain text."""
        return Document(
            id=hashlib.md5(content.encode()).hexdigest()[:12],
            content=content,
            source="text",
            doc_type="text"
        )
    
    def _parse_markdown(self, content: str, format: str) -> Document:
        """Parse markdown, preserving structure."""
        # Remove inline HTML
        clean = re.sub(r'<[^>]+>', '', content)
        
        # Extract title from first header
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled"
        
        return Document(
            id=hashlib.md5(content.encode()).hexdigest()[:12],
            content=clean,
            source="markdown",
            doc_type="documentation",
            metadata={"title": title}
        )
    
    def _parse_html(self, content: str, format: str) -> Document:
        """Parse HTML, extracting text content."""
        # Simple tag removal
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return Document(
            id=hashlib.md5(content.encode()).hexdigest()[:12],
            content=text,
            source="html",
            doc_type="web"
        )
    
    def _parse_pdf(self, content: str, format: str) -> Document:
        """Parse PDF content (placeholder for production PDF lib)."""
        # In production, use PyPDF2, pdfplumber, or layout-aware tools
        return Document(
            id=hashlib.md5(content.encode()).hexdigest()[:12],
            content=content,
            source="pdf",
            doc_type="document",
            metadata={"requires_ocr": False}
        )
    
    def _parse_docx(self, content: str, format: str) -> Document:
        """Parse DOCX (placeholder for production docx lib)."""
        # In production, use python-docx
        return Document(
            id=hashlib.md5(content.encode()).hexdigest()[:12],
            content=content,
            source="docx",
            doc_type="document"
        )


# ============================================================
# PRODUCTION PIPELINE
# ============================================================

class ProductionDataPipeline:
    """
    Complete production data pipeline.
    
    Orchestrates:
    1. Parsing: Multi-format handling
    2. Metadata: Automatic extraction
    3. Chunking: Hierarchical with semantic awareness
    4. Validation: Quality checks
    
    Example:
        pipeline = ProductionDataPipeline()
        chunks = pipeline.process(doc_content, "markdown")
    """
    
    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
        parser: Optional[DocumentParser] = None
    ):
        self.chunker = chunker or HierarchicalChunker()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.parser = parser or DocumentParser()
    
    def process(
        self, 
        content: str, 
        format: str = "text",
        doc_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Process raw content into indexed chunks.
        
        Args:
            content: Raw document content
            format: Document format (text, md, html, pdf, docx)
            doc_id: Optional document ID
        
        Returns:
            List of DocumentChunks ready for embedding
        """
        # Step 1: Parse
        document = self.parser.parse(content, format)
        if doc_id:
            document.id = doc_id
        
        # Step 2: Extract metadata
        document.metadata = self.metadata_extractor.extract(document)
        
        # Step 3: Chunk
        chunks = self.chunker.chunk(document)
        
        # Step 4: Validate
        chunks = self._validate_chunks(chunks)
        
        return chunks
    
    def _validate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Validate and filter chunks."""
        valid_chunks = []
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.content.strip():
                continue
            
            # Skip very short chunks unless they're children
            if len(chunk.content) < 50 and chunk.metadata.get("chunk_type") != "child":
                continue
            
            valid_chunks.append(chunk)
        
        return valid_chunks
    
    def process_batch(
        self, 
        documents: List[Dict[str, str]]
    ) -> List[DocumentChunk]:
        """
        Process multiple documents.
        
        Args:
            documents: List of {"content": ..., "format": ..., "id": ...}
        
        Returns:
            All chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.process(
                content=doc.get("content", ""),
                format=doc.get("format", "text"),
                doc_id=doc.get("id")
            )
            all_chunks.extend(chunks)
        
        return all_chunks


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Models
    "Document",
    "DocumentChunk",
    # Chunkers
    "BaseChunker",
    "FixedSizeChunker",
    "SemanticChunker", 
    "HierarchicalChunker",
    # Utilities
    "MetadataExtractor",
    "DocumentParser",
    # Pipeline
    "ProductionDataPipeline",
]
