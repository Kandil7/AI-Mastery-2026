"""
Legal Document RAG - Data Processing Module
============================================
Document ingestion, chunking, and preprocessing for legal documents.

Features:
- PDF and DOCX support
- Legal-specific chunking strategies
- Citation extraction
- Metadata enrichment

Author: AI-Mastery-2026
"""

import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DocumentMetadata:
    """Metadata for a legal document."""
    document_id: str
    filename: str
    title: Optional[str] = None
    document_type: str = "unknown"  # contract, brief, statute, etc.
    jurisdiction: Optional[str] = None
    date: Optional[datetime] = None
    parties: List[str] = field(default_factory=list)
    page_count: int = 0
    word_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "document_type": self.document_type,
            "jurisdiction": self.jurisdiction,
            "date": self.date.isoformat() if self.date else None,
            "parties": self.parties,
            "page_count": self.page_count,
            "word_count": self.word_count,
        }


@dataclass
class TextChunk:
    """A chunk of text from a legal document."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section": self.section,
            "citations": self.citations,
            "token_count": self.token_count,
        }


# ============================================================
# DOCUMENT LOADERS
# ============================================================

class PDFLoader:
    """Load and extract text from PDF documents."""
    
    @staticmethod
    def load(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Load PDF and extract text with metadata.
        
        Returns:
            Tuple of (full_text, metadata_dict)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF required. Install: pip install pymupdf")
        
        doc = fitz.open(file_path)
        
        text_parts = []
        page_count = len(doc)
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_parts.append(f"[PAGE {page_num + 1}]\n{text}")
        
        full_text = "\n\n".join(text_parts)
        
        metadata = {
            "page_count": page_count,
            "word_count": len(full_text.split()),
            "title": doc.metadata.get("title"),
        }
        
        doc.close()
        return full_text, metadata
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')


class DOCXLoader:
    """Load and extract text from DOCX documents."""
    
    @staticmethod
    def load(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Load DOCX and extract text."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx required. Install: pip install python-docx")
        
        doc = Document(file_path)
        
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        
        metadata = {
            "page_count": len(doc.sections),
            "word_count": len(full_text.split()),
        }
        
        return full_text, metadata
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        return file_path.lower().endswith('.docx')


class TextLoader:
    """Load plain text documents."""
    
    @staticmethod
    def load(file_path: str) -> Tuple[str, Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return text, {"word_count": len(text.split())}
    
    @staticmethod
    def is_supported(file_path: str) -> bool:
        return file_path.lower().endswith('.txt')


class DocumentLoader:
    """Unified document loader supporting multiple formats."""
    
    LOADERS = [PDFLoader, DOCXLoader, TextLoader]
    
    @classmethod
    def load(cls, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Load document using appropriate loader."""
        for loader in cls.LOADERS:
            if loader.is_supported(file_path):
                return loader.load(file_path)
        
        raise ValueError(f"Unsupported file format: {file_path}")
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        return ['.pdf', '.docx', '.txt']


# ============================================================
# CHUNKING STRATEGIES
# ============================================================

class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str, **kwargs) -> List[str]:
        raise NotImplementedError


class SentenceChunker(ChunkingStrategy):
    """
    Chunk by sentences with overlap.
    
    Good for: General legal documents, briefs
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, **kwargs) -> List[str]:
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_size + sentence_words > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep last few sentences for overlap
                overlap_words = 0
                overlap_start = len(current_chunk)
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_words += len(current_chunk[i].split())
                    if overlap_words >= self.overlap:
                        overlap_start = i
                        break
                
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_words
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


class SectionChunker(ChunkingStrategy):
    """
    Chunk by document sections/headers.
    
    Good for: Statutes, contracts with clear sections
    """
    
    SECTION_PATTERNS = [
        r'^(?:ARTICLE|Article|SECTION|Section|§)\s*\d+',
        r'^\d+\.\d+\s+[A-Z]',
        r'^(?:WHEREAS|NOW, THEREFORE|IN WITNESS WHEREOF)',
        r'^[A-Z][A-Z\s]{10,}\s*$',  # ALL CAPS headers
    ]
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.pattern = re.compile('|'.join(self.SECTION_PATTERNS), re.MULTILINE)
    
    def chunk(self, text: str, **kwargs) -> List[str]:
        sections = []
        
        # Find section boundaries
        matches = list(self.pattern.finditer(text))
        
        if not matches:
            # No sections found, fall back to sentence chunking
            return SentenceChunker(self.max_chunk_size).chunk(text)
        
        # Extract sections
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            
            # Split large sections
            if len(section_text.split()) > self.max_chunk_size:
                sub_chunks = SentenceChunker(self.max_chunk_size).chunk(section_text)
                sections.extend(sub_chunks)
            else:
                sections.append(section_text)
        
        return sections


class LegalChunker(ChunkingStrategy):
    """
    Legal-specific chunking combining multiple strategies.
    
    Features:
    - Section-aware chunking
    - Citation preservation
    - Clause boundary detection
    """
    
    def __init__(self, target_size: int = 400, max_size: int = 600, overlap: int = 50):
        self.target_size = target_size
        self.max_size = max_size
        self.overlap = overlap
    
    def chunk(self, text: str, **kwargs) -> List[str]:
        # First try section-based
        section_chunks = SectionChunker(self.max_size).chunk(text)
        
        # Then ensure each chunk is appropriately sized
        final_chunks = []
        for section in section_chunks:
            word_count = len(section.split())
            
            if word_count <= self.max_size:
                final_chunks.append(section)
            else:
                # Further split large sections
                sub_chunks = SentenceChunker(self.target_size, self.overlap).chunk(section)
                final_chunks.extend(sub_chunks)
        
        return final_chunks


# ============================================================
# CITATION EXTRACTION
# ============================================================

class CitationExtractor:
    """
    Extract legal citations from text.
    
    Supports:
    - Case citations (e.g., "123 F.3d 456")
    - Statute citations (e.g., "42 U.S.C. § 1983")
    - Regulation citations (e.g., "17 C.F.R. § 240.10b-5")
    """
    
    PATTERNS = {
        'case': r'\d+\s+[A-Z][A-Za-z.]+\s*(?:\d+[a-z]*\s+)?\d+(?:\s*\([^)]+\))?',
        'statute': r'\d+\s+U\.?S\.?C\.?\s*§\s*\d+[a-z]*',
        'regulation': r'\d+\s+C\.?F\.?R\.?\s*§\s*\d+(?:\.\d+)?',
        'section': r'§\s*\d+(?:\.\d+)?(?:\([a-z]\))?',
    }
    
    def __init__(self):
        self.compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PATTERNS.items()
        }
    
    def extract(self, text: str) -> List[Dict[str, str]]:
        """Extract all citations from text."""
        citations = []
        
        for citation_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                citations.append({
                    "type": citation_type,
                    "citation": match.strip(),
                })
        
        return citations
    
    def extract_unique(self, text: str) -> List[str]:
        """Extract unique citation strings."""
        citations = self.extract(text)
        return list(set(c["citation"] for c in citations))


# ============================================================
# DOCUMENT PROCESSOR
# ============================================================

class DocumentProcessor:
    """
    Complete document processing pipeline.
    
    Workflow:
        1. Load document
        2. Extract metadata
        3. Chunk text
        4. Extract citations
        5. Generate chunk IDs
    
    Example:
        >>> processor = DocumentProcessor()
        >>> doc, chunks = processor.process("contract.pdf")
    """
    
    def __init__(
        self,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        extract_citations: bool = True
    ):
        self.chunker = chunking_strategy or LegalChunker()
        self.extract_citations = extract_citations
        self.citation_extractor = CitationExtractor()
    
    def _generate_id(self, content: str) -> str:
        """Generate deterministic ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _detect_document_type(self, text: str, filename: str) -> str:
        """Detect legal document type."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        if 'contract' in filename_lower or 'agreement' in text_lower[:500]:
            return 'contract'
        elif 'brief' in filename_lower or 'court' in text_lower[:500]:
            return 'brief'
        elif any(x in text_lower[:500] for x in ['§', 'u.s.c.', 'statute']):
            return 'statute'
        elif 'regulation' in filename_lower or 'c.f.r.' in text_lower:
            return 'regulation'
        else:
            return 'legal_document'
    
    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from chunk if present."""
        match = re.search(r'\[PAGE (\d+)\]', text)
        if match:
            return int(match.group(1))
        return None
    
    def process(
        self,
        file_path: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[DocumentMetadata, List[TextChunk]]:
        """
        Process a document and return metadata and chunks.
        
        Args:
            file_path: Path to document file
            additional_metadata: Optional extra metadata
        
        Returns:
            Tuple of (DocumentMetadata, List[TextChunk])
        """
        path = Path(file_path)
        
        # Load document
        full_text, loader_metadata = DocumentLoader.load(file_path)
        
        # Create document metadata
        doc_id = self._generate_id(full_text + path.name)
        doc_metadata = DocumentMetadata(
            document_id=doc_id,
            filename=path.name,
            document_type=self._detect_document_type(full_text, path.name),
            page_count=loader_metadata.get("page_count", 0),
            word_count=loader_metadata.get("word_count", 0),
            title=loader_metadata.get("title"),
        )
        
        # Merge additional metadata
        if additional_metadata:
            for key, value in additional_metadata.items():
                if hasattr(doc_metadata, key):
                    setattr(doc_metadata, key, value)
        
        # Chunk document
        chunk_texts = self.chunker.chunk(full_text)
        
        # Create TextChunk objects
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunk_id = self._generate_id(f"{doc_id}_{idx}_{chunk_text[:50]}")
            
            citations = []
            if self.extract_citations:
                citations = self.citation_extractor.extract_unique(chunk_text)
            
            chunk = TextChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                content=chunk_text,
                chunk_index=idx,
                page_number=self._extract_page_number(chunk_text),
                citations=citations,
                token_count=len(chunk_text.split()) * 1.3,  # Rough estimate
            )
            chunks.append(chunk)
        
        logger.info(f"Processed {path.name}: {len(chunks)} chunks")
        return doc_metadata, chunks
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True
    ) -> Tuple[List[DocumentMetadata], List[TextChunk]]:
        """Process all documents in a directory."""
        all_metadata = []
        all_chunks = []
        
        path = Path(directory)
        pattern = '**/*' if recursive else '*'
        
        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in DocumentLoader.supported_extensions():
                try:
                    doc_meta, chunks = self.process(str(file_path))
                    all_metadata.append(doc_meta)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        return all_metadata, all_chunks


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'DocumentMetadata', 'TextChunk',
    'PDFLoader', 'DOCXLoader', 'TextLoader', 'DocumentLoader',
    'ChunkingStrategy', 'SentenceChunker', 'SectionChunker', 'LegalChunker',
    'CitationExtractor',
    'DocumentProcessor',
]
