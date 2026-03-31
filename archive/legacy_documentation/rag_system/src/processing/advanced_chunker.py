"""
Advanced Chunking Strategies for Production RAG 2026

Following RAG Pipeline Guide 2026 - Phase 2: Chunking Strategies

Implements all major chunking strategies:
1. Fixed-size chunking - Fast, predictable
2. Recursive chunking - Preserves structure (recommended for most cases)
3. Semantic chunking - Context-aware, high quality
4. Late chunking - Global context for complex reasoning
5. Agentic chunking - Query-aware, dynamic

Arabic Text Optimization:
- Proper sentence boundary detection
- Ayah and Hadith preservation
- Poetry verse handling
- Quranic verse boundaries

Usage:
    chunker = AdvancedChunker(strategy="semantic")
    chunks = chunker.chunk(document)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Generator
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# ==================== Data Classes ====================


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    FIXED = "fixed"  # Fast, predictable
    RECURSIVE = "recursive"  # Preserves structure (recommended)
    SEMANTIC = "semantic"  # Context-aware, high quality
    LATE = "late"  # Global context for complex reasoning
    AGENTIC = "agentic"  # Query-aware, dynamic
    ISLAMIC = "islamic"  # Specialized for Islamic texts


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    content: str
    chunk_id: str
    document_id: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields
    tokens: Optional[List[int]] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique chunk ID."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.document_id}_chunk_{content_hash}"

    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.content)


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""

    # Strategy
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    # Size parameters (in tokens or characters)
    chunk_size: int = 512  # Target size
    chunk_overlap: int = 50  # Overlap between chunks

    # Constraints
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

    # Arabic-specific
    preserve_arabic_marks: bool = True
    split_on_sentences: bool = True
    preserve_verses: bool = True  # Quranic verses
    preserve_hadith: bool = True  # Hadith units
    preserve_poetry: bool = True  # Poetry couplets

    # Tokenizer
    tokenizer: str = "cl100k_base"  # tiktoken encoding


# ==================== Tokenizers ====================


class Tokenizer:
    """Tokenization utilities for chunking."""

    def __init__(self, tokenizer_name: str = "cl100k_base"):
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding(self.tokenizer_name)
            except ImportError:
                logger.warning("tiktoken not installed, using character-based tokenization")
                self._tokenizer = None
        return self._tokenizer

    def encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        # Fallback: character-based
        return list(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text."""
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        # Fallback: character-based
        return "".join(tokens)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: word-based estimate
        return len(text.split())


# ==================== Base Chunker ====================


class BaseChunker:
    """Base class for chunkers."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.tokenizer = Tokenizer(config.tokenizer)

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """Main chunking entry point."""
        raise NotImplementedError

    def _create_chunk(
        self,
        content: str,
        document: Dict[str, Any],
        start_index: int,
        end_index: int,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """Create a chunk with metadata."""

        metadata = {
            "document_id": document.get("id", "unknown"),
            "source": document.get("source", "unknown"),
            "source_type": document.get("source_type", "unknown"),
            "chunk_method": self.config.strategy.value,
            "word_count": len(content.split()),
            "char_count": len(content),
            **(extra_metadata or {}),
        }

        # Add document metadata
        if "metadata" in document:
            metadata.update({
                f"doc_{k}": v
                for k, v in document["metadata"].items()
            })

        return Chunk(
            content=content,
            chunk_id="",  # Auto-generated
            document_id=document.get("id", "unknown"),
            start_index=start_index,
            end_index=end_index,
            metadata=metadata,
        )


# ==================== Fixed-Size Chunker ====================


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunking.

    Simple, fast, predictable.
    Best for: Speed-critical applications, uniform documents.

    Limitations:
    - May break sentences/contexts
    - No structure awareness
    """

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk document into fixed-size pieces."""

        content = document.get("content", "")
        doc_id = document.get("id", hashlib.md5(content.encode()).hexdigest())

        # Tokenize
        tokens = self.tokenizer.encode(content)

        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Create chunks
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk = self._create_chunk(
                content=chunk_text,
                document=document,
                start_index=i,
                end_index=i + len(chunk_tokens),
                extra_metadata={
                    "token_count": len(chunk_tokens),
                },
            )

            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks


# ==================== Recursive Chunker ====================


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking with hierarchical separators.

    Splits by structure:
    document → sections → subsections → paragraphs → sentences → words

    Best for: Most general-purpose use cases.
    Recommended by RAG Guide 2026 as default strategy.
    """

    # Hierarchical separators (tried in order)
    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Section breaks (###)
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentences
        "! ",      # Sentences (Arabic/English)
        "? ",      # Sentences (Arabic/English)
        "۔ ",      # Arabic sentence separator
        "؛ ",      # Arabic semicolon
        " ",       # Words
        "",        # Characters (last resort)
    ]

    # Arabic-specific separators
    ARABIC_SEPARATORS = [
        "\n\n\n",
        "\n\n",
        "\n",
        "۔ ",  # Arabic full stop
        "؛ ",  # Arabic semicolon
        "؟ ",  # Arabic question mark
        "﴿",  # Quranic verse start (preserved)
        "﴾",   # Quranic verse end
        " ",
        "",
    ]

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk document recursively by structure."""

        content = document.get("content", "")
        doc_id = document.get("id", hashlib.md5(content.encode()).hexdigest())

        # Choose separators based on content
        is_arabic = self._is_arabic(content)
        separators = self.ARABIC_SEPARATORS if is_arabic else self.DEFAULT_SEPARATORS

        # Recursive split
        chunks = self._recursive_split(
            text=content,
            separators=separators,
            doc_id=doc_id,
            document=document,
            start_offset=0,
        )

        logger.info(f"Created {len(chunks)} recursive chunks")
        return chunks

    def _is_arabic(self, text: str) -> bool:
        """Check if text is primarily Arabic."""
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        return arabic_chars / max(len(text), 1) > 0.5

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        doc_id: str,
        document: Dict[str, Any],
        start_offset: int,
    ) -> List[Chunk]:
        """
        Recursively split text by separators.

        Algorithm:
        1. Try to split by first separator
        2. If chunks are too large, recurse with next separator
        3. Merge small adjacent chunks
        4. Split chunks that are still too large
        """

        if not text.strip():
            return []

        # Check if text fits in one chunk
        text_len = self.tokenizer.count_tokens(text)
        if text_len <= self.config.chunk_size:
            return [self._create_chunk(
                content=text,
                document=document,
                start_index=start_offset,
                end_index=start_offset + len(text),
            )]

        # Try splitting by current separator
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []

        # Split while keeping separator
        if separator:
            parts = self._split_keep_separator(text, separator)
        else:
            parts = [text]

        # If only one part, try next separator or force split
        if len(parts) <= 1 and remaining_separators:
            return self._recursive_split(
                text=text,
                separators=remaining_separators,
                doc_id=doc_id,
                document=document,
                start_offset=start_offset,
            )

        # Process each part
        chunks = []
        current_offset = start_offset

        for part in parts:
            if not part.strip():
                current_offset += len(part)
                continue

            part_len = self.tokenizer.count_tokens(part)

            # If part is still too large, recurse
            if part_len > self.config.chunk_size and remaining_separators:
                sub_chunks = self._recursive_split(
                    text=part,
                    separators=remaining_separators,
                    doc_id=doc_id,
                    document=document,
                    start_offset=current_offset,
                )
                chunks.extend(sub_chunks)
            else:
                # Create chunk
                chunk = self._create_chunk(
                    content=part,
                    document=document,
                    start_index=current_offset,
                    end_index=current_offset + len(part),
                )
                chunks.append(chunk)

            current_offset += len(part)

        # Merge small adjacent chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _split_keep_separator(self, text: str, separator: str) -> List[str]:
        """Split text but keep separator with each chunk."""

        if not separator:
            return [text]

        parts = text.split(separator)
        result = []

        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + separator)
            else:
                result.append(part)

        return result

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge adjacent chunks that are too small."""

        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            combined_size = current.word_count + next_chunk.word_count

            if combined_size <= self.config.chunk_size:
                # Merge
                current = self._create_chunk(
                    content=current.content + "\n\n" + next_chunk.content,
                    document={"id": current.document_id},
                    start_index=current.start_index,
                    end_index=next_chunk.end_index,
                    extra_metadata=current.metadata,
                )
            else:
                # Keep separate
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged


# ==================== Semantic Chunker ====================


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using embedding similarity.

    Algorithm:
    1. Split into sentences
    2. Generate embeddings for sentences
    3. Find breakpoints where semantic similarity drops
    4. Group sentences between breakpoints into chunks

    Best for: High-value documents where context preservation is critical.

    Limitations:
    - Slower (requires embedding generation)
    - More expensive
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                )
                logger.info("Loaded embedding model for semantic chunking")
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to recursive chunking")
                self._embedding_model = None
        return self._embedding_model

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk document semantically."""

        content = document.get("content", "")
        doc_id = document.get("id", hashlib.md5(content.encode()).hexdigest())

        # Fall back to recursive if embedding model unavailable
        if not self.embedding_model:
            recursive_chunker = RecursiveChunker(self.config)
            return recursive_chunker.chunk(document)

        # Step 1: Split into sentences
        sentences = self._split_sentences(content)

        if len(sentences) <= 1:
            # Not enough sentences, return as single chunk
            return [self._create_chunk(
                content=content,
                document=document,
                start_index=0,
                end_index=len(content),
            )]

        # Step 2: Generate embeddings
        try:
            embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}, using recursive chunking")
            recursive_chunker = RecursiveChunker(self.config)
            return recursive_chunker.chunk(document)

        # Step 3: Find breakpoints
        breakpoints = self._find_breakpoints(embeddings, sentences)

        # Step 4: Group sentences into chunks
        chunks = self._group_sentences(
            sentences=sentences,
            breakpoints=breakpoints,
            document=document,
            doc_id=doc_id,
        )

        logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (Arabic-aware)."""

        # Arabic sentence patterns
        arabic_patterns = [
            r'۔\s*',  # Arabic full stop
            r'؛\s*',  # Arabic semicolon
            r'؟\s*',  # Arabic question mark
            r'\.\s*',  # English period
            r'!\s*',  # Exclamation
        ]

        # Combine patterns
        pattern = '|'.join(arabic_patterns)

        # Split
        sentences = re.split(pattern, text)

        # Filter empty and clean
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _find_breakpoints(
        self,
        embeddings: List[List[float]],
        sentences: List[str],
    ) -> List[int]:
        """
        Find breakpoints where semantic similarity drops significantly.

        Uses percentile-based thresholding.
        """

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if len(embeddings) < 2:
            return []

        # Calculate similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]
            similarities.append(sim)

        if not similarities:
            return []

        # Find threshold (percentile-based)
        threshold = np.percentile(similarities, self.config.semantic.get(
            'breakpoint_threshold_amount', 25
        ))

        # Find breakpoints where similarity drops below threshold
        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                # Check if chunk would be within size limits
                breakpoints.append(i + 1)

        return breakpoints

    def _group_sentences(
        self,
        sentences: List[str],
        breakpoints: List[int],
        document: Dict[str, Any],
        doc_id: str,
    ) -> List[Chunk]:
        """Group sentences into chunks based on breakpoints."""

        chunks = []
        current_sentences = []
        current_start = 0

        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)

            # Check if we should create a chunk
            current_text = " ".join(current_sentences)
            current_tokens = self.tokenizer.count_tokens(current_text)

            # Create chunk at breakpoint or when chunk is full
            if i + 1 in breakpoints or current_tokens >= self.config.chunk_size:
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunk = self._create_chunk(
                        content=chunk_text,
                        document=document,
                        start_index=current_start,
                        end_index=current_start + len(chunk_text),
                        extra_metadata={
                            "sentence_count": len(current_sentences),
                        },
                    )
                    chunks.append(chunk)

                current_sentences = []
                current_start += len(chunk_text)

        # Handle remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk = self._create_chunk(
                content=chunk_text,
                document=document,
                start_index=current_start,
                end_index=current_start + len(chunk_text),
                extra_metadata={
                    "sentence_count": len(current_sentences),
                },
            )
            chunks.append(chunk)

        return chunks


# ==================== Late Chunker ====================


class LateChunker(BaseChunker):
    """
    Late chunking for global context preservation.

    Algorithm:
    1. Process entire document to get global understanding
    2. Generate document-level embedding/summary
    3. Chunk while preserving connection to global context
    4. Add document summary to each chunk's metadata

    Best for: Complex reasoning, multi-hop QA, documents requiring global understanding.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy load LLM for summarization."""
        if self._llm_client is None:
            # Try to import LLM client
            try:
                from ..generation.generator import LLMClient, LLMProvider
                self._llm_client = LLMClient(provider=LLMProvider.MOCK)
            except ImportError:
                self._llm_client = None
        return self._llm_client

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk document with late chunking strategy."""

        content = document.get("content", "")
        doc_id = document.get("id", hashlib.md5(content.encode()).hexdigest())

        # Step 1: Generate document summary (if LLM available)
        summary = self._generate_summary(content)

        # Step 2: Chunk using recursive strategy
        recursive_chunker = RecursiveChunker(self.config)
        base_chunks = recursive_chunker.chunk(document)

        # Step 3: Add summary to each chunk's metadata
        for chunk in base_chunks:
            chunk.metadata["document_summary"] = summary
            chunk.metadata["chunk_position"] = f"{base_chunks.index(chunk) + 1}/{len(base_chunks)}"

        logger.info(f"Created {len(base_chunks)} late chunks with document summary")
        return base_chunks

    def _generate_summary(self, content: str) -> str:
        """Generate document summary."""

        if not self.llm_client:
            # Fallback: extract first paragraph as summary
            paragraphs = content.split("\n\n")
            return paragraphs[0][:500] if paragraphs else ""

        # Generate summary using LLM
        # (Implementation would use async LLM call)
        return "Document summary would be generated here."


# ==================== Agentic Chunker ====================


class AgenticChunker(BaseChunker):
    """
    Agentic (query-aware) chunking.

    Algorithm:
    1. Receive query along with document
    2. Identify relevant sections based on query
    3. Create chunks optimized for answering the query
    4. Prioritize chunks by relevance

    Best for: Multi-hop queries, complex QA scenarios.

    Limitations:
    - Requires query at chunking time
    - Slower than static chunking
    """

    def chunk(
        self,
        document: Dict[str, Any],
        query: Optional[str] = None,
    ) -> List[Chunk]:
        """Chunk document with query awareness."""

        content = document.get("content", "")
        doc_id = document.get("id", hashlib.md5(content.encode()).hexdigest())

        if not query:
            # Fall back to recursive if no query
            recursive_chunker = RecursiveChunker(self.config)
            return recursive_chunker.chunk(document)

        # Step 1: Identify relevant sections
        relevant_sections = self._identify_relevant_sections(content, query)

        # Step 2: Create chunks from relevant sections
        chunks = []
        for section in relevant_sections:
            chunk = self._create_chunk(
                content=section["content"],
                document=document,
                start_index=section["start"],
                end_index=section["end"],
                extra_metadata={
                    "relevance_score": section["relevance"],
                    "is_query_focused": True,
                },
            )
            chunks.append(chunk)

        # Step 3: Add some context chunks
        context_chunks = self._add_context_chunks(content, doc_id, document, chunks)
        chunks.extend(context_chunks)

        # Sort by relevance
        chunks.sort(
            key=lambda c: c.metadata.get("relevance_score", 0),
            reverse=True,
        )

        logger.info(f"Created {len(chunks)} agentic chunks for query")
        return chunks

    def _identify_relevant_sections(
        self,
        content: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Identify sections relevant to query."""

        # Simple approach: find paragraphs containing query keywords
        query_keywords = set(query.lower().split())

        paragraphs = content.split("\n\n")
        relevant = []

        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            overlap = len(query_keywords & set(para_lower.split()))

            if overlap > 0:
                relevance = overlap / len(query_keywords)
                relevant.append({
                    "content": para,
                    "start": content.find(para),
                    "end": content.find(para) + len(para),
                    "relevance": relevance,
                })

        # Sort by relevance
        relevant.sort(key=lambda x: x["relevance"], reverse=True)

        return relevant[:10]  # Top 10 relevant sections

    def _add_context_chunks(
        self,
        content: str,
        doc_id: str,
        document: Dict[str, Any],
        existing_chunks: List[Chunk],
    ) -> List[Chunk]:
        """Add general context chunks."""

        # Add first paragraph as context
        paragraphs = content.split("\n\n")
        if paragraphs:
            return [self._create_chunk(
                content=paragraphs[0][:500],
                document=document,
                start_index=0,
                end_index=min(len(paragraphs[0]), 500),
                extra_metadata={
                    "is_context": True,
                    "context_type": "introduction",
                },
            )]

        return []


# ==================== Islamic Text Chunker ====================


class IslamicTextChunker(BaseChunker):
    """
    Specialized chunker for Islamic texts.

    Handles:
    - Quranic verses (preserves ayah boundaries)
    - Hadith (preserves isnad + matn units)
    - Fiqh discussions (preserves mas'alah structure)
    - Poetry (preserves bayt couplets)
    """

    # Quranic verse pattern
    VERSE_PATTERN = r'﴿[^﴾]+﴾'

    # Hadith markers
    HADITH_MARKERS = [
        r'حدثنا\s+',
        r'أخبرنا\s+',
        r'قال\s+',
        r'عن\s+',
        r'عن النبي\s+',
        r'صلى الله عليه وسلم',
    ]

    # Fiqh markers
    FIQH_MARKERS = [
        r'مسألة\s+',
        r'فصل\s+',
        r'الباب\s+',
        r'الفصل\s+',
        r'الحكم\s+',
    ]

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """Chunk Islamic text with domain-aware boundaries."""

        content = document.get("content", "")
        doc_id = document.get("id", hashlib.md5(content.encode()).hexdigest())
        category = document.get("metadata", {}).get("category", "")

        # Detect text type
        text_type = self._detect_text_type(content, category)

        # Use appropriate strategy
        if text_type == "quran":
            chunks = self._chunk_quran(content, doc_id, document)
        elif text_type == "hadith":
            chunks = self._chunk_hadith(content, doc_id, document)
        elif text_type == "fiqh":
            chunks = self._chunk_fiqh(content, doc_id, document)
        elif text_type == "poetry":
            chunks = self._chunk_poetry(content, doc_id, document)
        else:
            # Fall back to recursive
            recursive_chunker = RecursiveChunker(self.config)
            chunks = recursive_chunker.chunk(document)

        logger.info(f"Created {len(chunks)} Islamic text chunks (type: {text_type})")
        return chunks

    def _detect_text_type(self, content: str, category: str) -> str:
        """Detect type of Islamic text."""

        # Check category first
        category_lower = category.lower() if category else ""

        if "تفسير" in category_lower or "قرآن" in category_lower:
            return "quran"
        if "حديث" in category_lower or "سنة" in category_lower:
            return "hadith"
        if "فقه" in category_lower:
            return "fiqh"
        if "شعر" in category_lower:
            return "poetry"

        # Check content patterns
        if re.search(self.VERSE_PATTERN, content):
            return "quran"
        if any(re.search(p, content) for p in self.HADITH_MARKERS):
            return "hadith"
        if any(re.search(p, content) for p in self.FIQH_MARKERS):
            return "fiqh"

        return "general"

    def _chunk_quran(
        self,
        content: str,
        doc_id: str,
        document: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk Quranic text preserving verse boundaries."""

        # Find all verses
        verses = re.findall(self.VERSE_PATTERN, content)

        if not verses:
            # No verses found, use recursive
            recursive_chunker = RecursiveChunker(self.config)
            return recursive_chunker.chunk(document)

        # Group verses into chunks
        chunks = []
        current_verses = []
        current_start = 0

        for verse in verses:
            current_verses.append(verse)
            current_text = " ".join(current_verses)

            # Check if chunk is full
            if self.tokenizer.count_tokens(current_text) >= self.config.chunk_size:
                # Create chunk without last verse
                if len(current_verses) > 1:
                    chunk_text = " ".join(current_verses[:-1])
                    chunk = self._create_chunk(
                        content=chunk_text,
                        document=document,
                        start_index=current_start,
                        end_index=current_start + len(chunk_text),
                        extra_metadata={
                            "verse_count": len(current_verses) - 1,
                            "text_type": "quran",
                        },
                    )
                    chunks.append(chunk)
                    current_start += len(chunk_text)

                current_verses = [verse]

        # Handle remaining verses
        if current_verses:
            chunk_text = " ".join(current_verses)
            chunk = self._create_chunk(
                content=chunk_text,
                document=document,
                start_index=current_start,
                end_index=current_start + len(chunk_text),
                extra_metadata={
                    "verse_count": len(current_verses),
                    "text_type": "quran",
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_hadith(
        self,
        content: str,
        doc_id: str,
        document: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk Hadith preserving isnad + matn units."""

        # Split by hadith markers
        hadith_units = []
        current_unit = ""
        current_start = 0

        lines = content.split("\n")
        for i, line in enumerate(lines):
            is_new_hadith = any(
                re.match(pattern, line)
                for pattern in self.HADITH_MARKERS
            )

            if is_new_hadith and current_unit:
                # Save current hadith
                hadith_units.append({
                    "content": current_unit.strip(),
                    "start": current_start,
                    "end": current_start + len(current_unit),
                })
                current_start += len(current_unit)
                current_unit = ""

            current_unit += line + "\n"

        # Add last hadith
        if current_unit.strip():
            hadith_units.append({
                "content": current_unit.strip(),
                "start": current_start,
                "end": current_start + len(current_unit),
            })

        # Create chunks
        chunks = []
        for unit in hadith_units:
            chunk = self._create_chunk(
                content=unit["content"],
                document=document,
                start_index=unit["start"],
                end_index=unit["end"],
                extra_metadata={
                    "text_type": "hadith",
                    "is_complete_hadith": True,
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_fiqh(
        self,
        content: str,
        doc_id: str,
        document: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk Fiqh text preserving mas'alah structure."""

        # Split by fiqh markers
        sections = []
        current_section = ""
        current_start = 0

        lines = content.split("\n")
        for line in lines:
            is_new_section = any(
                re.match(pattern, line)
                for pattern in self.FIQH_MARKERS
            )

            if is_new_section and current_section:
                sections.append({
                    "content": current_section.strip(),
                    "start": current_start,
                    "end": current_start + len(current_section),
                })
                current_start += len(current_section)
                current_section = ""

            current_section += line + "\n"

        if current_section.strip():
            sections.append({
                "content": current_section.strip(),
                "start": current_start,
                "end": current_start + len(current_section),
            })

        # Create chunks
        chunks = []
        for section in sections:
            chunk = self._create_chunk(
                content=section["content"],
                document=document,
                start_index=section["start"],
                end_index=section["end"],
                extra_metadata={
                    "text_type": "fiqh",
                    "is_masalah": True,
                },
            )
            chunks.append(chunk)

        return chunks

    def _chunk_poetry(
        self,
        content: str,
        doc_id: str,
        document: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk poetry preserving bayt (couplet) structure."""

        # Split by lines
        lines = content.split("\n")

        # Group into couplets (bayt)
        couplets = []
        current_couplet = []

        for line in lines:
            if line.strip():
                current_couplet.append(line.strip())

                if len(current_couplet) == 2:
                    couplets.append(" | ".join(current_couplet))
                    current_couplet = []

        # Handle remaining line
        if current_couplet:
            couplets.append(current_couplet[0])

        # Create chunks
        chunks = []
        current_text = ""
        current_start = 0

        for couplet in couplets:
            test_text = current_text + "\n" + couplet if current_text else couplet

            if self.tokenizer.count_tokens(test_text) >= self.config.chunk_size and current_text:
                # Create chunk
                chunk = self._create_chunk(
                    content=current_text,
                    document=document,
                    start_index=current_start,
                    end_index=current_start + len(current_text),
                    extra_metadata={
                        "text_type": "poetry",
                        "is_couplet": True,
                    },
                )
                chunks.append(chunk)
                current_start += len(current_text)
                current_text = ""

            current_text = test_text

        if current_text:
            chunk = self._create_chunk(
                content=current_text,
                document=document,
                start_index=current_start,
                end_index=current_start + len(current_text),
                extra_metadata={
                    "text_type": "poetry",
                    "is_couplet": True,
                },
            )
            chunks.append(chunk)

        return chunks


# ==================== Advanced Chunker Factory ====================


class AdvancedChunker:
    """
    Unified chunker interface supporting all strategies.

    Usage:
        chunker = AdvancedChunker(strategy="semantic")
        chunks = chunker.chunk(document)
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._chunker = self._create_chunker()

    def _create_chunker(self) -> BaseChunker:
        """Create appropriate chunker based on config."""

        strategy = self.config.strategy

        if strategy == ChunkingStrategy.FIXED:
            return FixedSizeChunker(self.config)
        elif strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(self.config)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(self.config)
        elif strategy == ChunkingStrategy.LATE:
            return LateChunker(self.config)
        elif strategy == ChunkingStrategy.AGENTIC:
            return AgenticChunker(self.config)
        elif strategy == ChunkingStrategy.ISLAMIC:
            return IslamicTextChunker(self.config)
        else:
            return RecursiveChunker(self.config)

    def chunk(self, document: Dict[str, Any], **kwargs) -> List[Chunk]:
        """Chunk document using configured strategy."""
        return self._chunker.chunk(document, **kwargs)

    def chunk_texts(
        self,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> List[List[Chunk]]:
        """Chunk multiple texts."""

        all_chunks = []

        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            document = {
                "id": doc_id,
                "content": text,
                "metadata": {},
            }

            chunks = self.chunk(document)
            all_chunks.append(chunks)

        return all_chunks


# ==================== Factory Functions ====================


def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> AdvancedChunker:
    """
    Create a chunker instance.

    Args:
        strategy: Chunking strategy (fixed, recursive, semantic, late, agentic, islamic)
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        **kwargs: Additional config options

    Returns:
        AdvancedChunker instance
    """

    strategy_map = {
        "fixed": ChunkingStrategy.FIXED,
        "recursive": ChunkingStrategy.RECURSIVE,
        "semantic": ChunkingStrategy.SEMANTIC,
        "late": ChunkingStrategy.LATE,
        "agentic": ChunkingStrategy.AGENTIC,
        "islamic": ChunkingStrategy.ISLAMIC,
    }

    config = ChunkingConfig(
        strategy=strategy_map.get(strategy.lower(), ChunkingStrategy.RECURSIVE),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )

    return AdvancedChunker(config)


def get_recommended_chunking(category: str) -> Dict[str, Any]:
    """
    Get recommended chunking settings for a document category.

    Args:
        category: Document category (quran, hadith, fiqh, general, etc.)

    Returns:
        Recommended chunking config
    """

    recommendations = {
        "quran": {
            "strategy": "islamic",
            "chunk_size": 384,
            "chunk_overlap": 30,
        },
        "hadith": {
            "strategy": "islamic",
            "chunk_size": 256,
            "chunk_overlap": 20,
        },
        "fiqh": {
            "strategy": "islamic",
            "chunk_size": 512,
            "chunk_overlap": 40,
        },
        "tafsir": {
            "strategy": "semantic",
            "chunk_size": 768,
            "chunk_overlap": 50,
        },
        "general": {
            "strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
    }

    return recommendations.get(category.lower(), recommendations["general"])


if __name__ == "__main__":
    # Demo
    print("Advanced Chunking Strategies - Demo")
    print("=" * 50)

    # Test document
    test_doc = {
        "id": "test_001",
        "content": """
بسم الله الرحمن الرحيم
هذا نص تجريبي لاختبار استراتيجيات التقطيع.
يتضمن النص فقرات متعددة لاختبار التقطيع العودي.
        """.strip(),
        "metadata": {"category": "general"},
    }

    # Test different strategies
    strategies = ["recursive", "fixed", "semantic"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        chunker = create_chunker(strategy=strategy, chunk_size=100)
        chunks = chunker.chunk(test_doc)

        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  Chunk {i+1}: {chunk.word_count} words, {chunk.char_count} chars")
