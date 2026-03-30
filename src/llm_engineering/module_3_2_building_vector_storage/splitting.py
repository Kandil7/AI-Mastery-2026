"""
Text Splitting Module

Production-ready text splitting strategies:
- Recursive character splitting
- Semantic splitting
- Token-based splitting
- Code-aware splitting

Features:
- Metadata preservation
- Parent-child relationships
- Overlap handling
- Custom separators
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata.

    Attributes:
        content: The text content
        metadata: Chunk metadata
        id: Unique chunk identifier
        parent_id: Optional parent document/chunk ID
        start_idx: Start index in original text
        end_idx: End index in original text
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    parent_id: Optional[str] = None
    start_idx: int = 0
    end_idx: int = 0
    chunk_index: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            parent_id=data.get("parent_id"),
            start_idx=data.get("start_idx", 0),
            end_idx=data.get("end_idx", 0),
            chunk_index=data.get("chunk_index", 0),
        )

    def __len__(self) -> int:
        return len(self.content)


class BaseSplitter(ABC):
    """Abstract base class for text splitters."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator
        self.add_start_index = add_start_index

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

    def split_documents(
        self,
        documents: List[Any],
        parent_id_key: str = "id",
    ) -> List[TextChunk]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects
            parent_id_key: Key to use for parent ID

        Returns:
            List of TextChunk objects
        """
        chunks = []

        for doc in documents:
            text = doc.content if hasattr(doc, "content") else str(doc)
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            parent_id = getattr(doc, "id", None) or metadata.get(parent_id_key)

            text_chunks = self.split_text(text)

            for i, chunk_text in enumerate(text_chunks):
                chunk = TextChunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    parent_id=parent_id,
                    chunk_index=i,
                )
                chunks.append(chunk)

        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks


class RecursiveSplitter(BaseSplitter):
    """
    Recursive character text splitter.

    Splits text by a hierarchy of separators, trying to keep
    chunks at meaningful boundaries (paragraphs, sentences, words).
    """

    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Triple newline
        "\n\n",    # Double newline (paragraph)
        "\n",      # Single newline
        ". ",      # Sentence
        "! ",      # Exclamation
        "? ",      # Question
        " ",       # Word
        "",        # Character
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split_text(self, text: str) -> List[str]:
        """Split text recursively by separators."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting logic."""
        final_chunks = []

        # Get appropriate separator
        separator = separators[-1]
        new_separators = []

        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        # Split by separator
        splits = text.split(separator)

        # Merge splits into chunks
        current_chunk = []
        current_length = 0

        for split in splits:
            split_with_separator = split + (separator if self.keep_separator and separator else "")
            split_length = self.length_function(split_with_separator)

            if current_length + split_length <= self.chunk_size:
                current_chunk.append(split_with_separator)
                current_length += split_length
            else:
                # Current chunk is full
                if current_chunk:
                    chunk_text = "".join(current_chunk)
                    final_chunks.append(chunk_text.strip())

                # Check if single split is too large
                if split_length > self.chunk_size:
                    # Recurse with remaining separators
                    sub_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # Start new chunk
                    current_chunk = [split_with_separator]
                    current_length = split_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            final_chunks.append(chunk_text.strip())

        # Handle overlaps
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._merge_overlaps(final_chunks)

        return final_chunks

    def _merge_overlaps(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks."""
        if not self.chunk_overlap:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-self.chunk_overlap:]

                # Find where overlap ends in current chunk
                if overlap_text in chunk:
                    overlap_end = chunk.find(overlap_text) + len(overlap_text)
                    chunk = chunk[overlap_end:]

            result.append(chunk)

        return result


class TokenSplitter(BaseSplitter):
    """
    Token-based text splitter.

    Uses tokenizers (tiktoken, etc.) for accurate token counting.
    Essential for LLM context window management.
    """

    def __init__(
        self,
        chunk_size: int = 512,  # Tokens
        chunk_overlap: int = 50,  # Tokens
        model_name: str = "gpt-4",
        encoding_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        self.model_name = model_name
        self._tokenizer = None
        self._encoding_name = encoding_name

        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        """Load tokenizer."""
        try:
            import tiktoken

            if self._encoding_name:
                self._tokenizer = tiktoken.get_encoding(self._encoding_name)
            else:
                self._tokenizer = tiktoken.encoding_for_model(self.model_name)

            # Override length function
            self.length_function = self._count_tokens
            logger.info(f"Loaded tokenizer for {self.model_name}")
        except ImportError:
            logger.warning("tiktoken not installed. Using character-based splitting.")
            self._tokenizer = None
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self._tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text) // 4  # Fallback estimate

    def split_text(self, text: str) -> List[str]:
        """Split text by token count."""
        if not self._tokenizer:
            # Fallback to character splitting
            splitter = RecursiveSplitter(
                chunk_size=self.chunk_size * 4,
                chunk_overlap=self.chunk_overlap * 4,
            )
            return splitter.split_text(text)

        tokens = self._tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size

            # Try to end at sentence boundary
            if end < len(tokens):
                # Decode to find sentence boundary
                chunk_tokens = tokens[start:end]
                chunk_text = self._tokenizer.decode(chunk_tokens)

                # Look for sentence endings
                for sep in [". ", "! ", "? ", "\n\n", "\n"]:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        # Adjust end to separator
                        end = start + len(self._tokenizer.encode(chunk_text[:last_sep + len(sep)]))
                        break

            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text.strip())

            start = end - self.chunk_overlap
            if start >= len(tokens):
                break

        return chunks


class SemanticSplitter(BaseSplitter):
    """
    Semantic text splitter.

    Uses embeddings to split text at semantic boundaries,
    keeping related content together.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.5,
        buffer_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size

        self._sentence_splitter = RecursiveSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separators=[". ", "! ", "? ", "\n"],
        )

    def split_text(self, text: str) -> List[str]:
        """Split text at semantic boundaries."""
        if not self.embedding_function:
            # Fallback to recursive splitting
            splitter = RecursiveSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return splitter.split_text(text)

        # First split into sentences
        sentences = self._sentence_splitter.split_text(text)

        if len(sentences) <= 1:
            return [text]

        # Compute embeddings for sentences
        embeddings = []
        for sentence in sentences:
            try:
                embedding = self.embedding_function(sentence)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed sentence: {e}")
                embeddings.append([0.0] * 384)  # Zero vector fallback

        # Find semantic boundaries
        splits = self._find_semantic_splits(sentences, embeddings)

        # Group sentences into chunks
        chunks = self._group_sentences(sentences, splits)

        return chunks

    def _find_semantic_splits(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
    ) -> List[int]:
        """Find indices where semantic shifts occur."""
        splits = [0]  # Always start at beginning

        for i in range(1, len(embeddings) - 1):
            # Compare with neighbors
            prev_sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            next_sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            # Large drop in similarity indicates boundary
            if prev_sim > self.similarity_threshold and next_sim < self.similarity_threshold:
                splits.append(i)

        splits.append(len(sentences))  # Always end at end
        return sorted(set(splits))

    def _group_sentences(
        self,
        sentences: List[str],
        splits: List[int],
    ) -> List[str]:
        """Group sentences into chunks based on splits."""
        chunks = []

        for i in range(len(splits) - 1):
            start = splits[i]
            end = splits[i + 1]

            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)

            # Handle chunk size
            if len(chunk_text) > self.chunk_size:
                # Further split if needed
                sub_splitter = RecursiveSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                sub_chunks = sub_splitter.split_text(chunk_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text.strip())

        return chunks

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class CodeSplitter(BaseSplitter):
    """
    Code-aware text splitter.

    Understands code structure (functions, classes, etc.)
    and splits at appropriate boundaries.
    """

    LANGUAGE_SEPARATORS = {
        "python": [
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "javascript": [
            "\nfunction ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            "\nclass ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "java": [
            "\npublic class ",
            "\nprivate class ",
            "\npublic ",
            "\nprivate ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "cpp": [
            "\nclass ",
            "\nvoid ",
            "\nint ",
            "\nfloat ",
            "\ndouble ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    }

    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.language = language.lower()
        self.separators = self.LANGUAGE_SEPARATORS.get(
            self.language,
            self.LANGUAGE_SEPARATORS["python"],
        )

    def split_text(self, text: str) -> List[str]:
        """Split code at structural boundaries."""
        # Try to detect language from content if not specified
        if self.language == "auto":
            self.language = self._detect_language(text)
            self.separators = self.LANGUAGE_SEPARATORS.get(
                self.language,
                self.LANGUAGE_SEPARATORS["python"],
            )

        # Use recursive splitter with code-aware separators
        splitter = RecursiveSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
        )

        chunks = splitter.split_text(text)

        # Post-process to ensure valid code blocks
        chunks = self._validate_code_chunks(chunks)

        return chunks

    def _detect_language(self, text: str) -> str:
        """Detect programming language from text."""
        # Simple heuristic detection
        if re.search(r"\bdef\s+\w+\s*\(", text):
            return "python"
        elif re.search(r"\bfunction\s+\w+\s*\(", text) or re.search(r"=>", text):
            return "javascript"
        elif re.search(r"\bpublic\s+class\s+", text) or re.search(r"\bSystem\.out\.println", text):
            return "java"
        elif re.search(r"#include\s*<", text) or re.search(r"\bstd::", text):
            return "cpp"
        return "python"  # Default

    def _validate_code_chunks(self, chunks: List[str]) -> List[str]:
        """Validate and fix code chunks."""
        validated = []

        for chunk in chunks:
            # Check for basic syntax issues
            if self._has_balanced_braces(chunk) and self._has_balanced_parens(chunk):
                validated.append(chunk)
            else:
                # Try to fix or skip invalid chunks
                fixed = self._try_fix_chunk(chunk)
                if fixed:
                    validated.append(fixed)

        return validated

    def _has_balanced_braces(self, text: str) -> bool:
        """Check if braces are balanced."""
        count = 0
        in_string = False
        string_char = None

        for char in text:
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif not in_string:
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1
                    if count < 0:
                        return False

        return count == 0

    def _has_balanced_parens(self, text: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        in_string = False

        for char in text:
            if char in '"\'':
                in_string = not in_string
            elif not in_string:
                if char == "(":
                    count += 1
                elif char == ")":
                    count -= 1
                    if count < 0:
                        return False

        return count == 0

    def _try_fix_chunk(self, chunk: str) -> Optional[str]:
        """Try to fix a code chunk."""
        # Simple fix: remove incomplete function/class definitions
        lines = chunk.split("\n")
        fixed_lines = []

        for line in lines:
            if re.match(r"^(def|class|function)\s+\w+", line) and not line.strip().endswith((":","{")):
                # Incomplete definition, skip
                continue
            fixed_lines.append(line)

        fixed = "\n".join(fixed_lines)
        if fixed.strip() and len(fixed) > 10:
            return fixed
        return None


class HierarchicalSplitter:
    """
    Hierarchical splitter with parent-child relationships.

    Creates small chunks for retrieval while maintaining
    references to larger parent chunks for context.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        overlap: int = 100,
        splitter_type: str = "recursive",
    ) -> None:
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap

        if splitter_type == "recursive":
            self._splitter_class = RecursiveSplitter
        elif splitter_type == "token":
            self._splitter_class = TokenSplitter
        elif splitter_type == "semantic":
            self._splitter_class = SemanticSplitter
        else:
            self._splitter_class = RecursiveSplitter

    def split(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[TextChunk], Dict[str, List[str]]]:
        """
        Split text into parent-child chunks.

        Args:
            text: Text to split
            metadata: Optional metadata

        Returns:
            Tuple of (child chunks, parent-child mapping)
        """
        metadata = metadata or {}

        # Create parent chunks
        parent_splitter = self._splitter_class(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.overlap,
        )
        parent_texts = parent_splitter.split_text(text)

        # Create child chunks from each parent
        child_splitter = self._splitter_class(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.overlap // 2,
        )

        children = []
        parent_child_map = {}
        child_index = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_id = hashlib.sha256(parent_text.encode()).hexdigest()[:16]
            parent_child_map[parent_id] = []

            child_texts = child_splitter.split_text(parent_text)

            for child_text in child_texts:
                child = TextChunk(
                    content=child_text,
                    metadata=metadata.copy(),
                    parent_id=parent_id,
                    chunk_index=child_index,
                )
                child.metadata["parent_chunk_size"] = len(parent_text)
                child.metadata["parent_index"] = parent_idx

                children.append(child)
                parent_child_map[parent_id].append(child.id)
                child_index += 1

        logger.info(
            f"Hierarchical split: {len(parent_texts)} parents -> {len(children)} children"
        )

        return children, parent_child_map


class TextSplitterFactory:
    """Factory for creating text splitters."""

    @staticmethod
    def create(
        splitter_type: str = "recursive",
        **kwargs: Any,
    ) -> BaseSplitter:
        """
        Create a text splitter.

        Args:
            splitter_type: Type of splitter
            **kwargs: Splitter configuration

        Returns:
            Configured splitter instance
        """
        splitters = {
            "recursive": RecursiveSplitter,
            "token": TokenSplitter,
            "semantic": SemanticSplitter,
            "code": CodeSplitter,
        }

        splitter_class = splitters.get(splitter_type.lower())
        if not splitter_class:
            raise ValueError(f"Unknown splitter type: {splitter_type}")

        return splitter_class(**kwargs)
