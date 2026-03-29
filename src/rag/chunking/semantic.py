"""
Semantic Chunking Strategy

Implements embedding-based semantic boundary detection.

Best for:
- High-value documents where context preservation is critical
- Documents without clear structural markers
- When semantic coherence is more important than speed

How it works:
1. Split text into sentences
2. Generate embeddings for each sentence
3. Calculate similarity between adjacent sentences
4. Find breakpoints where similarity drops significantly
5. Group sentences between breakpoints into chunks

Limitations:
- Slower (requires embedding generation)
- More expensive (API calls or model inference)
- Requires embedding model availability

Example:
    >>> from src.rag.chunking import SemanticChunker, ChunkingConfig
    >>> config = ChunkingConfig(
    ...     chunk_size=512,
    ...     similarity_threshold=0.5,
    ... )
    >>> chunker = SemanticChunker(config)
    >>> chunks = chunker.chunk({"id": "doc1", "content": "Long text..."})
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import BaseChunker, Chunk, ChunkingConfig

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using embedding similarity.

    This strategy identifies semantic boundaries by analyzing
    embedding similarity between adjacent text units (sentences).
    When similarity drops below a threshold, a chunk boundary
    is created.

    Attributes:
        config: Chunking configuration
        embedding_function: Optional custom embedding function
        _embedding_model: Cached embedding model

    Example:
        >>> def custom_embed(text: str) -> List[float]:
        ...     # Your embedding logic here
        ...     return [0.1, 0.2, 0.3, ...]
        >>> chunker = SemanticChunker(
        ...     ChunkingConfig(similarity_threshold=0.5),
        ...     embedding_function=custom_embed
        ... )
    """

    # Sentence ending patterns for multiple languages
    SENTENCE_PATTERNS = [
        r'(?<=[.!?])\s+',           # Standard sentence endings
        r'(?<=[.!?]["\'])\s+',      # With quotes
        r'(?<=[۔])\s+',             # Arabic full stop
        r'(?<=[؛])\s+',             # Arabic semicolon
        r'(?<=[؟])\s+',             # Arabic question mark
        r'(?<=\n)\s*',              # Newlines
    ]

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        """
        Initialize the semantic chunker.

        Args:
            config: Chunking configuration
            embedding_function: Optional custom embedding function.
                If not provided, will attempt to load sentence-transformers.
        """
        super().__init__(config)
        self._embedding_function = embedding_function
        self._embedding_model = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def _load_embedding_model(self) -> Optional[Any]:
        """
        Lazily load the embedding model.

        Attempts to load sentence-transformers for multilingual support.

        Returns:
            Loaded model or None if unavailable
        """
        if self._embedding_function:
            return self._embedding_function

        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.config.embedding_model
            )

            def embed_fn(texts: List[str]) -> List[List[float]]:
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = self._embedding_model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                return embeddings.tolist()

            self._embedding_function = embed_fn
            self._logger.info(
                f"Loaded embedding model: {self.config.embedding_model}"
            )

            return self._embedding_function

        except ImportError:
            self._logger.warning(
                "sentence-transformers not installed. "
                "Falling back to recursive chunking."
            )
            return None
        except Exception as e:
            self._logger.warning(f"Failed to load embedding model: {e}")
            return None

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Split document using semantic boundary detection.

        Args:
            document: Document dictionary with 'id' and 'content' fields

        Returns:
            List of Chunk objects with semantic coherence

        Raises:
            ValueError: If document is invalid
        """
        self._validate_document(document)

        content = self._clean_text(document.get("content", ""))
        doc_id = document.get("id", "unknown")

        # Split into sentences
        sentences = self._split_sentences(content)

        # Handle edge cases
        if len(sentences) <= 1:
            self._logger.debug(
                f"Document {doc_id} has {len(sentences)} sentence(s), "
                "returning as single chunk"
            )
            return [self._create_chunk(
                content=content,
                document=document,
                start_index=0,
                end_index=len(content),
                extra_metadata={"sentence_count": len(sentences)},
            )]

        # Try to get embeddings
        embed_fn = self._load_embedding_model()

        if not embed_fn:
            # Fall back to recursive chunking
            self._logger.info(
                f"Falling back to recursive chunking for {doc_id}"
            )
            from .recursive import RecursiveChunker
            recursive_chunker = RecursiveChunker(self.config)
            return recursive_chunker.chunk(document)

        # Generate embeddings for sentences
        try:
            embeddings = self._generate_embeddings(sentences, embed_fn)
        except Exception as e:
            self._logger.warning(
                f"Embedding generation failed for {doc_id}: {e}. "
                "Using recursive chunking."
            )
            from .recursive import RecursiveChunker
            recursive_chunker = RecursiveChunker(self.config)
            return recursive_chunker.chunk(document)

        # Find semantic breakpoints
        breakpoints = self._find_breakpoints(sentences, embeddings)

        # Group sentences into chunks
        chunks = self._group_sentences(
            sentences=sentences,
            breakpoints=breakpoints,
            document=document,
            content=content,
        )

        self._logger.info(
            f"Created {len(chunks)} semantic chunks from {len(sentences)} "
            f"sentences for document {doc_id}"
        )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses regex patterns for multiple languages.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Combine patterns
        pattern = "|".join(self.SENTENCE_PATTERNS)

        # Split
        sentences = re.split(pattern, text)

        # Clean and filter
        sentences = [
            s.strip()
            for s in sentences
            if s.strip() and len(s.strip()) > 1
        ]

        return sentences

    def _generate_embeddings(
        self,
        sentences: List[str],
        embed_fn: Callable[[List[str]], List[List[float]]],
    ) -> List[List[float]]:
        """
        Generate embeddings for sentences.

        Args:
            sentences: List of sentences
            embed_fn: Embedding function

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches to avoid memory issues
        batch_size = 32

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            try:
                batch_embeddings = embed_fn(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                self._logger.warning(
                    f"Failed to embed batch {i // batch_size}: {e}"
                )
                # Use zero vector as fallback
                zero_vec = [0.0] * 384
                embeddings.extend([zero_vec] * len(batch))

        return embeddings

    def _find_breakpoints(
        self,
        sentences: List[str],
        embeddings: List[List[float]],
    ) -> List[int]:
        """
        Find indices where semantic shifts occur.

        Uses cosine similarity between adjacent sentence embeddings.
        A breakpoint is created when similarity drops below threshold.

        Args:
            sentences: List of sentences
            embeddings: List of sentence embeddings

        Returns:
            List of breakpoint indices
        """
        if len(embeddings) < 2:
            return [0, len(sentences)]

        # Calculate similarities between adjacent sentences
        similarities: List[float] = []

        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(
                embeddings[i],
                embeddings[i + 1]
            )
            similarities.append(sim)

        if not similarities:
            return [0, len(sentences)]

        # Find threshold using percentile
        import statistics

        try:
            # Use configurable threshold or dynamic percentile
            threshold = self.config.similarity_threshold

            # Alternative: dynamic threshold based on distribution
            if threshold == 0.5:  # Default, use dynamic
                mean_sim = statistics.mean(similarities)
                std_sim = statistics.stdev(similarities) if len(similarities) > 1 else 0
                threshold = max(0.3, mean_sim - std_sim)

        except Exception:
            threshold = 0.5

        # Find breakpoints
        breakpoints = [0]  # Always start at beginning

        for i, sim in enumerate(similarities):
            if sim < threshold:
                # Check chunk size constraints
                prev_breakpoint = breakpoints[-1]
                chunk_sentences = sentences[prev_breakpoint:i + 1]
                chunk_tokens = sum(
                    self.token_counter.count(s) for s in chunk_sentences
                )

                # Create breakpoint if chunk is getting large
                if chunk_tokens >= self.config.chunk_size * 0.8:
                    breakpoints.append(i + 1)

        breakpoints.append(len(sentences))  # Always end at end

        return sorted(set(breakpoints))

    def _group_sentences(
        self,
        sentences: List[str],
        breakpoints: List[int],
        document: Dict[str, Any],
        content: str,
    ) -> List[Chunk]:
        """
        Group sentences into chunks based on breakpoints.

        Args:
            sentences: List of sentences
            breakpoints: List of breakpoint indices
            document: Source document
            content: Original content for index calculation

        Returns:
            List of Chunk objects
        """
        chunks = []

        for i in range(len(breakpoints) - 1):
            start_idx = breakpoints[i]
            end_idx = breakpoints[i + 1]

            chunk_sentences = sentences[start_idx:end_idx]

            if not chunk_sentences:
                continue

            chunk_text = " ".join(chunk_sentences)

            # Handle oversized chunks
            chunk_tokens = self.token_counter.count(chunk_text)

            if chunk_tokens > self.config.max_chunk_size:
                # Further split oversized chunks
                sub_chunks = self._split_oversized_chunk(
                    chunk_text,
                    document,
                    content,
                    chunk_sentences,
                )
                chunks.extend(sub_chunks)
            else:
                # Find character indices
                char_start = content.find(chunk_text)
                if char_start < 0:
                    # Fallback: estimate position
                    char_start = sum(
                        len(s) + 1
                        for s in sentences[:start_idx]
                    )

                char_end = char_start + len(chunk_text)

                chunk = self._create_chunk(
                    content=chunk_text,
                    document=document,
                    start_index=char_start,
                    end_index=char_end,
                    extra_metadata={
                        "sentence_count": len(chunk_sentences),
                        "chunk_method_detail": "semantic",
                        "avg_similarity": self._calculate_avg_similarity(
                            sentences[start_idx:end_idx]
                        ),
                    },
                )
                chunks.append(chunk)

        return chunks

    def _split_oversized_chunk(
        self,
        chunk_text: str,
        document: Dict[str, Any],
        content: str,
        sentences: List[str],
    ) -> List[Chunk]:
        """
        Split an oversized chunk using recursive strategy.

        Args:
            chunk_text: Text to split
            document: Source document
            content: Original content
            sentences: Sentences in the chunk

        Returns:
            List of smaller chunks
        """
        from .recursive import RecursiveChunker

        recursive_chunker = RecursiveChunker(
            ChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        )

        temp_doc = {
            "id": document.get("id", "unknown"),
            "content": chunk_text,
            "metadata": document.get("metadata", {}),
        }

        sub_chunks = recursive_chunker.chunk(temp_doc)

        # Update metadata to indicate semantic parent
        for chunk in sub_chunks:
            chunk.metadata["semantic_parent"] = True
            chunk.metadata["original_sentence_count"] = len(sentences)

        return sub_chunks

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (0 to 1)
        """
        if not a or not b:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _calculate_avg_similarity(
        self,
        sentences: List[str],
    ) -> float:
        """
        Calculate average similarity within a group of sentences.

        Args:
            sentences: List of sentences

        Returns:
            Average pairwise similarity
        """
        if len(sentences) < 2:
            return 1.0

        embed_fn = self._load_embedding_model()
        if not embed_fn:
            return 0.5

        try:
            embeddings = self._generate_embeddings(sentences, embed_fn)

            similarities = []
            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(
                    embeddings[i],
                    embeddings[i + 1]
                )
                similarities.append(sim)

            if similarities:
                return sum(similarities) / len(similarities)

        except Exception:
            pass

        return 0.5


def create_semantic_chunker(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    similarity_threshold: float = 0.5,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    embedding_function: Optional[Callable[[str], List[float]]] = None,
) -> SemanticChunker:
    """
    Factory function to create a SemanticChunker.

    Args:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        similarity_threshold: Threshold for semantic boundary (0-1)
        embedding_model: Model name for sentence-transformers
        embedding_function: Optional custom embedding function

    Returns:
        Configured SemanticChunker instance

    Example:
        >>> chunker = create_semantic_chunker(
        ...     similarity_threshold=0.3,  # More chunks
        ...     embedding_model="all-MiniLM-L6-v2"  # Faster model
        ... )
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
    )

    return SemanticChunker(config, embedding_function=embedding_function)
