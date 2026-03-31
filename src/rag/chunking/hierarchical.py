"""
Hierarchical Chunking Strategy

Implements parent-child chunk relationships for context-aware retrieval.

Best for:
- Multi-stage retrieval (coarse then fine)
- When you need both precision and context
- Complex documents requiring global understanding

How it works:
1. Create large "parent" chunks for context
2. Split each parent into smaller "child" chunks for retrieval
3. Maintain parent-child mappings
4. Retrieve children, expand to parents for generation

Example:
    >>> from src.rag.chunking import HierarchicalChunker, ChunkingConfig
    >>> config = ChunkingConfig(
    ...     parent_chunk_size=2000,
    ...     child_chunk_size=500,
    ...     chunk_overlap=50,
    ... )
    >>> chunker = HierarchicalChunker(config)
    >>> children, parent_map = chunker.chunk({"id": "doc1", "content": "Long text..."})
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseChunker, Chunk, ChunkingConfig
from .recursive import RecursiveChunker

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalChunkResult:
    """
    Result of hierarchical chunking operation.

    Attributes:
        children: List of child chunks for retrieval
        parents: List of parent chunks for context
        parent_child_map: Mapping from parent IDs to child IDs
        child_parent_map: Mapping from child IDs to parent IDs
    """

    children: List[Chunk]
    parents: List[Chunk]
    parent_child_map: Dict[str, List[str]]
    child_parent_map: Dict[str, str]


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking with parent-child relationships.

    This strategy creates two levels of chunks:
    - Parent chunks: Large, context-rich chunks for generation
    - Child chunks: Small, focused chunks for retrieval

    During retrieval:
    1. Search over child chunks for precision
    2. Retrieve corresponding parent chunks for context
    3. Use parent chunks in generation

    Attributes:
        config: Chunking configuration

    Example:
        >>> chunker = HierarchicalChunker(
        ...     ChunkingConfig(
        ...         parent_chunk_size=2000,
        ...         child_chunk_size=500,
        ...     )
        ... )
        >>> result = chunker.chunk({"id": "doc1", "content": "Long text..."})
        >>> print(f"Created {len(result.children)} children, {len(result.parents)} parents")
    """

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        splitter_type: str = "recursive",
    ) -> None:
        """
        Initialize the hierarchical chunker.

        Args:
            config: Chunking configuration
            splitter_type: Base splitter type ('recursive', 'token', 'semantic')
        """
        super().__init__(config)
        self._splitter_type = splitter_type
        self._logger = logging.getLogger(self.__class__.__name__)

        # Create splitters for parent and child levels
        self._parent_splitter = self._create_splitter(
            self.config.parent_chunk_size
        )
        self._child_splitter = self._create_splitter(
            self.config.child_chunk_size
        )

    def _create_splitter(self, chunk_size: int) -> RecursiveChunker:
        """
        Create a splitter with specified chunk size.

        Args:
            chunk_size: Target chunk size

        Returns:
            Configured RecursiveChunker
        """
        return RecursiveChunker(
            ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                min_chunk_size=self.config.min_chunk_size,
            )
        )

    def chunk(
        self,
        document: Dict[str, Any],
    ) -> HierarchicalChunkResult:
        """
        Split document into hierarchical parent-child chunks.

        Args:
            document: Document dictionary with 'id' and 'content' fields

        Returns:
            HierarchicalChunkResult with children, parents, and mappings

        Raises:
            ValueError: If document is invalid
        """
        self._validate_document(document)

        content = self._clean_text(document.get("content", ""))
        doc_id = document.get("id", "unknown")

        self._logger.debug(
            f"Starting hierarchical chunking for {doc_id}: "
            f"parent_size={self.config.parent_chunk_size}, "
            f"child_size={self.config.child_chunk_size}"
        )

        # Step 1: Create parent chunks
        parent_texts = self._parent_splitter.split_text(content)

        self._logger.debug(
            f"Created {len(parent_texts)} parent chunks for {doc_id}"
        )

        # Step 2: Create child chunks from each parent
        children: List[Chunk] = []
        parents: List[Chunk] = []
        parent_child_map: Dict[str, List[str]] = {}
        child_parent_map: Dict[str, str] = {}

        child_index = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            # Create parent chunk
            parent_id = self._generate_parent_id(doc_id, parent_idx, parent_text)

            parent_chunk = self._create_chunk(
                content=parent_text,
                document=document,
                start_index=self._find_text_position(content, parent_text),
                end_index=0,  # Will be set
                extra_metadata={
                    "chunk_level": "parent",
                    "parent_index": parent_idx,
                    "is_parent": True,
                },
            )
            parent_chunk.chunk_id = parent_id
            parent_chunk.end_index = parent_chunk.start_index + len(parent_text)
            parents.append(parent_chunk)

            # Create child chunks from this parent
            child_texts = self._child_splitter.split_text(parent_text)

            parent_child_map[parent_id] = []

            for child_idx, child_text in enumerate(child_texts):
                if not child_text.strip():
                    continue

                child_id = self._generate_child_id(parent_id, child_idx)

                child_chunk = self._create_chunk(
                    content=child_text,
                    document=document,
                    start_index=0,  # Relative to parent
                    end_index=len(child_text),
                    extra_metadata={
                        "chunk_level": "child",
                        "parent_index": parent_idx,
                        "child_index": child_idx,
                        "is_child": True,
                        "parent_chunk_size": len(parent_text),
                    },
                    parent_id=parent_id,
                )
                child_chunk.chunk_id = child_id
                children.append(child_chunk)

                # Update mappings
                parent_child_map[parent_id].append(child_id)
                child_parent_map[child_id] = parent_id

                child_index += 1

        self._logger.info(
            f"Hierarchical chunking complete for {doc_id}: "
            f"{len(parents)} parents -> {len(children)} children"
        )

        return HierarchicalChunkResult(
            children=children,
            parents=parents,
            parent_child_map=parent_child_map,
            child_parent_map=child_parent_map,
        )

    def chunk_simple(
        self,
        document: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Split document and return only child chunks.

        Convenience method for simple use cases where you only
        need the child chunks (with parent_id references).

        Args:
            document: Document dictionary

        Returns:
            List of child chunks with parent_id references
        """
        result = self.chunk(document)
        return result.children

    def get_parent_for_child(
        self,
        child_id: str,
        parents: List[Chunk],
    ) -> Optional[Chunk]:
        """
        Get the parent chunk for a given child ID.

        Args:
            child_id: Child chunk ID
            parents: List of parent chunks

        Returns:
            Parent chunk or None if not found
        """
        result = self.chunk({"id": "temp", "content": ""})

        parent_id = result.child_parent_map.get(child_id)
        if not parent_id:
            return None

        for parent in parents:
            if parent.chunk_id == parent_id:
                return parent

        return None

    def expand_children_to_parents(
        self,
        child_ids: List[str],
        result: HierarchicalChunkResult,
    ) -> List[Chunk]:
        """
        Expand retrieved child IDs to their parent chunks.

        Args:
            child_ids: List of retrieved child chunk IDs
            result: HierarchicalChunkResult from chunking

        Returns:
            List of unique parent chunks
        """
        parent_ids = set()

        for child_id in child_ids:
            parent_id = result.child_parent_map.get(child_id)
            if parent_id:
                parent_ids.add(parent_id)

        # Get parent chunks
        parents = [
            p for p in result.parents
            if p.chunk_id in parent_ids
        ]

        return parents

    def _generate_parent_id(
        self,
        doc_id: str,
        parent_idx: int,
        content: str,
    ) -> str:
        """
        Generate unique parent chunk ID.

        Args:
            doc_id: Document ID
            parent_idx: Parent chunk index
            content: Parent chunk content

        Returns:
            Unique parent ID
        """
        content_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()[:12]

        return f"{doc_id}_parent_{parent_idx:04d}_{content_hash}"

    def _generate_child_id(
        self,
        parent_id: str,
        child_idx: int,
    ) -> str:
        """
        Generate unique child chunk ID.

        Args:
            parent_id: Parent chunk ID
            child_idx: Child chunk index

        Returns:
            Unique child ID
        """
        return f"{parent_id}_child_{child_idx:04d}"

    def _find_text_position(
        self,
        content: str,
        text: str,
    ) -> int:
        """
        Find position of text within content.

        Args:
            content: Full document content
            text: Text to find

        Returns:
            Character position or 0 if not found
        """
        pos = content.find(text)
        return pos if pos >= 0 else 0


def create_hierarchical_chunker(
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 500,
    chunk_overlap: int = 100,
    min_chunk_size: int = 100,
    splitter_type: str = "recursive",
) -> HierarchicalChunker:
    """
    Factory function to create a HierarchicalChunker.

    Args:
        parent_chunk_size: Size of parent chunks in tokens
        child_chunk_size: Size of child chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        min_chunk_size: Minimum acceptable chunk size
        splitter_type: Base splitter type

    Returns:
        Configured HierarchicalChunker instance

    Example:
        >>> chunker = create_hierarchical_chunker(
        ...     parent_chunk_size=1500,
        ...     child_chunk_size=300,
        ... )
    """
    config = ChunkingConfig(
        chunk_size=child_chunk_size,  # Primary chunk size
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
    )

    return HierarchicalChunker(config, splitter_type=splitter_type)


# Import dataclass at module level
from dataclasses import dataclass

# Re-define after import to ensure proper ordering
@dataclass
class HierarchicalChunkResult:
    """
    Result of hierarchical chunking operation.

    Attributes:
        children: List of child chunks for retrieval
        parents: List of parent chunks for context
        parent_child_map: Mapping from parent IDs to child IDs
        child_parent_map: Mapping from child IDs to parent IDs
    """

    children: List[Chunk]
    parents: List[Chunk]
    parent_child_map: Dict[str, List[str]]
    child_parent_map: Dict[str, str]

    def get_parents_for_children(self, child_ids: List[str]) -> List[Chunk]:
        """
        Get parent chunks for a list of child IDs.

        Args:
            child_ids: List of child chunk IDs

        Returns:
            List of unique parent chunks
        """
        parent_ids = set()

        for child_id in child_ids:
            parent_id = self.child_parent_map.get(child_id)
            if parent_id:
                parent_ids.add(parent_id)

        return [p for p in self.parents if p.chunk_id in parent_ids]

    def get_children_for_parent(self, parent_id: str) -> List[Chunk]:
        """
        Get child chunks for a parent ID.

        Args:
            parent_id: Parent chunk ID

        Returns:
            List of child chunks
        """
        child_ids = self.parent_child_map.get(parent_id, [])
        return [c for c in self.children if c.chunk_id in child_ids]
