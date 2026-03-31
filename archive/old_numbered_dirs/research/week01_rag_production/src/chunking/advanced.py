# src/chunking/advanced.py
from __future__ import annotations

import logging
from typing import List

from src.retrieval import Document

from .config import ChunkingConfig
from .analyze import choose_strategy
from .factory import ChunkerFactory


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
        strategy = choose_strategy(document)

        # Create appropriate chunker
        chunker = ChunkerFactory.create_chunker(strategy, self.config)

        self.logger.info(f"Using {strategy.value} chunking strategy for document {document.id}")

        # Chunk the document
        return chunker.chunk_document(document)