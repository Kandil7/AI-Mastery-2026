# src/chunking/strategies/semantic.py
from __future__ import annotations

import re
from typing import List

from src.retrieval import Document

from ..base import BaseChunker
from ..config import ChunkingConfig
from ..spans import ChunkSpan, TextSpan
from ..tokenizer import CharCounter
from ..sanitize import sanitize_text


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
        self.counter = CharCounter()  # Use character counter by default

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Chunk document using semantic boundaries.

        Args:
            document: Document to chunk

        Returns:
            List of chunked documents
        """
        # Sanitize the input text
        sanitized_content = sanitize_text(
            document.content,
            max_chars=self.config.max_document_chars,
            strip_control_chars=self.config.strip_control_chars,
            normalize_newlines=self.config.normalize_newlines
        )
        
        # First, split by paragraphs
        paragraphs = self._split_by_paragraphs(sanitized_content)

        chunks = []
        chunk_index = 0

        for paragraph in paragraphs:
            # If paragraph is small enough, add as single chunk
            if self.counter.count(paragraph) <= self.config.chunk_size:
                # Calculate the span for this paragraph in the original document
                start_pos = sanitized_content.find(paragraph)
                end_pos = start_pos + len(paragraph)
                
                chunk_span = ChunkSpan(text=paragraph, span=TextSpan(start_pos, end_pos))
                chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index)
                chunks.append(chunk_doc)
                chunk_index += 1
                continue

            # Otherwise, split paragraph into sentences and group them
            sentences = self._split_by_sentences(paragraph)

            current_chunk = ""
            current_start = 0
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence

                if self.counter.count(test_chunk) <= self.config.chunk_size:
                    current_chunk = test_chunk
                else:
                    # If current chunk has content, save it
                    if current_chunk:
                        # Calculate the span for this chunk in the paragraph
                        start_pos = paragraph.find(current_chunk)
                        end_pos = start_pos + len(current_chunk)
                        
                        chunk_span = ChunkSpan(text=current_chunk.strip(), span=TextSpan(start_pos, end_pos))
                        chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index)
                        chunks.append(chunk_doc)
                        chunk_index += 1

                    # If the sentence itself is larger than chunk size,
                    # we need to further split it
                    if self.counter.count(sentence) > self.config.chunk_size:
                        sub_chunks = self._split_large_sentence(sentence)
                        for sub_chunk in sub_chunks:
                            # Calculate span for sub-chunk
                            sub_start = paragraph.find(sub_chunk)
                            sub_end = sub_start + len(sub_chunk)
                            
                            chunk_span = ChunkSpan(text=sub_chunk, span=TextSpan(sub_start, sub_end))
                            chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index)
                            chunks.append(chunk_doc)
                            chunk_index += 1
                    else:
                        current_chunk = sentence

            # Add any remaining content as a chunk
            if current_chunk:
                # Calculate the span for this remaining chunk
                start_pos = paragraph.find(current_chunk)
                end_pos = start_pos + len(current_chunk)
                
                chunk_span = ChunkSpan(text=current_chunk.strip(), span=TextSpan(start_pos, end_pos))
                chunk_doc = self._create_chunk_document(document, chunk_span, chunk_index)
                chunks.append(chunk_doc)
                chunk_index += 1

        # Apply overlap to the chunks
        chunks = self._apply_overlap_to_semantic_chunks(chunks)

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

                if self.counter.count(test_chunk) <= self.config.chunk_size:
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
                if self.counter.count(chunk) > self.config.chunk_size:
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
    
    def _apply_overlap_to_semantic_chunks(self, chunks: List[Document]) -> List[Document]:
        """Apply overlap to semantic chunks."""
        if not chunks or self.config.chunk_overlap <= 0:
            return chunks
        
        # For semantic chunks, we'll add overlap by appending content from previous chunks
        result = [chunks[0]]  # First chunk remains the same
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = result[-1]  # Use the last chunk in result (which may have overlap)
            
            # Extract the overlap from the previous chunk
            prev_content = previous_chunk.content
            prev_tokens = len(prev_content)  # Using character count as tokens for simplicity
            overlap_size = min(self.config.chunk_overlap, prev_tokens)
            
            if overlap_size > 0:
                overlap_text = prev_content[-overlap_size:]
                new_content = overlap_text + current_chunk.content
            else:
                new_content = current_chunk.content
            
            # Create a new document with the overlapped content
            new_doc = Document(
                id=current_chunk.id,
                content=new_content,
                source=current_chunk.source,
                doc_type=current_chunk.doc_type,
                metadata={
                    **current_chunk.metadata,
                    "has_overlap": True,
                    "overlap_from": previous_chunk.id
                },
                created_at=current_chunk.created_at,
                updated_at=current_chunk.updated_at,
                access_control=current_chunk.access_control,
                page_number=current_chunk.page_number,
                section_title=current_chunk.section_title
            )
            
            result.append(new_doc)
        
        return result