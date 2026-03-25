"""
Arabic text processing utilities for the RAG system.
"""

import re
import json
from typing import List, Tuple, Optional
from pathlib import Path


# Arabic character normalization mapping
ARABIC_NORMALIZATION = {
    # Normalize different forms of alef
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    # Normalize teh marbuta
    "ة": "ه",
    # Normalize yeh
    "ى": "ي",
    # Normalize teh
    "ة": "ه",
    # Remove diacritics (tashkeel)
    "ُ": "",
    "َ": "",
    "ِ": "",
    "ّ": "",
    "ً": "",
    "ٍ": "",
    "ٌ": "",
    "ْ": "",
    # Normalize waw with hamza
    "ؤ": "و",
    "ئ": "ي",
}

# Arabic punctuation marks to preserve
ARABIC_PUNCTUATION = set("،؛؟!.…")

# Common Arabic stopwords
ARABIC_STOPWORDS = set(
    [
        # Particles
        "في",
        "من",
        "إلى",
        "على",
        "عن",
        "مع",
        "بين",
        "منذ",
        "until",
        "until",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "هؤلاء",
        "أولئك",
        "الذي",
        "التي",
        "اللذين",
        "اللتين",
        "الذين",
        "اللاتي",
        "ما",
        "من",
        "هل",
        "لا",
        "لن",
        "لم",
        "إن",
        "أن",
        "كان",
        "كانت",
        "و",
        "أو",
        "ثم",
        "بل",
        "لكن",
        "لو",
        "لولا",
        "لما",
        "ب",
        "ك",
        "ل",
        "ف",
        "س",
        "will",
        "will",
        "قد",
        "أصبح",
        "أمسى",
        "ظل",
        "بات",
        "صار",
        "ليس",
        "ما",
        # Common verbs
        "قال",
        "قالوا",
        "قال النبي",
        "قال صلى الله عليه وسلم",
        "ذكر",
        "روى",
        "أخبر",
        "حدث",
        "ثبت",
        # Pronouns
        "أنا",
        "أنت",
        "أنتم",
        "نحن",
        "هم",
        "هن",
        "هو",
        "هي",
        # Prepositions
        "في",
        "من",
        "إلى",
        "على",
        "عن",
        "مع",
        "ب",
        "ل",
        "ك",
        "ف",
        "until",
    ]
)


class ArabicTextProcessor:
    """Process Arabic text for embedding and retrieval."""

    def __init__(
        self,
        remove_diacritics: bool = True,
        normalize_arabic: bool = True,
        remove_stopwords: bool = False,
        remove_tatweel: bool = True,
    ):
        self.remove_diacritics = remove_diacritics
        self.normalize_arabic = normalize_arabic
        self.remove_stopwords = remove_stopwords
        self.remove_tatweel = remove_tatweel

        # Compile regex patterns
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns."""
        # Pattern for Arabic text (letters + spaces + punctuation)
        self.arabic_pattern = re.compile(r"[\u0600-\u06FF\s\p{P}]+")

        # Pattern for non-Arabic text
        self.non_arabic_pattern = re.compile(r"[^\u0600-\u06FF]")

        # Pattern for tatweel (kashida)
        self.tatweel_pattern = re.compile(r"ـ+")

        # Pattern for multiple spaces
        self.multi_space_pattern = re.compile(r"\s+")

        # Sentence endings
        self.sentence_endings = re.compile(r"[.!?…؛؟]")

    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Input Arabic text

        Returns:
            Normalized Arabic text
        """
        if not text:
            return ""

        # Remove tatweel (kashida)
        if self.remove_tatweel:
            text = self.tatweel_pattern.sub("", text)

        # Remove diacritics (tashkeel)
        if self.remove_diacritics:
            for diacritic, replacement in ARABIC_NORMALIZATION.items():
                if len(diacritic) > 1:  # Skip simple replacements
                    text = text.replace(diacritic, replacement)
            # Remove remaining diacritic marks
            text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

        # Normalize Arabic characters
        if self.normalize_arabic:
            for char, replacement in ARABIC_NORMALIZATION.items():
                if len(char) == 1:
                    text = text.replace(char, replacement)

        # Normalize whitespace
        text = self.multi_space_pattern.sub(" ", text).strip()

        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove Arabic stopwords from text."""
        words = text.split()
        filtered_words = [w for w in words if w not in ARABIC_STOPWORDS]
        return " ".join(filtered_words)

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment Arabic text into sentences.

        Args:
            text: Input Arabic text

        Returns:
            List of sentences
        """
        # Split on sentence endings
        sentences = self.sentence_endings.split(text)

        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Arabic text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Normalize first
        text = self.normalize(text)

        # Split on whitespace and punctuation
        tokens = re.findall(r"[\u0600-\u06FF]+", text)

        # Optionally remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in ARABIC_STOPWORDS]

        return tokens

    def is_arabic(self, text: str) -> bool:
        """
        Check if text contains Arabic characters.

        Args:
            text: Input text

        Returns:
            True if text contains Arabic
        """
        return bool(re.search(r"[\u0600-\u06FF]", text))

    def get_text_stats(self, text: str) -> dict:
        """
        Get statistics about Arabic text.

        Args:
            text: Input text

        Returns:
            Dictionary with text statistics
        """
        return {
            "char_count": len(text),
            "arabic_char_count": len(re.findall(r"[\u0600-\u06FF]", text)),
            "word_count": len(self.tokenize(text)),
            "sentence_count": len(self.segment_sentences(text)),
            "is_arabic": self.is_arabic(text),
        }


class ArabicChunker:
    """Chunk Arabic text for RAG processing."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
        preserve_sentences: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.preserve_sentences = preserve_sentences
        self.text_processor = ArabicTextProcessor()

    def chunk_by_chars(
        self,
        text: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
    ) -> List[dict]:
        """
        Chunk text by character count.

        Args:
            text: Input text
            book_id: Book ID
            book_title: Book title
            author: Author name
            category: Category name

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        text_length = len(text)

        start = 0
        chunk_idx = 0

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            # Try to break at sentence boundary if preserving sentences
            if self.preserve_sentences and end < text_length:
                # Look for sentence ending before end
                sentence_end = text.rfind("。", start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind("؟", start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind("!", start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind(".", start, end)
                if sentence_end > start + self.min_chunk_size:
                    end = sentence_end + 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunk = {
                    "chunk_id": f"{book_id}_{chunk_idx}",
                    "book_id": book_id,
                    "book_title": book_title,
                    "author": author,
                    "category": category,
                    "content": chunk_text,
                    "start_char": start,
                    "end_char": end,
                }
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap
            chunk_idx += 1

        return chunks

    def chunk_by_semantic(
        self,
        text: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
        embeddings: Optional[List[float]] = None,
    ) -> List[dict]:
        """
        Chunk text semantically based on embeddings.
        Uses simple heuristic if embeddings not provided.

        Args:
            text: Input text
            book_id: Book ID
            book_title: Book title
            author: Author name
            category: Category name
            embeddings: Optional pre-computed embeddings

        Returns:
            List of chunk dictionaries
        """
        # For now, use sentence-based chunking
        # In production, could use embedding-based semantic chunking
        sentences = self.text_processor.segment_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        start_char = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            # If single sentence is too long, chunk it
            if sentence_length > self.chunk_size:
                # First, add current accumulated chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(
                        {
                            "chunk_id": f"{book_id}_{chunk_idx}",
                            "book_id": book_id,
                            "book_title": book_title,
                            "author": author,
                            "category": category,
                            "content": chunk_text,
                            "start_char": start_char,
                            "end_char": start_char + len(chunk_text),
                        }
                    )
                    chunk_idx += 1
                    start_char += len(chunk_text) + 1
                    current_chunk = []
                    current_length = 0

                # Then chunk the long sentence
                sub_chunks = self._chunk_long_sentence(
                    sentence,
                    book_id,
                    book_title,
                    author,
                    category,
                    chunk_idx,
                    start_char,
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                start_char = (
                    sum(len(s["content"]) for s in sub_chunks)
                    + start_char
                    + len(sub_chunks)
                )
                continue

            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        {
                            "chunk_id": f"{book_id}_{chunk_idx}",
                            "book_id": book_id,
                            "book_title": book_title,
                            "author": author,
                            "category": category,
                            "content": chunk_text,
                            "start_char": start_char,
                            "end_char": start_char + len(chunk_text),
                        }
                    )
                    chunk_idx += 1

                # Start new chunk with overlap
                overlap_text = (
                    " ".join(current_chunk[-2:]) if len(current_chunk) >= 2 else ""
                )
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = (
                    len(overlap_text) + len(sentence) if overlap_text else len(sentence)
                )
                start_char = (
                    start_char + len(chunk_text) - len(overlap_text)
                    if overlap_text
                    else start_char
                )
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    {
                        "chunk_id": f"{book_id}_{chunk_idx}",
                        "book_id": book_id,
                        "book_title": book_title,
                        "author": author,
                        "category": category,
                        "content": chunk_text,
                        "start_char": start_char,
                        "end_char": start_char + len(chunk_text),
                    }
                )

        return chunks

    def _chunk_long_sentence(
        self,
        sentence: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
        start_idx: int,
        start_char: int,
    ) -> List[dict]:
        """Chunk a long sentence into smaller pieces."""
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_length = 0
        idx = start_idx
        char_pos = start_char

        for word in words:
            word_length = len(word)

            if current_length + word_length > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "chunk_id": f"{book_id}_{idx}",
                        "book_id": book_id,
                        "book_title": book_title,
                        "author": author,
                        "category": category,
                        "content": chunk_text,
                        "start_char": char_pos,
                        "end_char": char_pos + len(chunk_text),
                    }
                )
                idx += 1
                char_pos += len(chunk_text) + 1
                current_chunk = []
                current_length = 0

            current_chunk.append(word)
            current_length += word_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "chunk_id": f"{book_id}_{idx}",
                    "book_id": book_id,
                    "book_title": book_title,
                    "author": author,
                    "category": category,
                    "content": chunk_text,
                    "start_char": char_pos,
                    "end_char": char_pos + len(chunk_text),
                }
            )

        return chunks


# Utility functions
def load_arabic_stopwords(filepath: str) -> set:
    """Load Arabic stopwords from file."""
    stopwords = set(ARABIC_STOPWORDS)

    if Path(filepath).exists():
        with open(filepath, "r", encoding="utf-8") as f:
            custom_stopwords = f.read().strip().split("\n")
            stopwords.update(custom_stopwords)

    return stopwords
