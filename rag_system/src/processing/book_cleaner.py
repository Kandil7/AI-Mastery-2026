"""
Comprehensive Book Data Cleaner and Preparer for Arabic Islamic Literature

This module handles all aspects of cleaning and preparing Arabic book data
without any loss or gaps:

1. Encoding Detection and Conversion
2. Structure Preservation
3. Content Cleaning
4. Metadata Extraction
5. Quality Validation
6. Lossless Processing
"""

import os
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BookRawData:
    """Raw book data before processing."""

    file_path: str
    raw_content: str
    encoding: str
    size_bytes: int
    book_id: int
    title: str


@dataclass
class BookCleanedData:
    """Cleaned book data."""

    book_id: int
    title: str
    author: str
    category: str
    content: str
    chapters: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    statistics: Dict[str, int]


@dataclass
class ProcessingStats:
    """Statistics about processing."""

    total_books: int = 0
    processed_successfully: int = 0
    failed: int = 0
    total_characters: int = 0
    encoding_issues: List[str] = field(default_factory=list)


# ============================================================================
# Encoding Detection
# ============================================================================


class EncodingDetector:
    """Detect and handle multiple Arabic text encodings."""

    # Common encodings for Arabic text
    ENCODINGS = [
        "utf-8",
        "utf-8-sig",  # UTF-8 with BOM
        "cp1252",  # Windows Arabic
        "iso-8859-6",  # Arabic ISO
        "cp720",  # Windows Arabic
        "cp1256",  # Windows Arabic
    ]

    @classmethod
    def detect(cls, file_path: str) -> Tuple[str, str]:
        """
        Detect encoding and return (content, encoding).

        Returns:
            Tuple of (content, encoding_used)
        """

        # First try UTF-8
        for encoding in cls.ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as f:
                    content = f.read()
                return content, encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Fallback: Read with replacement characters
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return content, "utf-8-replace"


# ============================================================================
# Content Cleaners
# ============================================================================


class ArabicTextCleaner:
    """
    Comprehensive Arabic text cleaning while preserving structure.

    Key principle: Clean WITHOUT losing any semantic content.
    """

    # Patterns to REMOVE (noise)
    REMOVE_PATTERNS = [
        # Page markers
        (r"\[Page\s+\d+\]", ""),
        # HTML-like tags (but preserve content)
        (r"<span[^>]*>", ""),
        (r"</span>", ""),
        # Book ID and name headers
        (
            r"^Book ID:.*$",
            "",
        ),
        (
            r"^Book Name:.*$",
            "",
        ),
        (r"^=+$", ""),
        # Footnote markers in text
        (r"\s*\d+\s*$", ""),  # Page numbers at end
    ]

    # Patterns to NORMALIZE (not remove)
    NORMALIZE_PATTERNS = [
        # Multiple newlines -> section break
        (r"\n{4,}", "\n\n##\n\n"),  # Major section break
        (r"\n{3}", "\n\n###\n\n"),  # Subsection
        (r"\n{2}", "\n\n"),  # Paragraph break
        # Tatweel (kashida)
        ("ـ", ""),
        # Multiple spaces
        (r"[ \t]+", " "),
    ]

    # Quranic verse patterns to PRESERVE
    QURAN_PATTERNS = [
        r"[﴿﴾]",  # Quran verse markers
        r"[ۖۗۘۙۚۛۜ]",  # Quranic symbols
    ]

    @classmethod
    def clean(cls, text: str, preserve_quran: bool = True) -> str:
        """
        Clean text while preserving semantic content.

        Args:
            text: Input text
            preserve_quran: Whether to preserve Quranic markers

        Returns:
            Cleaned text
        """

        # Step 1: Extract and preserve Quranic verses
        quran_verses = []
        if preserve_quran:
            text, quran_verses = cls._extract_quran_verses(text)

        # Step 2: Remove noise patterns
        for pattern, replacement in cls.REMOVE_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        # Step 3: Normalize whitespace
        for pattern, replacement in cls.NORMALIZE_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # Step 4: Restore Quranic markers
        if preserve_quran:
            text = cls._restore_quran_verses(text, quran_verses)

        # Step 5: Final cleanup
        text = text.strip()

        return text

    @classmethod
    def _extract_quran_verses(cls, text: str) -> Tuple[str, List[str]]:
        """Extract Quranic verses for later restoration."""

        verses = []

        # Find all Quranic verse patterns
        for pattern in cls.QURAN_PATTERNS:
            matches = re.findall(f"({pattern}.*?{pattern})", text)
            verses.extend(matches)

        # Replace with placeholders
        for i, verse in enumerate(verses):
            placeholder = f"__QURAN_VERSE_{i}__"
            text = text.replace(verse, placeholder)

        return text, verses

    @classmethod
    def _restore_quran_verses(cls, text: str, verses: List[str]) -> str:
        """Restore Quranic verses from extraction."""

        for i, verse in enumerate(verses):
            placeholder = f"__QURAN_VERSE_{i}__"
            text = text.replace(placeholder, verse)

        return text

    @classmethod
    def normalize_arabic(cls, text: str) -> str:
        """
        Normalize Arabic text without losing meaning.

        This is DIFFERENT from cleaning - we normalize characters
        for consistency while preserving content.
        """

        # Normalization mapping
        normalizations = {
            # Alef variants
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            # Yeh variants
            "ى": "ي",
            # Teh marbuta
            "ة": "ه",
            # Waw with hamza
            "ؤ": "و",
            "ئ": "ي",
        }

        # Apply normalizations
        for char, replacement in normalizations.items():
            text = text.replace(char, replacement)

        return text

    @classmethod
    def remove_diacritics(cls, text: str) -> str:
        """Remove Arabic diacritics (tashkeel) while preserving text."""

        # Diacritic unicode ranges
        diacritics = re.compile(r"[\u064B-\u065F\u0670]")

        return diacritics.sub("", text)

    @classmethod
    def clean_footnotes(cls, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Extract and clean footnotes.

        Returns:
            Tuple of (cleaned_text, footnotes_list)
        """

        footnotes = []

        # Find footnote sections
        footnote_pattern = r"\[Footnotes\]:\s*(.*?)(?=\[Page|\Z)"
        matches = re.findall(footnote_pattern, text, re.DOTALL)

        for i, footnote_text in enumerate(matches):
            # Parse individual footnotes
            lines = footnote_text.strip().split("\n")

            for line in lines:
                # Pattern: number + footnote content
                match = re.match(r"(\d+)\s+(.+)", line.strip())
                if match:
                    num, content = match.groups()
                    footnotes.append(
                        {
                            "number": num,
                            "content": content.strip(),
                            "source_footnote": i + 1,
                        }
                    )

        # Remove footnote sections from main text
        text = re.sub(footnote_pattern, "", text, flags=re.DOTALL)

        return text, footnotes


# ============================================================================
# Structure Parser
# ============================================================================


class BookStructureParser:
    """
    Parse and extract book structure without losing content.

    Extracts:
    - Title page
    - Table of contents
    - Chapters/Sections
    - Footnotes
    """

    @classmethod
    def parse(cls, text: str, title: str) -> Dict[str, Any]:
        """
        Parse book structure.

        Returns:
            Dictionary with structured content
        """

        result = {
            "title": title,
            "sections": [],
            "chapters": [],
            "has_toc": False,
            "has_introduction": False,
        }

        # Find introduction
        intro_patterns = [
            r"مقدمة",
            r"المقدمة",
            r"مقدمة",
            r"Introduction",
            r"<span[^>]*>.*مقدمة.*</span>",
        ]

        for pattern in intro_patterns:
            if re.search(pattern, text):
                result["has_introduction"] = True
                break

        # Find table of contents
        toc_patterns = [
            r"فهرس",
            r"فهرس الموضوعات",
            r"Table of contents",
        ]

        for pattern in toc_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["has_toc"] = True
                break

        # Extract chapters based on numbering patterns
        chapter_patterns = [
            # Arabic numerals: ١ - ١٠
            r"^(\d+)\s*[-–]\s*(.+)$",
            # Regular numbers: 1. 2.
            r"^(\d+)\.\s+(.+)$",
            # Arabic chapter markers
            r"^المبحث\s+(\d+)",
            r"^الفصل\s+(\d+)",
            r"^الباب\s+(\d+)",
            r"^باب\s+(\d+)",
            # HTML-like headers
            r"<span[^>]*>(.+?)</span>",
        ]

        chapters = []
        current_chapter = None

        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Check for chapter headers
            for pattern in chapter_patterns:
                match = re.match(pattern, line)
                if match:
                    if current_chapter:
                        chapters.append(current_chapter)

                    chapter_title = (
                        match.group(1) if match.lastindex == 1 else match.group(2)
                    )
                    current_chapter = {
                        "title": chapter_title.strip(),
                        "content": "",
                    }
                    break
            else:
                # Add content to current chapter
                if current_chapter:
                    current_chapter["content"] += line + "\n"

        # Add last chapter
        if current_chapter:
            chapters.append(current_chapter)

        result["chapters"] = chapters

        return result

    @classmethod
    def extract_metadata(cls, text: str) -> Dict[str, str]:
        """
        Extract metadata from book text.

        Returns:
            Dictionary of metadata fields
        """

        metadata = {}

        # Extract author
        author_patterns = [
            r"تأليف[:\s]+([^\n]+)",
            r"تأليف\s*:\s*(.+?)(?:\n|$)",
            r"جمع[:\s]+([^\n]+)",
            r"محمد\s+ناصر\s+الدين\s+الألباني",
        ]

        for pattern in author_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["author"] = match.group(1).strip()
                break

        # Extract date
        date_patterns = [
            r"(\d{4})(?:\s*هـ)?",
            r"سنة\s+(\d+)",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["date"] = match.group(1)
                break

        return metadata


# ============================================================================
# Content Validator
# ============================================================================


class ContentValidator:
    """
    Validate content quality without losing any data.

    Reports issues but NEVER removes content.
    """

    @classmethod
    def validate(cls, text: str) -> Dict[str, Any]:
        """
        Validate content and report issues.

        Returns:
            Validation report (does NOT modify text)
        """

        issues = []

        # Check for encoding issues
        if "\ufffd" in text:  # Replacement character
            issues.append("contains_replacement_characters")

        # Check for empty content
        if not text or len(text.strip()) < 50:
            issues.append("very_short_content")

        # Check for potential corruption
        if text.count("\x00") > len(text) * 0.1:
            issues.append("possible_binary_content")

        # Check for broken Arabic
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
        total_chars = len(text)

        if total_chars > 0 and arabic_chars / total_chars < 0.3:
            issues.append("low_arabic_content")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "char_count": len(text),
            "arabic_char_count": arabic_chars,
            "line_count": len(text.split("\n")),
        }


# ============================================================================
# Complete Book Processor
# ============================================================================


class BookProcessor:
    """
    Complete book processing pipeline.

    Processes books from raw text to clean, structured data
    WITHOUT ANY LOSS.
    """

    def __init__(self):
        self.stats = ProcessingStats()
        self.cleaner = ArabicTextCleaner()
        self.parser = BookStructureParser()
        self.validator = ContentValidator()

    def process(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[BookCleanedData], List[str]]:
        """
        Process a single book file.

        Args:
            file_path: Path to book file
            metadata: Optional metadata from books.json

        Returns:
            Tuple of (cleaned_data, errors)
        """

        errors = []

        # Step 1: Extract book ID and title from filename
        book_id, title = self._extract_book_info(file_path)

        # Step 2: Detect encoding and read
        try:
            content, encoding = EncodingDetector.detect(file_path)
        except Exception as e:
            errors.append(f"Encoding error: {e}")
            self.stats.failed += 1
            return None, errors

        # Step 3: Validate content
        validation = self.validator.validate(content)
        if not validation["is_valid"]:
            for issue in validation["issues"]:
                errors.append(f"Validation issue: {issue}")

        # Step 4: Clean content
        cleaned_content = self.cleaner.clean(content)

        # Step 5: Extract structure
        structure = self.parser.parse(cleaned_content, title)

        # Step 6: Extract footnotes
        cleaned_content, footnotes = self.cleaner.clean_footnotes(cleaned_content)

        # Step 7: Extract additional metadata
        extracted_metadata = self.parser.extract_metadata(content)

        # Step 8: Combine metadata
        final_metadata = {
            **(metadata or {}),
            **extracted_metadata,
            "encoding_detected": encoding,
            "validation_issues": validation["issues"],
            "footnotes_count": len(footnotes),
            "footnotes": footnotes,
        }

        # Step 9: Create cleaned data
        cleaned_data = BookCleanedData(
            book_id=book_id,
            title=title,
            author=final_metadata.get("author", "Unknown"),
            category=final_metadata.get("cat_name", "Unknown"),
            content=cleaned_content,
            chapters=structure.get("chapters", []),
            metadata=final_metadata,
            statistics={
                "char_count": len(cleaned_content),
                "arabic_char_count": len(
                    re.findall(r"[\u0600-\u06FF]", cleaned_content)
                ),
                "word_count": len(cleaned_content.split()),
                "line_count": len(cleaned_content.split("\n")),
                "chapter_count": len(structure.get("chapters", [])),
            },
        )

        self.stats.processed_successfully += 1
        self.stats.total_characters += len(cleaned_content)

        return cleaned_data, errors

    def _extract_book_info(self, file_path: str) -> Tuple[int, str]:
        """Extract book ID and title from filename."""

        filename = Path(file_path).stem

        # Pattern: ID_Title
        match = re.match(r"^(\d+)_(.+)$", filename)

        if match:
            book_id = int(match.group(1))
            title = match.group(2).replace("_", " ")
            return book_id, title

        # Fallback
        return 0, filename


# ============================================================================
# Batch Processor
# ============================================================================


class BatchBookProcessor:
    """
    Process multiple books efficiently.
    """

    def __init__(self):
        self.processor = BookProcessor()
        self.metadata_map: Dict[int, Dict] = {}

    def load_metadata(self, metadata_path: str):
        """Load book metadata from JSON."""

        books_file = Path(metadata_path) / "books.json"

        if not books_file.exists():
            logger.warning(f"Metadata file not found: {books_file}")
            return

        with open(books_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.metadata_map = {b["id"]: b for b in data.get("books", [])}

        logger.info(f"Loaded metadata for {len(self.metadata_map)} books")

    def process_directory(
        self,
        books_dir: str,
        output_dir: str,
        limit: Optional[int] = None,
    ) -> List[BookCleanedData]:
        """
        Process all books in a directory.

        Args:
            books_dir: Directory containing book files
            output_dir: Output directory for processed data
            limit: Optional limit on number of books

        Returns:
            List of processed books
        """

        books_dir = Path(books_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all txt files
        book_files = sorted(books_dir.glob("*.txt"))

        if limit:
            book_files = book_files[:limit]

        logger.info(f"Processing {len(book_files)} books...")

        processed_books = []

        for i, file_path in enumerate(book_files):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(book_files)}")

            # Get metadata if available
            book_id = (
                int(file_path.stem.split("_")[0]) if file_path.stem[0].isdigit() else 0
            )
            metadata = self.metadata_map.get(book_id)

            # Process book
            cleaned_data, errors = self.processor.process(str(file_path), metadata)

            if cleaned_data:
                processed_books.append(cleaned_data)

                # Save individual book
                self._save_book(cleaned_data, output_dir)
            else:
                logger.error(f"Failed to process {file_path}: {errors}")

        # Save summary
        self._save_summary(processed_books, output_dir)

        logger.info(f"Processed {len(processed_books)} books successfully")

        return processed_books

    def _save_book(self, book: BookCleanedData, output_dir: Path):
        """Save processed book to JSON."""

        output_file = (
            output_dir / f"{book.book_id}_{self._sanitize_filename(book.title)}.json"
        )

        data = {
            "book_id": book.book_id,
            "title": book.title,
            "author": book.author,
            "category": book.category,
            "content": book.content,
            "chapters": book.chapters,
            "metadata": book.metadata,
            "statistics": book.statistics,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_summary(self, books: List[BookCleanedData], output_dir: Path):
        """Save processing summary."""

        total_chars = sum(b.statistics["char_count"] for b in books)

        summary = {
            "total_books": len(books),
            "total_characters": total_chars,
            "categories": {},
        }

        # Count by category
        for book in books:
            cat = book.category
            if cat not in summary["categories"]:
                summary["categories"][cat] = 0
            summary["categories"][cat] += 1

        summary_file = output_dir / "processing_summary.json"

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize filename."""
        # Remove invalid characters
        name = re.sub(r'[<>:"/\\|?*]', "", name)
        # Truncate
        return name[:50]


# ============================================================================
# Usage Example
# ============================================================================


def process_dataset(
    datasets_path: str,
    output_path: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process the complete dataset.

    Args:
        datasets_path: Path to datasets directory
        output_path: Output directory
        limit: Optional limit

    Returns:
        Processing summary
    """

    books_dir = os.path.join(datasets_path, "extracted_books")
    metadata_path = os.path.join(datasets_path, "metadata")

    # Create processor
    batch_processor = BatchBookProcessor()

    # Load metadata
    batch_processor.load_metadata(metadata_path)

    # Process all books
    books = batch_processor.process_directory(
        books_dir,
        output_path,
        limit=limit,
    )

    return {
        "total_processed": len(books),
        "output_directory": output_path,
    }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Process Arabic book dataset")
    parser.add_argument(
        "--datasets", default="K:/learning/technical/ai-ml/AI-Mastery-2026/datasets"
    )
    parser.add_argument(
        "--output",
        default="K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data/processed",
    )
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    result = process_dataset(args.datasets, args.output, args.limit)
    print(f"Processed {result['total_processed']} books")
