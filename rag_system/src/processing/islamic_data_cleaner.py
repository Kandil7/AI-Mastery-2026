"""
Comprehensive Data Cleaning and Preparation Pipeline for Islamic Literature

This module handles the complete data pipeline:
1. File reading with proper encoding handling
2. Metadata extraction from text headers
3. Content cleaning (removing structural markers, fixing encoding)
4. Preserving all content without loss
5. Arabic text normalization
6. Validation and quality checks
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class BookMetadata:
    """Metadata for a book in the dataset."""

    book_id: int
    title: str
    category_id: int
    category_name: str
    author_ids: List[int]
    author_names: List[str]
    date: int
    file_name: str
    guid: str
    short_id: str
    size_mb: float
    book_type: int  # 1=original, 4=research, etc.

    # Computed
    content_hash: str = ""
    raw_text: str = ""
    cleaned_text: str = ""
    word_count: int = 0

    # Quality indicators
    encoding_issues: List[str] = field(default_factory=list)
    cleaning_applied: List[str] = field(default_factory=list)


@dataclass
class CleaningReport:
    """Report of cleaning operations."""

    total_files: int = 0
    successful: int = 0
    failed: int = 0
    encoding_errors: int = 0

    issues_by_type: Dict[str, int] = field(default_factory=dict)
    files_with_issues: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Configuration
# ============================================================================

# Structural markers to remove (not content)
MARKERS_TO_REMOVE = [
    r"\[Page\s+\d+\]",
    r"\[Footnotes\]:",
    r"\[Footnotes\]",
    r"<span[^>]*>.*?</span>",
    r"Book ID:\s*\d+",
    r"Book Name:.*",
    r"=+{20,}",
]

# Patterns that indicate structure but might contain content
STRUCTURE_PATTERNS = [
    r"\[Page\s+(\d+)\]",
    r"\[Footnotes?\](.*?)(?=\[Page|\Z)",
]

# Arabic text normalization rules
ARABIC_NORMALIZATION = {
    # Alef variants
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    # Ta marbuta
    "ة": "ه",
    # Yaa variants
    "ى": "ي",
    "ئ": "ي",
    # Waw variants
    "ؤ": "و",
    # Remove diacritics (tashkeel)
    "ً": "",
    "ٌ": "",
    "ٍ": "",
    "َ": "",
    "ُ": "",
    "ِ": "",
    "ّ": "",
    "ْ": "",
    "ٰ": "",
    # Normalize whitespace
    "\u200b": "",  # Zero width space
    "\u200c": "",  # ZWNJ - keep for now
    "\u200d": "",  # ZWJ - keep for now
    "\u200e": "",  # LRM
    "\u200f": "",  # RLM
    "\ufeff": "",  # BOM
}

# Valid encodings to try
ENCODINGS_TO_TRY = [
    "utf-8",
    "utf-8-sig",
    "windows-1256",
    "iso-8859-6",
    "cp720",
    "cp1252",
]


# ============================================================================
# Core Functions
# ============================================================================


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for consistent processing."""

    if not text:
        return ""

    result = text

    # Apply normalization rules
    for old, new in ARABIC_NORMALIZATION.items():
        result = result.replace(old, new)

    # Remove excessive whitespace
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)

    return result.strip()


def remove_structural_markers(text: str) -> Tuple[str, List[str]]:
    """Remove structural markers from text while preserving content."""

    applied = []
    result = text

    # Remove page markers
    result = re.sub(r"\[Page\s+\d+\]", "", result)
    applied.append("page_markers")

    # Remove HTML-like spans
    result = re.sub(r"<span[^>]*>", "", result)
    result = result.replace("</span>", "")
    applied.append("html_spans")

    # Remove separator lines
    result = re.sub(r"=+\s*$", "", result, flags=re.MULTILINE)
    applied.append("separator_lines")

    return result, applied


def detect_encoding_issues(text: str) -> List[str]:
    """Detect encoding issues in text."""

    issues = []

    # Check for replacement characters
    if "\ufffd" in text:
        issues.append("replacement_character")

    # Check for unusual control characters
    unusual_chars = re.findall(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", text)
    if unusual_chars:
        issues.append(f"control_chars_{len(unusual_chars)}")

    # Check for broken Arabic (common pattern)
    if re.search(r"[إأآٱ][^إأآٱا-ي\s]", text):
        issues.append("possible_encoding")

    # Check for malformed numbers
    if re.search(r"\d+\.\d+\.\d+\.\d+", text):
        issues.append("ip_like_numbers")

    return issues


def safe_read_file(file_path: str) -> Tuple[Optional[str], List[str]]:
    """
    Safely read a file with multiple encoding attempts.

    Returns:
        Tuple of (text, list of issues encountered)
    """

    issues = []

    # Try each encoding
    for encoding in ENCODINGS_TO_TRY:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            # Check if content looks valid
            if content and len(content) > 10:
                # Try to detect encoding issues
                detected_issues = detect_encoding_issues(content)
                if detected_issues:
                    issues.extend(detected_issues)

                return content, issues

        except UnicodeDecodeError as e:
            issues.append(f"encoding_{encoding}_failed")
            continue
        except Exception as e:
            issues.append(f"read_error: {str(e)}")
            return None, issues

    # Last resort: try binary read with error handling
    try:
        with open(file_path, "rb") as f:
            raw = f.read()

        # Try to decode with errors='replace'
        content = raw.decode("utf-8", errors="replace")
        issues.append("fallback_to_replace")
        return content, issues

    except Exception as e:
        issues.append(f"binary_read_failed: {str(e)}")
        return None, issues


def extract_book_header_info(text: str) -> Dict[str, Any]:
    """Extract book ID and title from the file header."""

    result = {"book_id": None, "title": None, "header_found": False}

    # Look for header pattern at the start
    lines = text.split("\n")[:10]

    for line in lines:
        # Match "Book ID: 123"
        match = re.match(r"Book ID:\s*(\d+)", line)
        if match:
            result["book_id"] = int(match.group(1))
            result["header_found"] = True

        # Match "Book Name: ..."
        match = re.match(r"Book Name:\s*(.+)", line)
        if match:
            result["title"] = match.group(1).strip()

    return result


def clean_islamic_text(
    text: str, preserve_structure: bool = False
) -> Tuple[str, List[str]]:
    """
    Clean Islamic Arabic text while preserving all meaningful content.

    Args:
        text: Raw text to clean
        preserve_structure: Whether to preserve structural markers

    Returns:
        Tuple of (cleaned_text, list of operations applied)
    """

    if not text:
        return "", ["empty_input"]

    operations = []
    result = text

    # Step 1: Extract and verify header
    header_info = extract_book_header_info(result)
    if header_info["header_found"]:
        operations.append("header_extracted")

    # Step 2: Remove structural markers if not preserving
    if not preserve_structure:
        result, marker_ops = remove_structural_markers(result)
        operations.extend(marker_ops)

    # Step 3: Normalize Arabic text
    result = normalize_arabic_text(result)
    operations.append("arabic_normalized")

    # Step 4: Remove excessive whitespace but preserve paragraphs
    # Keep double newlines as paragraph breaks
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()
    operations.append("whitespace_normalized")

    return result, operations


def validate_content(
    original: str, cleaned: str, min_word_count: int = 10
) -> Dict[str, Any]:
    """Validate that cleaning didn't lose significant content."""

    issues = []

    # Check word count
    original_words = len(original.split())
    cleaned_words = len(cleaned.split())

    loss_percentage = (original_words - cleaned_words) / max(original_words, 1) * 100

    if loss_percentage > 20:
        issues.append(f"high_loss_{loss_percentage:.1f}%")

    if cleaned_words < min_word_count:
        issues.append(f"too_short_{cleaned_words}")

    # Check for empty result
    if not cleaned:
        issues.append("empty_result")

    return {
        "valid": len(issues) == 0,
        "original_words": original_words,
        "cleaned_words": cleaned_words,
        "loss_percentage": loss_percentage,
        "issues": issues,
    }


def compute_text_hash(text: str) -> str:
    """Compute hash of text content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ============================================================================
# Main Processing Classes
# ============================================================================


class IslamicTextCleaner:
    """
    Comprehensive cleaner for Islamic Arabic texts.

    Handles:
    - Multiple encodings
    - Structural markers
    - Arabic normalization
    - Content validation
    """

    def __init__(self, preserve_structure: bool = False):
        self.preserve_structure = preserve_structure
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "encoding_issues": 0,
            "content_loss_warnings": 0,
        }

    def process_file(
        self,
        file_path: str,
        expected_book_id: Optional[int] = None,
        expected_title: Optional[str] = None,
    ) -> Optional[BookMetadata]:
        """
        Process a single book file.

        Args:
            file_path: Path to the book file
            expected_book_id: Expected book ID from metadata
            expected_title: Expected title from metadata

        Returns:
            BookMetadata if successful, None if failed
        """

        self.stats["total_processed"] += 1

        # Read file
        raw_content, read_issues = safe_read_file(file_path)

        if not raw_content:
            logger.warning(f"Failed to read: {file_path} - Issues: {read_issues}")
            self.stats["failed"] += 1
            return None

        if read_issues:
            self.stats["encoding_issues"] += len(read_issues)

        # Extract header info
        header_info = extract_book_header_info(raw_content)

        # Clean the text
        cleaned_content, clean_ops = clean_islamic_text(
            raw_content, preserve_structure=self.preserve_structure
        )

        # Validate
        validation = validate_content(raw_content, cleaned_content)

        if not validation["valid"]:
            logger.warning(f"Validation issues for {file_path}: {validation['issues']}")
            self.stats["content_loss_warnings"] += 1

        # Extract book ID from filename or header
        book_id = expected_book_id
        if not book_id:
            # Try to extract from filename (format: 123_title.txt)
            match = re.match(r"^(\d+)_", os.path.basename(file_path))
            if match:
                book_id = int(match.group(1))

        # Get title
        title = expected_title or header_info.get("title", "")

        # Create metadata
        metadata = BookMetadata(
            book_id=book_id or 0,
            title=title or os.path.basename(file_path),
            category_id=0,  # Will be filled from books.json
            category_name="",
            author_ids=[],
            author_names=[],
            date=99999,
            file_name=os.path.basename(file_path),
            guid="",
            short_id="",
            size_mb=os.path.getsize(file_path) / (1024 * 1024),
            book_type=1,
            raw_text=raw_content,
            cleaned_text=cleaned_content,
            word_count=len(cleaned_content.split()),
            content_hash=compute_text_hash(cleaned_content),
            encoding_issues=read_issues,
            cleaning_applied=clean_ops,
        )

        self.stats["successful"] += 1
        return metadata

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()


class DatasetLoader:
    """
    Load and manage the complete Islamic literature dataset.

    Integrates:
    - Extracted book files
    - Metadata from JSON files
    - Author information
    - Category mappings
    """

    def __init__(self, dataset_path: str, metadata_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path)
        self.books_dir = self.dataset_path / "extracted_books"
        self.metadata_path = metadata_path or str(
            self.dataset_path / "metadata" / "books.json"
        )

        # Load metadata
        self.books_metadata: Dict[int, Dict] = {}
        self.authors_metadata: Dict[int, Dict] = {}
        self.categories: Dict[int, str] = {}

        self._load_metadata()

    def _load_metadata(self):
        """Load all metadata files."""

        # Load books metadata
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                books_data = json.load(f)

            for book in books_data.get("books", []):
                self.books_metadata[book["id"]] = book

                # Load category
                self.categories[book["cat_id"]] = book.get("cat_name", "")

            logger.info(f"Loaded metadata for {len(self.books_metadata)} books")

        except Exception as e:
            logger.error(f"Failed to load books metadata: {e}")

        # Load authors metadata
        authors_path = str(Path(self.metadata_path).parent / "authors.json")
        try:
            with open(authors_path, "r", encoding="utf-8") as f:
                authors_data = json.load(f)

            for author in authors_data.get("authors", []):
                self.authors_metadata[author["id"]] = author

            logger.info(f"Loaded metadata for {len(self.authors_metadata)} authors")

        except Exception as e:
            logger.warning(f"Failed to load authors metadata: {e}")

    def get_book_metadata(self, book_id: int) -> Optional[Dict]:
        """Get metadata for a specific book."""
        return self.books_metadata.get(book_id)

    def get_author_name(self, author_id: int) -> Optional[str]:
        """Get author name by ID."""
        author = self.authors_metadata.get(author_id)
        return author.get("name") if author else None

    def get_category_name(self, category_id: int) -> str:
        """Get category name by ID."""
        return self.categories.get(category_id, "غير محدد")

    def get_all_book_files(self) -> List[Tuple[int, str]]:
        """
        Get all book files with their IDs.

        Returns:
            List of (book_id, file_path) tuples
        """

        files = []

        for file_path in self.books_dir.glob("*.txt"):
            # Extract book ID from filename
            match = re.match(r"^(\d+)_", file_path.name)
            if match:
                book_id = int(match.group(1))
                files.append((book_id, str(file_path)))

        return sorted(files, key=lambda x: x[0])

    def enrich_book_metadata(self, metadata: BookMetadata) -> BookMetadata:
        """Enrich basic metadata with full metadata from JSON."""

        # Get full metadata from books.json
        book_data = self.books_metadata.get(metadata.book_id)

        if book_data:
            metadata.category_id = book_data.get("cat_id", 0)
            metadata.category_name = book_data.get("cat_name", "")
            metadata.guid = book_data.get("guid", "")
            metadata.short_id = book_data.get("short_id", "")
            metadata.date = book_data.get("date", 99999)
            metadata.book_type = book_data.get("type", 1)

            # Get author info
            for author in book_data.get("authors", []):
                metadata.author_ids.append(author.get("id", 0))
                metadata.author_names.append(author.get("name", ""))

        return metadata


class CompleteDataProcessor:
    """
    Complete data processing pipeline.

    Processes all books in the dataset with:
    1. File reading
    2. Text cleaning
    3. Metadata enrichment
    4. Quality validation
    5. Chunk preparation for indexing
    """

    def __init__(
        self, dataset_path: str, output_dir: str, preserve_structure: bool = False
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.loader = DatasetLoader(dataset_path)
        self.cleaner = IslamicTextCleaner(preserve_structure)

        # Processing results
        self.processed_books: Dict[int, BookMetadata] = {}
        self.failed_books: List[Dict[str, Any]] = []

    def process_all_books(
        self, limit: Optional[int] = None, batch_size: int = 100
    ) -> CleaningReport:
        """
        Process all books in the dataset.

        Args:
            limit: Maximum number of books to process
            batch_size: Number of books to process before saving checkpoint

        Returns:
            CleaningReport with processing statistics
        """

        report = CleaningReport()

        # Get all book files
        book_files = self.loader.get_all_book_files()

        if limit:
            book_files = book_files[:limit]

        report.total_files = len(book_files)

        logger.info(f"Processing {len(book_files)} books...")

        for i, (book_id, file_path) in enumerate(book_files):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(book_files)}")

            # Get expected metadata
            expected_meta = self.loader.get_book_metadata(book_id)
            expected_title = expected_meta.get("title") if expected_meta else None

            # Process file
            result = self.cleaner.process_file(
                file_path, expected_book_id=book_id, expected_title=expected_title
            )

            if result:
                # Enrich with metadata
                result = self.loader.enrich_book_metadata(result)
                self.processed_books[book_id] = result
                report.successful += 1
            else:
                self.failed_books.append({"book_id": book_id, "file_path": file_path})
                report.failed += 1

                # Track specific issues
                for issue in self.cleaner.stats.get("encoding_issues", []):
                    report.issues_by_type[issue] = (
                        report.issues_by_type.get(issue, 0) + 1
                    )

        # Save results
        self._save_results()

        logger.info(
            f"Completed: {report.successful} successful, {report.failed} failed"
        )

        return report

    def _save_results(self):
        """Save processing results to disk."""

        # Save cleaned texts
        output_file = self.output_dir / "cleaned_books.json"

        data = {"total_books": len(self.processed_books), "books": []}

        for book_id, metadata in self.processed_books.items():
            data["books"].append(
                {
                    "book_id": metadata.book_id,
                    "title": metadata.title,
                    "category": metadata.category_name,
                    "authors": metadata.author_names,
                    "date": metadata.date,
                    "word_count": metadata.word_count,
                    "content_hash": metadata.content_hash,
                    "cleaned_text": metadata.cleaned_text[:50000],  # Truncate for JSON
                    "encoding_issues": metadata.encoding_issues,
                    "cleaning_applied": metadata.cleaning_applied,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved cleaned books to {output_file}")

    def get_chunks_for_indexing(self, chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Get text chunks ready for indexing.

        Each chunk includes:
        - Content
        - Metadata (book_id, title, category, author)
        - Position info
        """

        chunks = []

        for book_id, metadata in self.processed_books.items():
            text = metadata.cleaned_text

            # Split into chunks
            words = text.split()

            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i : i + chunk_size])

                chunks.append(
                    {
                        "content": chunk_text,
                        "book_id": metadata.book_id,
                        "title": metadata.title,
                        "category": metadata.category_name,
                        "authors": metadata.author_names,
                        "author_str": ", ".join(metadata.author_names),
                        "date": metadata.date,
                        "chunk_index": i // chunk_size,
                        "content_hash": compute_text_hash(chunk_text),
                    }
                )

        return chunks

    def save_chunks_for_indexing(self, filename: str = "indexing_chunks.json"):
        """Save chunks to file for indexing pipeline."""

        chunks = self.get_chunks_for_indexing()

        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_chunks": len(chunks), "chunks": chunks},
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

        return output_path


# ============================================================================
# Utility Functions
# ============================================================================


def quick_clean_text(text: str) -> str:
    """Quick text cleaning for immediate use."""
    cleaned, _ = clean_islamic_text(text, preserve_structure=False)
    return cleaned


def analyze_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Analyze the dataset to understand its structure and issues.

    Returns:
        Dictionary with analysis results
    """

    loader = DatasetLoader(dataset_path)
    cleaner = IslamicTextCleaner()

    analysis = {
        "total_books_in_metadata": len(loader.books_metadata),
        "total_book_files": 0,
        "categories": {},
        "date_range": {"min": 99999, "max": 0},
        "authors": set(),
        "sample_files": [],
        "encoding_issues": [],
    }

    # Get all files
    book_files = loader.get_all_book_files()
    analysis["total_book_files"] = len(book_files)

    # Analyze first few files
    for book_id, file_path in book_files[:10]:
        meta = loader.get_book_metadata(book_id)

        result = cleaner.process_file(file_path)

        if result:
            analysis["sample_files"].append(
                {
                    "book_id": book_id,
                    "title": meta.get("title", "") if meta else "",
                    "word_count": result.word_count,
                    "issues": result.encoding_issues,
                }
            )

            analysis["encoding_issues"].extend(result.encoding_issues)

    # Aggregate category info
    for book_id, meta in loader.books_metadata.items():
        cat_name = meta.get("cat_name", "unknown")
        analysis["categories"][cat_name] = analysis["categories"].get(cat_name, 0) + 1

        date = meta.get("date", 99999)
        if date != 99999:
            analysis["date_range"]["min"] = min(analysis["date_range"]["min"], date)
            analysis["date_range"]["max"] = max(analysis["date_range"]["max"], date)

        for author in meta.get("authors", []):
            analysis["authors"].add(author.get("name", ""))

    analysis["total_authors"] = len(analysis["authors"])
    analysis["authors"] = list(analysis["authors"])[:100]  # Limit for output

    return analysis


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <dataset_path> [output_dir]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "rag_system/data"

    print(f"Analyzing dataset at: {dataset_path}")

    # Analyze
    analysis = analyze_dataset(dataset_path)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

    # Process
    print(f"\nProcessing all books...")
    processor = CompleteDataProcessor(dataset_path, output_dir)
    report = processor.process_all_books()

    print(f"\nProcessing Report:")
    print(f"  Total files: {report.total_files}")
    print(f"  Successful: {report.successful}")
    print(f"  Failed: {report.failed}")
    print(f"  Encoding issues: {report.encoding_errors}")

    # Save chunks
    processor.save_chunks_for_indexing()
    print("\nChunks saved successfully!")
