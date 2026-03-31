"""
Book Processor for Arabic LLM Dataset

This module processes extracted books from the Shamela dataset and converts
them into training examples using instruction templates.

Features:
- Load book metadata from JSON/SQLite
- Process extracted text files
- Extract segments suitable for instruction tuning
- Apply templates based on book category and content
- Generate structured training examples
"""

import os
import json
import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass
import re
from tqdm import tqdm

from .schema import (
    TrainingExample, Role, Skill, Level, Domain, Style, TaskType,
    DatasetConfig, DatasetStatistics, validate_example
)
from .templates import (
    get_templates, get_random_template, POETRY_METERS, POETRY_TOPICS
)


@dataclass
class Book:
    """Represents a book from the Shamela dataset"""
    id: int
    guid: str
    title: str
    category: str
    author_id: int
    author_name: str
    author_death: Optional[int]
    file_path: str
    size_mb: float
    content: Optional[str] = None


@dataclass
class TextSegment:
    """A segment of text suitable for generating training examples"""
    text: str
    segment_type: str  # verse, prose, hadith, poetry, heading
    book_id: int
    book_title: str
    author_name: str
    category: str
    start_pos: int
    end_pos: int


class BookProcessor:
    """
    Process Shamela books for Arabic LLM training data generation.
    
    This class handles:
    - Loading book metadata
    - Reading extracted text files
    - Segmenting text into training-ready chunks
    - Categorizing content by type
    """
    
    def __init__(
        self,
        books_dir: str,
        metadata_dir: str,
        output_dir: str,
    ):
        """
        Initialize the book processor.
        
        Args:
            books_dir: Path to extracted books directory
            metadata_dir: Path to metadata directory
            output_dir: Path to output processed data
        """
        self.books_dir = Path(books_dir)
        self.metadata_dir = Path(metadata_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.books_db_path = self.metadata_dir / "books.db"
        self.books_json_path = self.metadata_dir / "books.json"
        self.categories_path = self.metadata_dir / "categories.json"
        self.authors_path = self.metadata_dir / "authors.json"
        
        # Category mappings for training
        self.linguistic_categories = {
            "كتب اللغة": {"domain": Domain.LITERATURE, "style": Style.FUSHА_CLASSICAL},
            "التفسير": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "كتب السنة": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "الأدب": {"domain": Domain.LITERATURE, "style": Style.FUSHА_CLASSICAL},
            "الشعر": {"domain": Domain.LITERATURE, "style": Style.FUSHА_CLASSICAL},
            "البلاغة": {"domain": Domain.EDUCATION, "style": Style.FUSHА_CLASSICAL},
            "النحو": {"domain": Domain.EDUCATION, "style": Style.FUSHА_CLASSICAL},
            "التجويد والقراءات": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "العقيدة": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "الفقه العام": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "الفرق والردود": {"domain": Domain.ISLAMIC_STUDIES, "style": Style.FUSHА_CLASSICAL},
            "التراجم والطبقات": {"domain": Domain.HERITAGE, "style": Style.FUSHА_CLASSICAL},
        }
        
        # Cache for loaded books
        self._books_cache: Dict[int, Book] = {}
        self._metadata_loaded = False
    
    def load_metadata(self) -> int:
        """
        Load book metadata from SQLite or JSON.
        
        Returns:
            Number of books loaded
        """
        if self._metadata_loaded:
            return len(self._books_cache)
        
        books_loaded = 0
        
        # Try SQLite first
        if self.books_db_path.exists():
            books_loaded = self._load_from_sqlite()
        elif self.books_json_path.exists():
            books_loaded = self._load_from_json()
        else:
            raise FileNotFoundError(
                f"No metadata found. Expected at: {self.books_db_path} or {self.books_json_path}"
            )
        
        self._metadata_loaded = True
        return books_loaded
    
    def _load_from_sqlite(self) -> int:
        """Load metadata from SQLite database"""
        conn = sqlite3.connect(self.books_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Load only extracted books
        cursor.execute("""
            SELECT book_id, guid, title, cat_name, author_str, file, size_mb
            FROM books
            WHERE extracted = 1
        """)
        
        # Load authors
        cursor.execute("SELECT author_id, name, death FROM authors")
        authors = {row["author_id"]: dict(row) for row in cursor.fetchall()}
        
        books_count = 0
        for row in cursor.fetchall():
            author_id = int(row["author_str"]) if row["author_str"].isdigit() else None
            author_info = authors.get(author_id, {})
            
            book = Book(
                id=row["book_id"],
                guid=row["guid"],
                title=row["title"],
                category=row["cat_name"],
                author_id=author_id,
                author_name=author_info.get("name", "Unknown"),
                author_death=author_info.get("death"),
                file_path=str(self.books_dir / row["file"]),
                size_mb=row["size_mb"],
            )
            self._books_cache[book.id] = book
            books_count += 1
        
        conn.close()
        return books_count
    
    def _load_from_json(self) -> int:
        """Load metadata from JSON file"""
        with open(self.books_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load authors
        authors = {}
        if self.authors_path.exists():
            with open(self.authors_path, 'r', encoding='utf-8') as f:
                authors_data = json.load(f)
                authors = {a["id"]: a for a in authors_data.get("authors", [])}
        
        books_count = 0
        for book_data in data.get("books", []):
            if not book_data.get("extracted", False):
                continue
            
            author_id = int(book_data["author_str"]) if book_data["author_str"].isdigit() else None
            author_info = authors.get(author_id, {})
            
            book = Book(
                id=book_data["id"],
                guid=book_data["guid"],
                title=book_data["title"],
                category=book_data["cat_name"],
                author_id=author_id,
                author_name=author_info.get("name", "Unknown"),
                author_death=author_info.get("death"),
                file_path=str(self.books_dir / book_data["file"]),
                size_mb=book_data["size_mb"],
            )
            self._books_cache[book.id] = book
            books_count += 1
        
        return books_count
    
    def load_book_content(self, book_id: int) -> Optional[str]:
        """
        Load content of a specific book.
        
        Args:
            book_id: ID of the book to load
            
        Returns:
            Book content as string, or None if not found
        """
        if book_id not in self._books_cache:
            return None
        
        book = self._books_cache[book_id]
        
        if book.content is not None:
            return book.content
        
        # Try to read the file
        if os.path.exists(book.file_path):
            with open(book.file_path, 'r', encoding='utf-8') as f:
                book.content = f.read()
        
        return book.content
    
    def segment_text(
        self,
        text: str,
        book: Book,
        min_length: int = 50,
        max_length: int = 500,
    ) -> List[TextSegment]:
        """
        Segment text into training-ready chunks.
        
        Args:
            text: Full book text
            book: Book metadata
            min_length: Minimum segment length
            max_length: Maximum segment length
            
        Returns:
            List of TextSegment objects
        """
        segments = []
        
        # Detect content type
        content_type = self._detect_content_type(text, book.category)
        
        if content_type == "poetry":
            segments = self._segment_poetry(text, book, min_length, max_length)
        elif content_type == "hadith":
            segments = self._segment_hadith(text, book, min_length, max_length)
        elif content_type == "verse":
            segments = self._segment_quranic_verses(text, book, min_length, max_length)
        else:
            segments = self._segment_prose(text, book, min_length, max_length)
        
        return segments
    
    def _detect_content_type(self, text: str, category: str) -> str:
        """Detect the type of content in text"""
        if category == "الشعر":
            return "poetry"
        if category == "كتب السنة":
            return "hadith"
        if category == "التفسير":
            return "verse"
        
        # Check for poetry markers
        poetry_indicators = [
            r"^\s*[أوقدبلهمن]\s+\w+.*\n\s*\w+.*$",  # Two-line structure
            r"بحر\s+\w+",  # Meter mention
            r"قافية",  # Rhyme mention
        ]
        for pattern in poetry_indicators:
            if re.search(pattern, text[:1000], re.MULTILINE):
                return "poetry"
        
        return "prose"
    
    def _segment_poetry(
        self,
        text: str,
        book: Book,
        min_length: int,
        max_length: int,
    ) -> List[TextSegment]:
        """Segment poetry text into verses"""
        segments = []
        lines = text.split('\n')
        
        current_verse = []
        current_text = ""
        start_pos = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this looks like a verse line
            if len(line) < 100:  # Verse lines are typically shorter
                current_verse.append(line)
                if len(current_verse) >= 2:  # Complete bayt
                    verse_text = " ".join(current_verse)
                    if min_length <= len(verse_text) <= max_length:
                        segments.append(TextSegment(
                            text=verse_text,
                            segment_type="poetry",
                            book_id=book.id,
                            book_title=book.title,
                            author_name=book.author_name,
                            category=book.category,
                            start_pos=start_pos,
                            end_pos=start_pos + len(verse_text),
                        ))
                    current_verse = []
                    start_pos = i
            else:
                # Prose commentary on poetry
                if len(line) >= min_length:
                    segments.append(TextSegment(
                        text=line[:max_length],
                        segment_type="prose",
                        book_id=book.id,
                        book_title=book.title,
                        author_name=book.author_name,
                        category=book.category,
                        start_pos=start_pos,
                        end_pos=start_pos + len(line),
                    ))
        
        return segments
    
    def _segment_hadith(
        self,
        text: str,
        book: Book,
        min_length: int,
        max_length: int,
    ) -> List[TextSegment]:
        """Segment hadith text"""
        segments = []
        
        # Common hadith markers
        hadith_patterns = [
            r"(حدثنا|أخبرنا|عن|قال رسول الله|قال النبي)",
            r"ﷺ",
            r"صلى الله عليه وسلم",
        ]
        
        # Split by hadith markers
        parts = re.split(r'(?=(?:' + '|'.join(hadith_patterns) + r'))', text)
        
        for part in parts:
            part = part.strip()
            if len(part) >= min_length:
                # Truncate if too long
                if len(part) > max_length * 2:
                    part = part[:max_length * 2]
                
                segments.append(TextSegment(
                    text=part,
                    segment_type="hadith",
                    book_id=book.id,
                    book_title=book.title,
                    author_name=book.author_name,
                    category=book.category,
                    start_pos=text.find(part),
                    end_pos=text.find(part) + len(part),
                ))
        
        return segments
    
    def _segment_quranic_verses(
        self,
        text: str,
        book: Book,
        min_length: int,
        max_length: int,
    ) -> List[TextSegment]:
        """Segment Quranic verses from tafsir"""
        segments = []
        
        # Quranic verse pattern
        verse_pattern = r'﴿([^﴾]+)﴾'
        
        for match in re.finditer(verse_pattern, text):
            verse = match.group(1).strip()
            if len(verse) >= 20:  # Minimum verse length
                segments.append(TextSegment(
                    text=verse,
                    segment_type="verse",
                    book_id=book.id,
                    book_title=book.title,
                    author_name=book.author_name,
                    category=book.category,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
        
        return segments
    
    def _segment_prose(
        self,
        text: str,
        book: Book,
        min_length: int,
        max_length: int,
    ) -> List[TextSegment]:
        """Segment prose text into sentences/paragraphs"""
        segments = []
        
        # Split by sentence endings
        sentences = re.split(r'[.!?۔]\s+', text)
        
        current_segment = []
        current_length = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            
            if current_length + sentence_len <= max_length:
                current_segment.append(sentence)
                current_length += sentence_len
            else:
                if current_length >= min_length:
                    segment_text = ' '.join(current_segment)
                    segments.append(TextSegment(
                        text=segment_text,
                        segment_type="prose",
                        book_id=book.id,
                        book_title=book.title,
                        author_name=book.author_name,
                        category=book.category,
                        start_pos=start_pos,
                        end_pos=start_pos + len(segment_text),
                    ))
                
                current_segment = [sentence]
                current_length = sentence_len
                start_pos = text.find(sentence)
        
        # Add last segment
        if current_length >= min_length:
            segment_text = ' '.join(current_segment)
            segments.append(TextSegment(
                text=segment_text,
                segment_type="prose",
                book_id=book.id,
                book_title=book.title,
                author_name=book.author_name,
                category=book.category,
                start_pos=start_pos,
                end_pos=start_pos + len(segment_text),
            ))
        
        return segments
    
    def get_books_by_category(self, categories: List[str]) -> List[Book]:
        """Get all books from specified categories"""
        return [
            book for book in self._books_cache.values()
            if book.category in categories
        ]
    
    def process_books(
        self,
        categories: Optional[List[str]] = None,
        max_books: Optional[int] = None,
        min_segments_per_book: int = 10,
        max_segments_per_book: int = 100,
    ) -> Generator[TextSegment, None, None]:
        """
        Process books and yield segments.
        
        Args:
            categories: Filter by categories (None = all)
            max_books: Maximum number of books to process
            min_segments_per_book: Minimum segments to extract per book
            max_segments_per_book: Maximum segments to extract per book
            
        Yields:
            TextSegment objects
        """
        if not self._metadata_loaded:
            self.load_metadata()
        
        # Filter books
        books = list(self._books_cache.values())
        
        if categories:
            books = [b for b in books if b.category in categories]
        
        # Shuffle for randomness
        random.shuffle(books)
        
        if max_books:
            books = books[:max_books]
        
        # Process each book
        for book in tqdm(books, desc="Processing books"):
            content = self.load_book_content(book.id)
            if not content:
                continue
            
            segments = self.segment_text(content, book)
            
            # Sample segments if too many
            if len(segments) > max_segments_per_book:
                segments = random.sample(segments, max_segments_per_book)
            
            # Yield segments
            for segment in segments:
                yield segment
    
    def save_processed_data(
        self,
        segments: List[TextSegment],
        output_file: str,
    ) -> int:
        """
        Save processed segments to JSON file.
        
        Args:
            segments: List of segments to save
            output_file: Output file path
            
        Returns:
            Number of segments saved
        """
        output_path = self.output_dir / output_file
        
        data = {
            "total_segments": len(segments),
            "generated_at": __import__('datetime').datetime.now().isoformat(),
            "segments": [
                {
                    "text": s.text,
                    "segment_type": s.segment_type,
                    "book_id": s.book_id,
                    "book_title": s.book_title,
                    "author_name": s.author_name,
                    "category": s.category,
                }
                for s in segments
            ],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return len(segments)


def process_all_books(
    books_dir: str,
    metadata_dir: str,
    output_dir: str,
    config: DatasetConfig,
) -> DatasetStatistics:
    """
    Process all books and generate statistics.
    
    Args:
        books_dir: Path to extracted books
        metadata_dir: Path to metadata
        output_dir: Path to save processed data
        config: Dataset configuration
        
    Returns:
        DatasetStatistics object
    """
    processor = BookProcessor(books_dir, metadata_dir, output_dir)
    
    # Load metadata
    num_books = processor.load_metadata()
    print(f"Loaded metadata for {num_books} books")
    
    # Process books
    all_segments = []
    stats = {
        "by_category": {},
        "by_type": {},
    }
    
    for segment in processor.process_books(
        categories=config.source_categories,
        max_books=1000,  # Limit for initial processing
    ):
        all_segments.append(segment)
        
        # Update stats
        cat = segment.category
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        
        seg_type = segment.segment_type
        stats["by_type"][seg_type] = stats["by_type"].get(seg_type, 0) + 1
    
    # Save processed data
    processor.save_processed_data(all_segments, "processed_segments.json")
    
    print(f"\nProcessed {len(all_segments)} segments")
    print(f"Categories: {stats['by_category']}")
    print(f"Types: {stats['by_type']}")
    
    return DatasetStatistics(
        total_examples=len(all_segments),
        source_books=len(set(s.book_id for s in all_segments)),
    )


if __name__ == "__main__":
    # Test the processor
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python book_processor.py <books_dir> <metadata_dir> [output_dir]")
        sys.exit(1)
    
    books_dir = sys.argv[1]
    metadata_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/raw"
    
    config = DatasetConfig()
    stats = process_all_books(books_dir, metadata_dir, output_dir, config)
    
    print(f"\nStatistics: {stats.to_dict()}")
