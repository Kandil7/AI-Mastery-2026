"""
Specialized Chunking Strategies for Islamic Literature

Each category of Islamic literature has unique structural features:
- Quranic texts (Tafsir): Organized by verses and chapters
- Hadith collections: Organized by hadith units with isnad
- Fiqh books: Organized by topics and chapters
- Poetry: Organized by poems/bait units
- History/Biography: Organized by events/people

This module provides specialized chunking for each type.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BookCategory(Enum):
    """Types of Islamic literature based on structure."""

    # Core Religious
    TAFSIR = "tafsir"  # Quranic exegesis
    HADITH = "hadith"  # Hadith collections
    FIQH = "fiqh"  # Jurisprudence
    AQEEDAH = "aqeedah"  # Theology
    USUL = "usul"  # Principles of Fiqh

    # Language
    GRAMMAR = "grammar"  # النحو
    LEXICON = "lexicon"  # المعاجم
    LITERATURE = "literature"  # الأدب
    POETRY = "poetry"  # الشعر

    # History
    HISTORY = "history"  # التاريخ
    BIOGRAPHY = "biography"  # التراجم
    GEOGRAPHY = "geography"  # البلدان

    # Other
    SPIRITUALITY = "spirituality"  # الرقائق
    GENERAL = "general"  # General


@dataclass
class ChunkConfig:
    """Configuration for chunking a specific category."""

    category: BookCategory
    name_arabic: str

    # Chunking parameters
    chunk_size: int = 512  # target words
    overlap: int = 50  # overlap words

    # Special handling
    preserve_structural_markers: bool = False
    split_on_patterns: List[str] = field(default_factory=list)
    min_chunk_size: int = 50
    max_chunk_size: int = 2000

    # Post-processing
    add_metadata: bool = True
    extract_chapter_info: bool = False


# Category-specific configurations
CATEGORY_CONFIGS: Dict[BookCategory, ChunkConfig] = {
    BookCategory.TAFSIR: ChunkConfig(
        category=BookCategory.TAFSIR,
        name_arabic="التفسير",
        chunk_size=384,
        overlap=30,
        split_on_patterns=[
            r"\n\n+",  # Paragraphs
            r"(?:﴿|«)[^»]+(?:﴾|»)",  # Quranic verses
            r"تفسير\s+سورة",  # Surah starts
        ],
        preserve_structural_markers=True,
        extract_chapter_info=True,
    ),
    BookCategory.HADITH: ChunkConfig(
        category=BookCategory.HADITH,
        name_arabic="الحديث",
        chunk_size=256,
        overlap=20,
        split_on_patterns=[
            r"\n\n+",
            r"صحيح|ضعيف|حسن",  # Hadith grade
            r"حدثنا|أخبرنا|قال",  # Narrator phrases
            r"(?<=।)[A-Z]",  # Hadith separators if present
        ],
        preserve_structural_markers=True,
        min_chunk_size=30,
        max_chunk_size=500,
    ),
    BookCategory.FIQH: ChunkConfig(
        category=BookCategory.FIQH,
        name_arabic="الفقه",
        chunk_size=512,
        overlap=40,
        split_on_patterns=[
            r"\n\n+",
            r"CHAPTER|الباب|الفصل",  # Chapter markers
            r"مسألة|فصل",  # Issue markers
            r"(?:۞|⚫)",  # Section markers
        ],
        extract_chapter_info=True,
    ),
    BookCategory.AQEEDAH: ChunkConfig(
        category=BookCategory.AQEEDAH,
        name_arabic="العقيدة",
        chunk_size=512,
        overlap=50,
        split_on_patterns=[
            r"\n\n+",
            r"المبحث|الفصل|الباب",
            r"مسألة",
        ],
        extract_chapter_info=True,
    ),
    BookCategory.USUL: ChunkConfig(
        category=BookCategory.USUL,
        name_arabic="أصول الفقه",
        chunk_size=512,
        overlap=40,
        split_on_patterns=[
            r"\n\n+",
            r"المبحث|الفصل|الباب",
        ],
    ),
    BookCategory.GRAMMAR: ChunkConfig(
        category=BookCategory.GRAMMAR,
        name_arabic="النحو",
        chunk_size=384,
        overlap=30,
        split_on_patterns=[
            r"\n\n+",
            r"المبحث|الفصل|القاعدة",
            r"مثال|باب",
        ],
        extract_chapter_info=True,
    ),
    BookCategory.LEXICON: ChunkConfig(
        category=BookCategory.LEXICON,
        name_arabic="المعاجم",
        chunk_size=192,
        overlap=10,
        split_on_patterns=[
            r"\n\n+",
            r"^[\u0621-\u064a]+:",  # Word entries
        ],
        min_chunk_size=20,
        max_chunk_size=400,
    ),
    BookCategory.LITERATURE: ChunkConfig(
        category=BookCategory.LITERATURE,
        name_arabic="الأدب",
        chunk_size=512,
        overlap=40,
        split_on_patterns=[
            r"\n\n+",
            r"المقالة|الفصل",
        ],
    ),
    BookCategory.POETRY: ChunkConfig(
        category=BookCategory.POETRY,
        name_arabic="الشعر",
        chunk_size=256,
        overlap=20,
        split_on_patterns=[
            r"\n\n+",
            r"—+",  # Poetic line separator
            r"\n(?=[\u0621-\u064a])",  # Arabic line start
        ],
        min_chunk_size=30,
        max_chunk_size=400,
    ),
    BookCategory.HISTORY: ChunkConfig(
        category=BookCategory.HISTORY,
        name_arabic="التاريخ",
        chunk_size=768,
        overlap=50,
        split_on_patterns=[
            r"\n\n+",
            r"في\s+سنة|عام",  # Date references
            r"الباب|الفصل",
        ],
        extract_chapter_info=True,
    ),
    BookCategory.BIOGRAPHY: ChunkConfig(
        category=BookCategory.BIOGRAPHY,
        name_arabic="التراجم",
        chunk_size=512,
        overlap=40,
        split_on_patterns=[
            r"\n\n+",
            r"ترجمة|سير",
            r"تابع",
        ],
    ),
    BookCategory.GEOGRAPHY: ChunkConfig(
        category=BookCategory.GEOGRAPHY,
        name_arabic="البلدان",
        chunk_size=512,
        overlap=40,
        split_on_patterns=[
            r"\n\n+",
            r"في\s+البلدان|مدن",
        ],
    ),
    BookCategory.SPIRITUALITY: ChunkConfig(
        category=BookCategory.SPIRITUALITY,
        name_arabic="الرقائق",
        chunk_size=512,
        overlap=40,
        split_on_patterns=[
            r"\n\n+",
            r"المبحث|الفصل",
            r"فائدة",
        ],
    ),
    BookCategory.GENERAL: ChunkConfig(
        category=BookCategory.GENERAL,
        name_arabic="عام",
        chunk_size=512,
        overlap=40,
        split_on_patterns=[
            r"\n\n+",
        ],
    ),
}


# Category detection keywords
CATEGORY_KEYWORDS = {
    BookCategory.TAFSIR: [
        "تفسير",
        "القرآن",
        "آية",
        "سورة",
        "ناسخ",
        "منسوخ",
        "سبب نزول",
        "مكي",
        "مدني",
    ],
    BookCategory.HADITH: [
        "حديث",
        "صحيح",
        "سنن",
        "مسند",
        "إسناد",
        "راوي",
        "تخريج",
        "علة",
        "ضعيف",
    ],
    BookCategory.FIQH: [
        "فقه",
        "حكم",
        "يجوز",
        "لا يجوز",
        "مسألة",
        "باب",
        "فصل",
        "راجح",
        "مرجوح",
    ],
    BookCategory.AQEEDAH: [
        "عقيدة",
        "توحيد",
        "صفات",
        "قدر",
        "إيمان",
        "شرك",
        "كفر",
        "سلف",
        "أهل سنة",
    ],
    BookCategory.USUL: ["أصول", "دليل", "حجة", "قياس", "استحسان", "اجتهاد"],
    BookCategory.GRAMMAR: ["نحو", "إعراب", "بناء", "مرفوع", "منصوب", "مجرور"],
    BookCategory.LEXICON: ["معجم", "غريب", "لغة", "معنى", "مادة"],
    BookCategory.LITERATURE: ["أدب", "مقالة", "خطبة", "فصاحة", "بلاغة"],
    BookCategory.POETRY: ["شعر", "بيت", "قصيدة", "بحر", "روي", "قافية"],
    BookCategory.HISTORY: ["تاريخ", "حكم", "دولة", "خلافة", "فتح", "معركة"],
    BookCategory.BIOGRAPHY: ["ترجمة", "سير", "طبقات", "وفيات", "تاريخ"],
    BookCategory.GEOGRAPHY: ["بلدان", "مدن", "جغرافيا", "رحلة", "وصف"],
    BookCategory.SPIRITUALITY: ["رقائق", "أدب", "تربية", "تهذيب", "مجاهدة"],
}


def detect_category(title: str, category_name: str = "") -> BookCategory:
    """
    Detect the category of a book based on title and metadata.

    Args:
        title: Book title
        category_name: Category from metadata

    Returns:
        Detected BookCategory
    """

    # First check metadata category
    category_map = {
        "التفسير": BookCategory.TAFSIR,
        "علوم القرآن": BookCategory.TAFSIR,
        "التجويد": BookCategory.TAFSIR,
        "كتب السنة": BookCategory.HADITH,
        "شروح الحديث": BookCategory.HADITH,
        "علوم الحديث": BookCategory.HADITH,
        "الفقه الحنفي": BookCategory.FIQH,
        "الفقه المالكي": BookCategory.FIQH,
        "الفقه الشافعي": BookCategory.FIQH,
        "الفقه الحنبلي": BookCategory.FIQH,
        "الفقه العام": BookCategory.FIQH,
        "مسائل فقهية": BookCategory.FIQH,
        "أصول الفقه": BookCategory.USUL,
        "العقيدة": BookCategory.AQEEDAH,
        "الفرق والردود": BookCategory.AQEEDAH,
        "النحو والصرف": BookCategory.GRAMMAR,
        "الغريب والمعاجم": BookCategory.LEXICON,
        "اللغة العربية": BookCategory.LEXICON,
        "الأدب": BookCategory.LITERATURE,
        "الدواوين": BookCategory.POETRY,
        "العروض": BookCategory.POETRY,
        "التاريخ": BookCategory.HISTORY,
        "التراجم": BookCategory.BIOGRAPHY,
        "البلدان": BookCategory.GEOGRAPHY,
        "الرقائق": BookCategory.SPIRITUALITY,
    }

    if category_name in category_map:
        return category_map[category_name]

    # Fall back to keyword detection
    title_lower = title.lower()

    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in title_lower)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]

    return BookCategory.GENERAL


class IslamicTextChunker:
    """
    Specialized chunker for Islamic literature.

    Handles different category types with appropriate chunking strategies.
    """

    def __init__(self):
        self.category_detect_fn = detect_category

    def chunk_text(
        self,
        text: str,
        category: BookCategory,
        custom_config: Optional[ChunkConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk text according to category-specific rules.

        Args:
            text: Text to chunk
            category: Book category
            custom_config: Optional custom configuration

        Returns:
            List of chunk dictionaries
        """

        config = custom_config or CATEGORY_CONFIGS.get(
            category, CATEGORY_CONFIGS[BookCategory.GENERAL]
        )

        # First split by structural patterns
        segments = self._split_by_patterns(text, config)

        # Then chunk each segment
        chunks = []
        for seg_idx, segment in enumerate(segments):
            segment_chunks = self._create_chunks(segment, config, segment_index=seg_idx)
            chunks.extend(segment_chunks)

        return chunks

    def _split_by_patterns(self, text: str, config: ChunkConfig) -> List[str]:
        """Split text by category-specific patterns."""

        if not config.split_on_patterns:
            return [text]

        # Start with simple paragraph split
        segments = [text]

        for pattern in config.split_on_patterns:
            new_segments = []
            for segment in segments:
                # Split but keep delimiters
                parts = re.split(f"({pattern})", segment)

                # Combine parts with their delimiters
                current = ""
                for part in parts:
                    if re.match(pattern, part):
                        # Add previous and the delimiter as a segment
                        if current.strip():
                            new_segments.append(current)
                        new_segments.append(part)
                        current = ""
                    else:
                        current += part

                if current.strip():
                    new_segments.append(current)

            segments = new_segments

        # Merge small segments
        merged = self._merge_small_segments(segments, config.min_chunk_size)

        return merged

    def _merge_small_segments(self, segments: List[str], min_size: int) -> List[str]:
        """Merge segments that are too small."""

        if not segments:
            return segments

        merged = []
        current = ""

        for seg in segments:
            current_len = len(current.split())
            seg_len = len(seg.split())

            if current_len < min_size and seg_len < min_size:
                # Merge both
                current = current + "\n\n" + seg
            elif current_len < min_size:
                # Add to next
                current = current + "\n\n" + seg
            else:
                if current:
                    merged.append(current)
                current = seg

        if current:
            merged.append(current)

        return merged

    def _create_chunks(
        self, text: str, config: ChunkConfig, segment_index: int = 0
    ) -> List[Dict[str, Any]]:
        """Create chunks from text segment."""

        words = text.split()

        if not words:
            return []

        chunks = []

        # Calculate number of chunks
        num_chunks = (len(words) - 1) // (config.chunk_size - config.overlap) + 1

        for i in range(num_chunks):
            start = i * (config.chunk_size - config.overlap)
            end = min(start + config.chunk_size, len(words))

            chunk_text = " ".join(words[start:end])

            # Skip if too small
            if len(chunk_text.split()) < config.min_chunk_size:
                continue

            chunks.append(
                {
                    "content": chunk_text,
                    "chunk_index": i,
                    "segment_index": segment_index,
                    "word_count": len(chunk_text.split()),
                }
            )

        return chunks

    def chunk_book(
        self,
        text: str,
        title: str,
        category_name: str,
        book_id: int,
        author: str = "",
        date: int = 99999,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a complete book with full metadata.

        Args:
            text: Book text
            title: Book title
            category_name: Category from metadata
            book_id: Book ID
            author: Author name(s)
            date: Composition date

        Returns:
            List of enriched chunks
        """

        # Detect category
        category = self.category_detect_fn(title, category_name)

        # Get chunks
        chunks = self.chunk_text(text, category)

        # Enrich with metadata
        for chunk in chunks:
            chunk.update(
                {
                    "book_id": book_id,
                    "title": title,
                    "category": category_name,
                    "category_type": category.value,
                    "author": author,
                    "date": date,
                }
            )

        return chunks


class SemanticIslamicChunker(IslamicTextChunker):
    """
    Enhanced chunker that also tries to maintain semantic boundaries.

    Uses additional heuristics to avoid breaking:
    - Quranic verses
    - Hadith statements
    - Poetry couplets
    - Lists and enumerations
    """

    def __init__(self):
        super().__init__()

        # Patterns that indicate semantic boundaries
        self.boundary_patterns = [
            # Quranic verses
            r"﴿[^﴾]+﴾",
            # Hadith grade indicators
            r"(?:صحيح|حسن|ضعيف|موضوع)\s+(?:بن|عن)",
            # Poetry line
            r"^—+$",
            # Chapter headings
            r"^(?:الباب|الفصل|المبحث|التقسيم)\s+\d+",
            # Numbered items
            r"^\d+\.",
        ]

    def _identify_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """
        Identify semantic boundaries in text.

        Returns:
            List of (position, boundary_type) tuples
        """

        boundaries = []

        for pattern in self.boundary_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                boundaries.append((match.start(), pattern))

        return sorted(boundaries, key=lambda x: x[0])

    def chunk_text(
        self,
        text: str,
        category: BookCategory,
        custom_config: Optional[ChunkConfig] = None,
    ) -> List[Dict[str, Any]]:
        """Enhanced chunking with semantic awareness."""

        # Get base config
        config = custom_config or CATEGORY_CONFIGS.get(
            category, CATEGORY_CONFIGS[BookCategory.GENERAL]
        )

        # For certain categories, use semantic-aware chunking
        if category in [BookCategory.TAFSIR, BookCategory.HADITH, BookCategory.POETRY]:
            return self._semantic_chunk(text, config)

        # Otherwise use standard chunking
        return super().chunk_text(text, category, custom_config)

    def _semantic_chunk(self, text: str, config: ChunkConfig) -> List[Dict[str, Any]]:
        """Chunk while preserving semantic units."""

        # Find all semantic boundaries
        boundaries = self._identify_boundaries(text)

        if not boundaries:
            # Fall back to standard
            return self._create_chunks(text, config)

        # Split at boundaries
        segments = []
        last_pos = 0

        for pos, pattern in boundaries:
            if pos > last_pos:
                segments.append(text[last_pos:pos])
            last_pos = pos

        if last_pos < len(text):
            segments.append(text[last_pos:])

        # Chunk each segment
        all_chunks = []
        for seg_idx, segment in enumerate(segments):
            if not segment.strip():
                continue

            seg_chunks = self._create_chunks(segment, config, segment_index=seg_idx)
            all_chunks.extend(seg_chunks)

        return all_chunks


# ============================================================================
# Factory Functions
# ============================================================================


def create_chunker(
    semantic: bool = False, category: Optional[BookCategory] = None
) -> IslamicTextChunker:
    """
    Create a chunker instance.

    Args:
        semantic: Use semantic-aware chunker
        category: Specific category to chunk for

    Returns:
        Chunker instance
    """

    if semantic:
        return SemanticIslamicChunker()

    return IslamicTextChunker()


def get_category_config(category: BookCategory) -> ChunkConfig:
    """Get the chunking configuration for a category."""
    return CATEGORY_CONFIGS.get(category, CATEGORY_CONFIGS[BookCategory.GENERAL])


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import json

    # Test with sample texts
    test_texts = [
        ("تفسير سورة البقرة", BookCategory.TAFSIR),
        ("صحيح البخاري", BookCategory.HADITH),
        ("الفرائض", BookCategory.FIQH),
    ]

    chunker = IslamicTextChunker()

    for title, category in test_texts:
        print(f"\n--- Testing {category.value} ---")
        config = get_category_config(category)
        print(f"Chunk size: {config.chunk_size}")
        print(f"Overlap: {config.overlap}")
        print(f"Split patterns: {config.split_on_patterns[:2]}")
