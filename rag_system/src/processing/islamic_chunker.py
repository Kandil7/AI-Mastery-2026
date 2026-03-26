"""
Advanced Islamic Literature Chunking System

Specialized chunking strategies optimized for Islamic literature:
1. Semantic Chapter-Based Chunking
2. Hadith-Aware Chunking (preserve chains)
3. Verse-Aware Chunking (for Tafsir)
4. Fiqh Ruling Chunking
5. Context-Preserving Chunking
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    chunk_id: str
    content: str
    book_id: int
    book_title: str
    author: str
    category: str
    chunk_type: str  # 'chapter', 'verse', 'hadith', 'ruling', 'general'
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


# ============================================================================
# Chunking Strategies
# ============================================================================


class IslamicLiteratureChunker:
    """
    Specialized chunker for Islamic literature.

    Preserves:
    - Quranic verses (for Tafsir)
    - Hadith chains (for Hadith)
    - Fiqh rulings (for Fiqh)
    - Chapter structure
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_book(
        self,
        content: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
    ) -> List[Chunk]:
        """
        Chunk a complete book using the best strategy.

        Args:
            content: Book content
            book_id: Book ID
            book_title: Book title
            author: Author name
            category: Book category

        Returns:
            List of chunks
        """

        # Determine chunking strategy based on category
        if self._is_tafsir(category):
            return self._chunk_by_verses(content, book_id, book_title, author, category)
        elif self._is_hadith(category):
            return self._chunk_by_hadith(content, book_id, book_title, author, category)
        elif self._is_fiqh(category):
            return self._chunk_by_rulings(
                content, book_id, book_title, author, category
            )
        else:
            return self._chunk_by_semantic(
                content, book_id, book_title, author, category
            )

    def _is_tafsir(self, category: str) -> bool:
        """Check if category is Tafsir (Quranic exegesis)."""
        tafsir_categories = [
            "التفسير",
            "علوم القرآن وأصول التفسير",
            "التجويد والقراءات",
        ]
        return category in tafsir_categories

    def _is_hadith(self, category: str) -> bool:
        """Check if category is Hadith."""
        hadith_categories = [
            "كتب السنة",
            "شروح الحديث",
            "التخريج والأطراف",
            "العلل والسؤلات الحديثية",
            "علوم الحديث",
        ]
        return category in hadith_categories

    def _is_fiqh(self, category: str) -> bool:
        """Check if category is Fiqh (jurisprudence)."""
        fiqh_categories = [
            "الفقه وأصوله",
            "مسائل فقهية",
            "الفقه الحنفي",
            "الفقه المالكي",
            "الفقه الشافعي",
            "الفقه الحنبلي",
            "الفقه العام",
            "أصول الفقه",
            "علوم الفقه والقواعد الفقهية",
            "الفتاوى",
        ]
        return category in fiqh_categories

    # =========================================================================
    # Tafsir Chunking (by verses)
    # =========================================================================

    def _chunk_by_verses(
        self,
        content: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
    ) -> List[Chunk]:
        """Chunk by Quranic verses - preserves verse context."""

        chunks = []

        # Find all Quranic verses in content
        # Pattern: verses with markers like ﴿...﴾ or specific verse patterns
        verse_pattern = r"[﴿﴾](.*?)[﴿﴾]"
        verses = list(re.finditer(verse_pattern, content))

        if verses:
            # Chunk around verses
            current_pos = 0
            chunk_idx = 0

            for verse_match in verses:
                # Get surrounding context
                start = max(0, verse_match.start() - 200)
                end = min(len(content), verse_match.end() + 200)

                # Extract verse content
                verse_text = verse_match.group(0)

                # Create chunk with verse
                chunk_content = content[start:end]

                chunks.append(
                    Chunk(
                        chunk_id=f"{book_id}_verse_{chunk_idx}",
                        content=chunk_content,
                        book_id=book_id,
                        book_title=book_title,
                        author=author,
                        category=category,
                        chunk_type="verse",
                        start_char=start,
                        end_char=end,
                        metadata={
                            "verse_text": verse_text,
                            "has_quran_verse": True,
                        },
                    )
                )

                chunk_idx += 1
                current_pos = verse_match.end()
        else:
            # Fallback to semantic chunking
            return self._chunk_by_semantic(
                content, book_id, book_title, author, category
            )

        return chunks

    # =========================================================================
    # Hadith Chunking (preserve chains)
    # =========================================================================

    def _chunk_by_hadith(
        self,
        content: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
    ) -> List[Chunk]:
        """Chunk by hadith - preserves narrator chains."""

        chunks = []

        # Find hadith markers
        # Common patterns:
        # - "حدثنا", "أخبرنا", "قال"
        # - Hadith numbering patterns
        hadith_patterns = [
            r"(?:قال|حدثنا|أخبرنا|روى|ثبت)\s+[^\.]+",  # Chains
            r"\d+_\w+",  # Hadith numbers
            r"حديث[:\s]+",  # "Hadith:" markers
        ]

        # Split by common hadith separators
        # Pattern: "1." or "1 -" at start of lines
        hadith_splits = re.split(r"\n(?=\d+[\.\-]\s*)", content)

        if len(hadith_splits) > 1:
            # Process each hadith as a chunk
            for i, hadith_text in enumerate(hadith_splits):
                if len(hadith_text.strip()) < 50:
                    continue

                # Extract hadith number
                num_match = re.match(r"^(\d+)", hadith_text.strip())
                hadith_num = num_match.group(1) if num_match else str(i)

                # Find narrator chain
                chain_match = re.search(r"(?:حدثنا|أخبرنا|روى)\s+([^\n]+)", hadith_text)
                chain = chain_match.group(1) if chain_match else None

                chunks.append(
                    Chunk(
                        chunk_id=f"{book_id}_hadith_{hadith_num}",
                        content=hadith_text.strip()[:1000],  # Limit length
                        book_id=book_id,
                        book_title=book_title,
                        author=author,
                        category=category,
                        chunk_type="hadith",
                        start_char=0,
                        end_char=len(hadith_text),
                        metadata={
                            "hadith_number": hadith_num,
                            "narrator_chain": chain,
                        },
                    )
                )
        else:
            # Fallback
            return self._chunk_by_semantic(
                content, book_id, book_title, author, category
            )

        return chunks

    # =========================================================================
    # Fiqh Chunking (by rulings)
    # =========================================================================

    def _chunk_by_rulings(
        self,
        content: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
    ) -> List[Chunk]:
        """Chunk by fiqh rulings - preserves rule structure."""

        chunks = []

        # Fiqh ruling markers
        ruling_patterns = [
            r"(?:الأحكام?|حكم)\s*:",  # "Rule:"
            r"\d+\s*-\s*(?:يجب|يحرم|يجوز|يسن)",  # Numbered rulings
            r"(?:فصل|باب|مبحث)\s*\d+",  # Chapter markers
        ]

        # Split by rulings
        # Use numbered sections
        sections = re.split(r"\n(?=\d+[\.\-]\s+[أ-إ])", content)

        current_pos = 0
        chunk_idx = 0

        for section in sections:
            if len(section.strip()) < 50:
                current_pos += len(section)
                continue

            # Extract ruling number
            num_match = re.match(r"^(\d+)", section.strip())
            ruling_num = num_match.group(1) if num_match else str(chunk_idx)

            # Try to extract the actual ruling
            ruling_match = re.search(
                r"(?:يجب|يحرم|يجوز|يصح|لا يجوز|يسن|يستحب)\s+[^\.]+", section[:200]
            )
            ruling_text = ruling_match.group(0) if ruling_match else None

            chunks.append(
                Chunk(
                    chunk_id=f"{book_id}_ruling_{ruling_num}",
                    content=section.strip()[: self.chunk_size],
                    book_id=book_id,
                    book_title=book_title,
                    author=author,
                    category=category,
                    chunk_type="ruling",
                    start_char=current_pos,
                    end_char=current_pos + len(section),
                    metadata={
                        "ruling_number": ruling_num,
                        "ruling_summary": ruling_text,
                    },
                )
            )

            current_pos += len(section)
            chunk_idx += 1

        # If no rulings found, use semantic chunking
        if not chunks:
            return self._chunk_by_semantic(
                content, book_id, book_title, author, category
            )

        return chunks

    # =========================================================================
    # Semantic Chunking (general)
    # =========================================================================

    def _chunk_by_semantic(
        self,
        content: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
    ) -> List[Chunk]:
        """
        Chunk by semantic sections - preserves meaning.

        Uses multiple signals to identify natural breaks:
        - Chapter titles
        - Paragraphs
        - Section markers
        """

        chunks = []

        # First try to split by chapters/sections
        chapter_patterns = [
            r"\n##\s+",  # Markdown headers
            r"\n###\s+",  # Markdown subheaders
            r"\n\*+\s*",  # Bullet points
            r"\n\d+\.\s+[أ-إ]",  # Numbered Arabic
        ]

        # Try splitting by chapters
        sections = None
        for pattern in chapter_patterns:
            sections = re.split(pattern, content)
            if len(sections) > 1:
                break

        # If no clear sections, use paragraphs
        if not sections or len(sections) <= 1:
            sections = content.split("\n\n")

        current_pos = 0
        chunk_idx = 0
        current_chunk = []
        current_length = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Check if adding this section exceeds chunk size
            if current_length + len(section) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_content = "\n\n".join(current_chunk)

                chunks.append(
                    Chunk(
                        chunk_id=f"{book_id}_chunk_{chunk_idx}",
                        content=chunk_content,
                        book_id=book_id,
                        book_title=book_title,
                        author=author,
                        category=category,
                        chunk_type="general",
                        start_char=current_pos - len(chunk_content),
                        end_char=current_pos,
                        metadata={},
                    )
                )

                # Start new chunk with overlap
                overlap_text = "\n\n".join(current_chunk[-2:])
                current_chunk = [overlap_text, section]
                current_length = len(overlap_text) + len(section)
                chunk_idx += 1
            else:
                current_chunk.append(section)
                current_length += len(section)

            current_pos += len(section)

        # Add remaining content
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_id=f"{book_id}_chunk_{chunk_idx}",
                    content=chunk_content,
                    book_id=book_id,
                    book_title=book_title,
                    author=author,
                    category=category,
                    chunk_type="general",
                    start_char=current_pos - len(chunk_content),
                    end_char=current_pos,
                    metadata={},
                )
            )

        return chunks


# ============================================================================
# Utility Functions
# ============================================================================


def create_chunks_from_book(
    book_data: Dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Create chunks from processed book data.

    Args:
        book_data: Processed book from BookCleanedData
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunk dictionaries
    """

    chunker = IslamicLiteratureChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = chunker.chunk_book(
        content=book_data["content"],
        book_id=book_data["book_id"],
        book_title=book_data["title"],
        author=book_data["author"],
        category=book_data["category"],
    )

    # Convert to dictionaries
    return [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "book_id": c.book_id,
            "book_title": c.book_title,
            "author": c.author,
            "category": c.category,
            "chunk_type": c.chunk_type,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "metadata": c.metadata,
        }
        for c in chunks
    ]


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Test chunking
    sample_text = """
    ## مقدمة
    
    الحمد لله رب العالمين، والصلاة والسلام على أشرف المرسلين.
    
    ## الفصل الأول: الإيمان
    
    الإيمان بالله تعالى هو أصل الدين.
    
    حديث: حدثنا أبو هريرة قال: قال النبي صلى الله عليه وسلم: "الإيمان بضع وسبعون شعبة".
    
    حكم: يجب على كل مسلم الإيمان بالله ورسوله.
    """

    chunker = IslamicLiteratureChunker(chunk_size=500)

    chunks = chunker._chunk_by_semantic(
        sample_text,
        book_id=1,
        book_title="كتاب الإيمان",
        author="الشيخ",
        category="العقيدة",
    )

    print(f"Created {len(chunks)} chunks")

    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_id} ---")
        print(f"Type: {chunk.chunk_type}")
        print(f"Content: {chunk.content[:100]}...")
