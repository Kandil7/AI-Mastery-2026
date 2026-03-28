"""
System Book Datasets Integration Module

This module provides integration with the structured system_book_datasets
which contains indexed Islamic knowledge databases including:
- Quranic Tafseer (tafseer.db)
- Hadith collections (hadeeth.db)
- Book indexes and metadata (store/)
- Search indexes (Lucene format)

This structured data complements the extracted_books corpus by providing:
- Verified hadith chains (isnad)
- Quranic verse mappings
- Author bibliographies
- Cross-references between books

Architecture:
- service/*.db: Service databases for tafseer, hadith, tarajim
- store/*: Lucene search indexes for fast retrieval
- book/*: 1000+ indexed book segments
- user/*: User data and hints

Integration with Arabic LLM:
- Provides structured training examples
- Enables verification of hadith authenticity
- Supports Quranic exegesis tasks
- Enables scholarly chain of transmission analysis
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging


@dataclass
class HadithRecord:
    """Hadith record with chain of transmission"""
    hadith_id: int
    book_id: int
    book_name: str
    hadith_number: int
    arabic_text: str
    narrator: str  # Rawi (الراوي)
    isnad: str  # Chain of transmission
    grade: Optional[str]  # Hadith grade (صحيح، حسن، ضعيف)
    chapter: str
    reference: str


@dataclass
class TafseerRecord:
    """Tafseer (Quranic exegesis) record"""
    tafseer_id: int
    surah_number: int
    surah_name: str
    ayah_number: int
    arabic_text: str  # Quranic verse
    tafseer_text: str  # Exegesis
    author_id: int
    author_name: str
    book_id: int


@dataclass
class BookIndex:
    """Book index entry from store"""
    book_id: int
    title: str
    author_id: int
    author_name: str
    category_id: int
    segments: List[int]  # Segment IDs


class SystemBookIntegration:
    """
    Integration layer for system_book_datasets.
    
    Provides access to:
    1. Hadith databases with authentication chains
    2. Tafseer databases with verse mappings
    3. Book indexes and cross-references
    4. Lucene search indexes (future integration)
    """
    
    def __init__(self, base_dir: str, logger: Optional[logging.Logger] = None):
        self.base_dir = Path(base_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Database paths
        self.service_dir = self.base_dir / "service"
        self.store_dir = self.base_dir / "store"
        self.book_dir = self.base_dir / "book"
        
        # Initialize database connections
        self._init_databases()
    
    def _init_databases(self):
        """Initialize database connections"""
        self.connections = {}
        
        # Service databases
        service_dbs = ['hadeeth.db', 'tafseer.db', 'trajim.db', 'S1.db', 'S2.db']
        
        for db_name in service_dbs:
            db_path = self.service_dir / db_name
            if db_path.exists():
                try:
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    self.connections[db_name] = conn
                    self.logger.info(f"Connected to {db_name}")
                except Exception as e:
                    self.logger.error(f"Failed to connect to {db_name}: {e}")
    
    def get_hadith(self, hadith_id: int) -> Optional[HadithRecord]:
        """
        Retrieve hadith by ID with full chain of transmission.
        
        Args:
            hadith_id: Hadith identifier
            
        Returns:
            HadithRecord with full metadata
        """
        if 'hadeeth.db' not in self.connections:
            return None
        
        conn = self.connections['hadeeth.db']
        cursor = conn.cursor()
        
        # Query hadith with metadata
        query = """
            SELECT h.*, b.name as book_name, c.name as chapter_name
            FROM service h
            JOIN book b ON h.book_id = b.id
            JOIN chapter c ON h.chapter_id = c.id
            WHERE h.id = ?
        """
        
        cursor.execute(query, (hadith_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return HadithRecord(
            hadith_id=row['id'],
            book_id=row['book_id'],
            book_name=row['book_name'],
            hadith_number=row['hadith_number'],
            arabic_text=row['text'],
            narrator=row['narrator'],
            isnad=row['isnad'],
            grade=row.get('grade'),
            chapter=row.get('chapter_name', ''),
            reference=row['reference']
        )
    
    def get_tafseer(self, surah: int, ayah: int) -> List[TafseerRecord]:
        """
        Retrieve tafseer for a specific Quranic verse.
        
        Args:
            surah: Surah number (1-114)
            ayah: Ayah number
            
        Returns:
            List of TafseerRecord from different scholars
        """
        if 'tafseer.db' not in self.connections:
            return []
        
        conn = self.connections['tafseer.db']
        cursor = conn.cursor()
        
        query = """
            SELECT t.*, a.name as author_name, b.title as book_title
            FROM service t
            JOIN author a ON t.author_id = a.id
            JOIN book b ON t.book_id = b.id
            WHERE t.surah_number = ? AND t.ayah_number = ?
        """
        
        cursor.execute(query, (surah, ayah))
        rows = cursor.fetchall()
        
        return [
            TafseerRecord(
                tafseer_id=row['id'],
                surah_number=row['surah_number'],
                surah_name=row['surah_name'],
                ayah_number=row['ayah_number'],
                arabic_text=row['ayah_text'],
                tafseer_text=row['tafseer_text'],
                author_id=row['author_id'],
                author_name=row['author_name'],
                book_id=row['book_id']
            )
            for row in rows
        ]
    
    def get_book_index(self, book_id: int) -> Optional[BookIndex]:
        """
        Retrieve book index from store.
        
        Args:
            book_id: Book identifier
            
        Returns:
            BookIndex with segment mappings
        """
        # Check Lucene index files
        book_index_dir = self.store_dir / "book"
        
        # Read segment information
        segments_file = book_index_dir / "segments_c"
        if segments_file.exists():
            with open(segments_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
                return BookIndex(
                    book_id=book_id,
                    title=segments.get('title', ''),
                    author_id=segments.get('author_id', 0),
                    author_name=segments.get('author_name', ''),
                    category_id=segments.get('category_id', 0),
                    segments=segments.get('segments', [])
                )
        
        return None
    
    def search_hadith_by_narrator(self, narrator: str) -> List[HadithRecord]:
        """
        Search hadith by narrator (Companion or Successor).
        
        Args:
            narrator: Name of narrator (راوي)
            
        Returns:
            List of hadiths narrated by this person
        """
        if 'hadeeth.db' not in self.connections:
            return []
        
        conn = self.connections['hadeeth.db']
        cursor = conn.cursor()
        
        query = """
            SELECT h.*, b.name as book_name
            FROM service h
            JOIN book b ON h.book_id = b.id
            WHERE h.narrator LIKE ?
        """
        
        cursor.execute(query, (f'%{narrator}%',))
        rows = cursor.fetchall()
        
        return [
            HadithRecord(
                hadith_id=row['id'],
                book_id=row['book_id'],
                book_name=row['book_name'],
                hadith_number=row['hadith_number'],
                arabic_text=row['text'],
                narrator=row['narrator'],
                isnad=row['isnad'],
                grade=row.get('grade'),
                chapter='',
                reference=row['reference']
            )
            for row in rows
        ]
    
    def get_isnad_chain(self, hadith_id: int) -> List[str]:
        """
        Extract full chain of transmission (إسناد) for hadith verification.
        
        Args:
            hadith_id: Hadith identifier
            
        Returns:
            List of narrators in chain from Prophet to collector
        """
        hadith = self.get_hadith(hadith_id)
        if not hadith:
            return []
        
        # Parse isnad (typically: "حدثنا X، حدثنا Y، عن Z، عن النبي")
        isnad_text = hadith.isnad
        narrators = []
        
        # Split by common isnad markers
        markers = ['حدثنا', 'أخبرنا', 'عن', 'سمعت']
        parts = isnad_text
        
        for marker in markers:
            parts = parts.replace(marker, '|')
        
        potential_narrators = [p.strip() for p in parts.split('|') if p.strip()]
        
        # Filter out non-narrator text
        for name in potential_narrators:
            if len(name) > 2 and not any(word in name for word in ['قال', 'الرسول']):
                narrators.append(name)
        
        return narrators
    
    def get_cross_references(self, book_id: int) -> List[Dict]:
        """
        Get cross-references to other books.
        
        Args:
            book_id: Source book ID
            
        Returns:
            List of cross-reference dictionaries
        """
        references = []
        
        # Check esnad (إسناد) directory for chain references
        esnad_dir = self.store_dir / "esnad"
        if esnad_dir.exists():
            for file in esnad_dir.glob("*.txt"):
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if str(book_id) in content:
                        references.append({
                            'type': 'isnad',
                            'source': file.name,
                            'book_id': book_id
                        })
        
        return references
    
    def get_author_bibliography(self, author_id: int) -> List[Dict]:
        """
        Get complete bibliography for an author.
        
        Args:
            author_id: Author identifier
            
        Returns:
            List of books by this author
        """
        bibliography = []
        
        # Check author directory
        author_dir = self.store_dir / "author"
        if author_dir.exists():
            author_file = author_dir / str(author_id)
            if author_file.exists():
                with open(author_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    bibliography = data.get('books', [])
        
        return bibliography
    
    def close(self):
        """Close all database connections"""
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()


# =============================================================================
# INTEGRATION WITH ARABIC LLM SCHEMA
# =============================================================================

def create_hadith_training_examples(integration: SystemBookIntegration, 
                                     limit: int = 100) -> List[Dict]:
    """
    Generate training examples from hadith database.
    
    Creates examples for roles:
    - muhaddith: Hadith authentication
    - faqih: Legal rulings from hadith
    - historian: Historical context
    """
    examples = []
    
    # Sample hadiths from different books
    for book_id in range(1, 10):  # First 9 books (Bukhari, Muslim, etc.)
        for hadith_num in range(1, 12):
            if len(examples) >= limit:
                break
            
            hadith = integration.get_hadith(book_id * 1000 + hadith_num)
            if not hadith:
                continue
            
            # Muhaddith role: Authentication
            examples.append({
                'role': 'muhaddith',
                'skills': ['hadith', 'hadith_mustalah'],
                'instruction': f"بيّن درجة هذا الحديث: «{hadith.arabic_text[:100]}»",
                'input': f"الراوي: {hadith.narrator}\nالإسناد: {hadith.isnad[:200]}",
                'output': f"الحديث: {hadith.grade or 'needs_verification'}\n" +
                         f"المخرج: {hadith.book_name}\n" +
                         f"الرقم: {hadith.hadith_number}",
                'level': 'advanced',
                'domain': 'islamic_studies',
                'source': 'hadeeth_db',
                'hadith_id': hadith.hadith_id
            })
            
            # Faqih role: Legal ruling
            if hadith.grade == 'صحيح':
                examples.append({
                    'role': 'faqih',
                    'skills': ['fiqh', 'hadith'],
                    'instruction': f"ما الحكم الشرعي المستفاد من هذا الحديث؟",
                    'input': hadith.arabic_text[:200],
                    'output': f"الحكم: [يتم استخراج الحكم من الحديث]\n" +
                             f"الدليل: {hadith.book_name}\n" +
                             f"الدرجة: {hadith.grade}",
                    'level': 'advanced',
                    'domain': 'islamic_studies',
                    'source': 'hadeeth_db'
                })
    
    return examples


def create_tafseer_training_examples(integration: SystemBookIntegration,
                                      surah_range: Tuple[int, int] = (1, 10),
                                      limit: int = 100) -> List[Dict]:
    """
    Generate training examples from tafseer database.
    
    Creates examples for roles:
    - mufassir: Quranic exegesis
    - tutor: Teaching tafsir
    - linguist: Linguistic analysis
    """
    examples = []
    
    for surah in range(surah_range[0], surah_range[1] + 1):
        for ayah in range(1, 20):  # First 20 ayahs of each surah
            if len(examples) >= limit:
                break
            
            tafseer_records = integration.get_tafseer(surah, ayah)
            if not tafseer_records:
                continue
            
            # Use first tafseer record
            tafseer = tafseer_records[0]
            
            # Mufassir role: Exegesis
            examples.append({
                'role': 'mufassir',
                'skills': ['tafsir', 'quran_sciences'],
                'instruction': f"فسر قوله تعالى: ﴿{tafseer.arabic_text}﴾",
                'input': f"السورة: {tafseer.surah_name}، الآية: {ayah}",
                'output': tafseer.tafseer_text[:500],
                'level': 'advanced',
                'domain': 'islamic_studies',
                'source': 'tafseer_db',
                'author': tafseer.author_name
            })
            
            # Tutor role: Teaching
            examples.append({
                'role': 'tutor',
                'skills': ['tafsir', 'qa'],
                'instruction': f"اشرح معنى هذه الآية للطلاب: ﴿{tafseer.arabic_text[:50]}»",
                'input': f"من سورة {tafseer.surah_name}",
                'output': f"شرح مبسط: {tafseer.tafseer_text[:300]}...",
                'level': 'intermediate',
                'domain': 'education',
                'source': 'tafseer_db'
            })
    
    return examples


def create_isnad_analysis_examples(integration: SystemBookIntegration,
                                    limit: int = 50) -> List[Dict]:
    """
    Generate examples for isnad (chain of transmission) analysis.
    
    Creates examples for roles:
    - muhaddith: Hadith verification
    - genealogist: Narrator biography
    - historian: Historical context
    """
    examples = []
    
    for hadith_id in range(1, limit * 2):
        if len(examples) >= limit:
            break
        
        hadith = integration.get_hadith(hadith_id)
        if not hadith or not hadith.isnad:
            continue
        
        # Get isnad chain
        narrators = integration.get_isnad_chain(hadith_id)
        if len(narrators) < 3:
            continue
        
        # Muhaddith role: Chain analysis
        examples.append({
            'role': 'muhaddith',
            'skills': ['hadith_mustalah', 'genealogy'],
            'instruction': "حلّل سند هذا الحديث من حيث الاتصال والانقطاع",
            'input': f"الحديث: {hadith.arabic_text[:100]}...\n" +
                    f"السند: {hadith.isnad[:300]}",
            'output': f"عدد الرواة: {len(narrators)}\n" +
                     f"الرواة: {' → '.join(narrators[:5])}...\n" +
                     f"الحكم: [متصل/منقطع]",
            'level': 'specialist',
            'domain': 'islamic_studies',
            'source': 'hadeeth_db'
        })
    
    return examples


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for system book integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="System Book Datasets Integration")
    parser.add_argument("--base-dir", default="datasets/system_book_datasets",
                       help="Base directory for system book datasets")
    parser.add_argument("--output-dir", default="data/system_examples",
                       help="Output directory for generated examples")
    parser.add_argument("--limit", type=int, default=100,
                       help="Limit number of examples to generate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("system_integration")
    
    # Initialize integration
    integration = SystemBookIntegration(args.base_dir, logger)
    
    # Generate examples
    logger.info("Generating hadith examples...")
    hadith_examples = create_hadith_training_examples(integration, args.limit)
    logger.info(f"Generated {len(hadith_examples)} hadith examples")
    
    logger.info("Generating tafseer examples...")
    tafseer_examples = create_tafseer_training_examples(integration, (1, 5), args.limit)
    logger.info(f"Generated {len(tafseer_examples)} tafseer examples")
    
    logger.info("Generating isnad analysis examples...")
    isnad_examples = create_isnad_analysis_examples(integration, args.limit // 2)
    logger.info(f"Generated {len(isnad_examples)} isnad examples")
    
    # Save examples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to JSONL
    import json
    all_examples = hadith_examples + tafseer_examples + isnad_examples
    
    output_file = output_dir / "system_examples.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(all_examples)} examples to {output_file}")
    
    # Close connections
    integration.close()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
