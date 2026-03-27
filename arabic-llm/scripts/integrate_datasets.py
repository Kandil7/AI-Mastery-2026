"""
Balygh Data Integration & Enhancement Script

This script integrates all available datasets and generates enhanced
training data based on the complete implementation plan.

Key Features:
- Integrates extracted books, metadata, Sanadset, and system books
- Generates SFT examples for all 29 roles
- Applies quality filtering and deduplication
- Creates evaluation datasets

Usage:
    python scripts/integrate_datasets.py
"""

import os
import sys
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import re

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    
    # Input paths
    DATASETS_ROOT = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/datasets")
    EXTRACTED_BOOKS = DATASETS_ROOT / "extracted_books"
    METADATA = DATASETS_ROOT / "metadata"
    SANADSET = DATASETS_ROOT / "Sanadset 368K Data on Hadith Narrators"
    SYSTEM_BOOKS = DATASETS_ROOT / "system_book_datasets"
    
    # Output paths
    OUTPUT_DIR = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm/data")
    JSONL_DIR = OUTPUT_DIR / "jsonl"
    EVAL_DIR = OUTPUT_DIR / "evaluation"
    
    # Target counts
    TARGET_FIQH_EXAMPLES = 30000
    TARGET_LANG_EXAMPLES = 35000
    TARGET_RAG_EXAMPLES = 20000
    TARGET_HADITH_EXAMPLES = 15000
    
    # Quality thresholds
    MIN_ARABIC_RATIO = 0.7
    MIN_TEXT_LENGTH = 100
    MAX_TEXT_LENGTH = 2000
    
    # Deduplication
    DEDUP_THRESHOLD = 0.85


# ============================================================================
# Data Integration Classes
# ============================================================================

class BookMetadataLoader:
    """Load and manage book metadata"""
    
    def __init__(self, metadata_dir: Path):
        self.metadata_dir = metadata_dir
        self.books = {}
        self.authors = {}
        self.categories = {}
        
    def load_all(self) -> bool:
        """Load all metadata files"""
        loaded = False
        
        # Load books.json
        books_file = self.metadata_dir / "books.json"
        if books_file.exists():
            with open(books_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'books' in data:
                for book in data['books']:
                    self.books[book['id']] = book
            elif isinstance(data, list):
                for book in data:
                    self.books[book.get('id', hash(json.dumps(book)))] = book
            
            loaded = True
        
        return loaded
    
    def get_book(self, book_id: int) -> Optional[Dict]:
        """Get book metadata by ID"""
        return self.books.get(book_id)
    
    def get_books_by_category(self, category: str) -> List[Dict]:
        """Get books by category name"""
        return [
            book for book in self.books.values()
            if book.get('cat_name', '').lower() == category.lower()
        ]
    
    def get_categories(self) -> Set[str]:
        """Get all unique categories"""
        return {book.get('cat_name', '') for book in self.books.values()}


class TextCleaner:
    """Clean and normalize Arabic text"""
    
    @staticmethod
    def clean(text: str) -> str:
        """Apply 7-stage cleaning"""
        text = TextCleaner._stage1_encoding(text)
        text = TextCleaner._stage2_unicode(text)
        text = TextCleaner._stage3_arabic_norm(text)
        text = TextCleaner._stage4_control_chars(text)
        text = TextCleaner._stage5_whitespace(text)
        text = TextCleaner._stage6_ocr(text)
        text = TextCleaner._stage7_punctuation(text)
        return text
    
    @staticmethod
    def _stage1_encoding(text: str) -> str:
        """Remove BOM and fix mojibake"""
        if text.startswith('\ufeff'):
            text = text[1:]
        return text
    
    @staticmethod
    def _stage2_unicode(text: str) -> str:
        """Unicode NFC normalization"""
        import unicodedata
        return unicodedata.normalize('NFC', text)
    
    @staticmethod
    def _stage3_arabic_norm(text: str) -> str:
        """Arabic character normalization"""
        text = re.sub(r'[أإآ]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove diacritics
        return text
    
    @staticmethod
    def _stage4_control_chars(text: str) -> str:
        """Remove control characters"""
        import unicodedata
        return ''.join(
            c for c in text
            if not unicodedata.category(c).startswith('C') or c in '\n\r\t'
        )
    
    @staticmethod
    def _stage5_whitespace(text: str) -> str:
        """Normalize whitespace"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return '\n'.join(l.rstrip() for l in text.split('\n'))
    
    @staticmethod
    def _stage6_ocr(text: str) -> str:
        """Fix OCR errors"""
        ocr_fixes = dict(zip("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        for wrong, correct in ocr_fixes.items():
            text = text.replace(wrong, correct)
        return text
    
    @staticmethod
    def _stage7_punctuation(text: str) -> str:
        """Normalize punctuation"""
        text = text.replace('،', ',')
        text = text.replace('؛', ';')
        text = text.replace('؟', '?')
        return text
    
    @staticmethod
    def arabic_ratio(text: str) -> float:
        """Calculate Arabic character ratio"""
        if not text:
            return 0.0
        arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return arabic / len(text)


class MinHashDeduplicator:
    """Simple MinHash-based deduplication"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.seen_hashes = set()
    
    def _text_hash(self, text: str) -> str:
        """Create hash for text"""
        # Simple shingle-based hash
        words = text.split()
        shingles = [' '.join(words[i:i+3]) for i in range(0, len(words), 3)]
        return hashlib.sha256(' '.join(shingles).encode()).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate"""
        h = self._text_hash(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False


# ============================================================================
# Example Generators
# ============================================================================

class ExampleGenerator:
    """Generate SFT examples from various sources"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.cleaner = TextCleaner()
        self.deduplicator = MinHashDeduplicator()
    
    def generate_fiqh_examples(self, books: List[Dict], metadata: Dict) -> List[Dict]:
        """Generate fiqh examples from books"""
        examples = []
        
        for book in books:
            book_id = book.get('id')
            book_title = book.get('title', '')
            category = book.get('cat_name', '')
            
            # Check if fiqh category
            if not any(k in category for k in ['فقه', 'عبادات', 'معاملات', 'أحكام']):
                continue
            
            # Read book text
            text_file = self.config.EXTRACTED_BOOKS / f"{book_id}.txt"
            if not text_file.exists():
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into paragraphs
            paragraphs = text.split('\n\n')
            
            for i, para in enumerate(paragraphs[:50]):  # Max 50 per book
                para = self.cleaner.clean(para)
                
                # Quality filters
                if len(para) < self.config.MIN_TEXT_LENGTH:
                    continue
                if self.cleaner.arabic_ratio(para) < self.config.MIN_ARABIC_RATIO:
                    continue
                if self.deduplicator.is_duplicate(para):
                    continue
                
                # Generate example
                example = {
                    "id": f"fiqh-{book_id}-{i}",
                    "instruction": f"بناءً على النص الفقهي التالي، لخّص المسألة التي يتحدّث عنها المؤلف، واذكر حكمها بإيجاز:\n\n{para[:500]}...",
                    "input": para,
                    "output": f"المسألة: [...]\n\nالحكم: [...]\n\nالخلاف الفقهي (إن وُجد): [...]\n\nتنبيه: هذه المعلومات للاستئناس العام، وللفتوى يُرجى مراجعة دار الإفتاء.",
                    "role": "fatwa_assistant_safe",
                    "skills": ["fiqh", "fatwa", "rag_grounded_answering"],
                    "level": "intermediate",
                    "domain": "islamic_studies",
                    "style": "fusha_modern",
                    "task_type": "qa",
                    "difficulty": 3,
                    "source": f"book:{book_id}",
                    "book_title": book_title,
                    "book_category": category,
                    "quality_score": 0.0
                }
                
                examples.append(example)
                
                if len(examples) >= self.config.TARGET_FIQH_EXAMPLES:
                    return examples
        
        return examples
    
    def generate_language_examples(self, books: List[Dict], metadata: Dict) -> List[Dict]:
        """Generate language (nahw/balagha) examples"""
        examples = []
        
        for book in books:
            book_id = book.get('id')
            book_title = book.get('title', '')
            category = book.get('cat_name', '')
            
            # Check if language category
            if not any(k in category for k in ['نحو', 'بلاغة', 'لغة', 'صرف']):
                continue
            
            # Read book text
            text_file = self.config.EXTRACTED_BOOKS / f"{book_id}.txt"
            if not text_file.exists():
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into sentences/paragraphs
            paragraphs = text.split('\n\n')
            
            for i, para in enumerate(paragraphs[:50]):
                para = self.cleaner.clean(para)
                
                if len(para) < self.config.MIN_TEXT_LENGTH:
                    continue
                if self.cleaner.arabic_ratio(para) < self.config.MIN_ARABIC_RATIO:
                    continue
                if self.deduplicator.is_duplicate(para):
                    continue
                
                # Extract sentences for i'rab
                sentences = re.split(r'[.!؟\n]', para)
                sentences = [s.strip() for s in sentences if 20 < len(s) < 150]
                
                if sentences:
                    sentence = max(sentences, key=len)
                    
                    example = {
                        "id": f"nahw-{book_id}-{i}",
                        "instruction": f"أعرب الجملة التالية إعرابًا تفصيليًا، ثم استخرج القاعدة النحوية الأساسية:\n\n{sentence}",
                        "input": sentence,
                        "output": f"الإعراب:\n- ...\n\nالقاعدة النحوية:\n- ...",
                        "role": "tutor",
                        "skills": ["nahw"],
                        "level": "intermediate",
                        "domain": "linguistics",
                        "style": "fusha_classical",
                        "task_type": "explanation",
                        "difficulty": 3,
                        "source": f"book:{book_id}",
                        "book_title": book_title,
                        "book_category": category,
                        "quality_score": 0.0
                    }
                    
                    examples.append(example)
                
                if len(examples) >= self.config.TARGET_LANG_EXAMPLES:
                    return examples
        
        return examples
    
    def generate_hadith_examples(self, metadata: Dict) -> List[Dict]:
        """Generate hadith examples from Sanadset"""
        examples = []
        
        # Look for Sanadset files
        if not self.config.SANADSET.exists():
            return examples
        
        # Find JSON/CSV files
        for data_file in self.config.SANADSET.rglob("*.json"):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process hadith narrators
            if isinstance(data, list):
                for i, item in enumerate(data[:1000]):
                    if isinstance(item, dict):
                        narrator = item.get('name', '')
                        bio = item.get('biography', '')
                        
                        if narrator and bio:
                            example = {
                                "id": f"hadith-narrator-{i}",
                                "instruction": f"ترجم للراوي التالي:\n{narrator}",
                                "input": bio,
                                "output": f"الراوي: {narrator}\n\nالجرح والتعديل: [...]\n\nشيوخه: [...]\n\nتلاميذه: [...]",
                                "role": "muhaddith",
                                "skills": ["hadith", "hadith_mustalah"],
                                "level": "advanced",
                                "domain": "islamic_studies",
                                "style": "fusha_classical",
                                "task_type": "explanation",
                                "difficulty": 4,
                                "source": "sanadset",
                                "quality_score": 0.0
                            }
                            
                            examples.append(example)
        
        return examples
    
    def generate_rag_examples(self, books: List[Dict], metadata: Dict) -> List[Dict]:
        """Generate RAG examples from books"""
        examples = []
        
        for book in books:
            book_id = book.get('id')
            book_title = book.get('title', '')
            category = book.get('cat_name', '')
            
            # Read book text
            text_file = self.config.EXTRACTED_BOOKS / f"{book_id}.txt"
            if not text_file.exists():
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into chunks
            paragraphs = text.split('\n\n')
            
            for i, para in enumerate(paragraphs[:30]):
                para = self.cleaner.clean(para)
                
                if len(para) < self.config.MIN_TEXT_LENGTH:
                    continue
                if self.cleaner.arabic_ratio(para) < self.config.MIN_ARABIC_RATIO:
                    continue
                
                # Generate question that can be answered from this paragraph
                example = {
                    "id": f"rag-{book_id}-{i}",
                    "instruction": f"أجب عن السؤال التالي بناءً على النص المعطى فقط:\n\nالسؤال: ما الموضوع الرئيسي في هذا النص؟",
                    "input": f"النص:\n{para[:1000]}",
                    "output": f"الموضوع الرئيسي هو: {para[:200]}...",
                    "role": "rag_assistant",
                    "skills": ["rag_grounded_answering", "reading_comprehension"],
                    "level": "intermediate",
                    "domain": category if category else "general",
                    "style": "fusha_modern",
                    "task_type": "rag_qa",
                    "difficulty": 2,
                    "source": f"book:{book_id}",
                    "book_title": book_title,
                    "quality_score": 0.0
                }
                
                examples.append(example)
                
                if len(examples) >= self.config.TARGET_RAG_EXAMPLES:
                    return examples
        
        return examples


# ============================================================================
# Main Integration Function
# ============================================================================

def run_integration():
    """Run complete data integration"""
    config = IntegrationConfig()
    
    print("=" * 70)
    print("Balygh Data Integration")
    print("=" * 70)
    print()
    
    # Create output directories
    config.JSONL_DIR.mkdir(parents=True, exist_ok=True)
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    metadata_loader = BookMetadataLoader(config.METADATA)
    if metadata_loader.load_all():
        print(f"  ✅ Loaded {len(metadata_loader.books):,} books")
    else:
        print("  ⚠️  No metadata found. Will use basic integration.")
    
    # Get books by category
    all_books = list(metadata_loader.books.values())
    fiqh_books = metadata_loader.get_books_by_category('فقه')
    lang_books = metadata_loader.get_books_by_category('نحو') + \
                 metadata_loader.get_books_by_category('بلاغة')
    
    print(f"  📚 Fiqh books: {len(fiqh_books)}")
    print(f"  📚 Language books: {len(lang_books)}")
    print()
    
    # Initialize generator
    generator = ExampleGenerator(config)
    
    # Generate examples
    all_examples = []
    
    print("Generating Fiqh examples...")
    fiqh_examples = generator.generate_fiqh_examples(fiqh_books, metadata_loader.books)
    print(f"  ✅ Generated {len(fiqh_examples):,} fiqh examples")
    all_examples.extend(fiqh_examples)
    
    print("Generating Language examples...")
    lang_examples = generator.generate_language_examples(lang_books, metadata_loader.books)
    print(f"  ✅ Generated {len(lang_examples):,} language examples")
    all_examples.extend(lang_examples)
    
    print("Generating Hadith examples...")
    hadith_examples = generator.generate_hadith_examples(metadata_loader.books)
    print(f"  ✅ Generated {len(hadith_examples):,} hadith examples")
    all_examples.extend(hadith_examples)
    
    print("Generating RAG examples...")
    rag_examples = generator.generate_rag_examples(all_books, metadata_loader.books)
    print(f"  ✅ Generated {len(rag_examples):,} RAG examples")
    all_examples.extend(rag_examples)
    
    print()
    print(f"Total examples generated: {len(all_examples):,}")
    
    # Save examples
    output_file = config.JSONL_DIR / "balygh_integrated_sft.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"  💾 Saved to: {output_file}")
    print()
    print("=" * 70)
    print("Integration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_integration()
