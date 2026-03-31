"""
Arabic Text Deduplication Module - Balygh (بليغ)

This module provides 3-level deduplication for Arabic text:
1. Exact document dedup (SHA-256 hash)
2. Near-duplicate dedup (MinHash LSH, threshold=0.8)
3. Sentence-level dedup (3-sentence spans)

Based on the implementation plan from llm_arabic_plan.md
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
import re

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False
    import warnings
    warnings.warn("datasketch not installed. Install with: pip install datasketch")


# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Document:
    """A text document for deduplication"""
    id: str
    text: str
    source: str = ""
    metadata: Dict = field(default_factory=dict)
    hash: str = ""
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(self.text.encode('utf-8')).hexdigest()


@dataclass
class DeduplicationStats:
    """Statistics about deduplication process"""
    total_documents: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    sentence_duplicates: int = 0
    unique_documents: int = 0
    
    total_chars_before: int = 0
    total_chars_after: int = 0
    
    start_time: str = ""
    end_time: str = ""
    
    def to_dict(self) -> dict:
        return {
            "total_documents": self.total_documents,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "sentence_duplicates": self.sentence_duplicates,
            "unique_documents": self.unique_documents,
            "total_chars_before": self.total_chars_before,
            "total_chars_after": self.total_chars_after,
            "deduplication_ratio": round(1 - (self.unique_documents / max(self.total_documents, 1)), 3),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# ============================================================================
# LEVEL 1: EXACT DEDUPLICATION
# ============================================================================

class ExactDeduplicator:
    """
    Exact document deduplication using SHA-256 hashing.
    
    Fast O(1) lookup for duplicate detection.
    """
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.stats = {
            "total": 0,
            "duplicates": 0,
        }
    
    def add(self, doc: Document) -> bool:
        """
        Add document and return True if unique, False if duplicate.
        
        Args:
            doc: Document to add
            
        Returns:
            True if document is unique, False if duplicate
        """
        self.stats["total"] += 1
        
        if doc.hash in self.seen_hashes:
            self.stats["duplicates"] += 1
            return False
        
        self.seen_hashes.add(doc.hash)
        return True
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is an exact duplicate"""
        hash_val = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return hash_val in self.seen_hashes
    
    def get_stats(self) -> dict:
        """Get deduplication statistics"""
        return {
            "total_processed": self.stats["total"],
            "exact_duplicates": self.stats["duplicates"],
            "unique_count": len(self.seen_hashes),
            "duplicate_ratio": round(self.stats["duplicates"] / max(self.stats["total"], 1), 3),
        }


# ============================================================================
# LEVEL 2: NEAR-DUPLICATE DEDUPLICATION (MinHash LSH)
# ============================================================================

class NearDuplicateDeduplicator:
    """
    Near-duplicate detection using MinHash and Locality Sensitive Hashing (LSH).
    
    Identifies documents that are similar but not identical (e.g., 80%+ similar).
    
    Uses character-level n-gram shingles for Arabic text.
    """
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        """
        Initialize MinHash LSH deduplicator.
        
        Args:
            threshold: Similarity threshold (0.0-1.0) for considering duplicates
            num_perm: Number of permutation functions for MinHash (higher = more accurate)
        """
        if not DATASKETCH_AVAILABLE:
            raise ImportError(
                "datasketch library required for near-duplicate detection. "
                "Install with: pip install datasketch"
            )
        
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes: Dict[str, MinHash] = {}
        self.documents: Dict[str, Document] = {}
        self.stats = {
            "total": 0,
            "near_duplicates": 0,
        }
    
    def _shingles(self, text: str, n: int = 5) -> Set[str]:
        """
        Create character-level n-gram shingles for Arabic text.
        
        Args:
            text: Input text
            n: Size of n-grams (default 5 for Arabic)
            
        Returns:
            Set of n-gram shingles
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create character n-grams
        shingles = set()
        for i in range(max(len(text) - n + 1, 1)):
            shingle = text[i:i+n]
            shingles.add(shingle)
        
        return shingles
    
    def add(self, doc: Document) -> Tuple[bool, Optional[List[str]]]:
        """
        Add document and return (is_unique, similar_doc_ids).
        
        Args:
            doc: Document to add
            
        Returns:
            Tuple of (is_unique, list of similar document IDs if any)
        """
        self.stats["total"] += 1
        
        # Create MinHash for document
        m = MinHash(num_perm=self.num_perm)
        shingles = self._shingles(doc.text)
        
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        
        # Query LSH for similar documents
        similar_docs = self.lsh.query(m)
        
        if similar_docs:
            self.stats["near_duplicates"] += 1
            return False, similar_docs
        
        # Insert into LSH
        try:
            self.lsh.insert(doc.id, m)
            self.minhashes[doc.id] = m
            self.documents[doc.id] = doc
            return True, None
        except ValueError as e:
            logger.warning(f"Failed to insert document {doc.id}: {e}")
            return False, None
    
    def is_similar(self, text: str, threshold: float = None) -> Tuple[bool, List[str]]:
        """
        Check if text is similar to any indexed document.
        
        Args:
            text: Text to check
            threshold: Optional override for similarity threshold
            
        Returns:
            Tuple of (is_similar, list of similar document IDs)
        """
        if not DATASKETCH_AVAILABLE:
            return False, []
        
        m = MinHash(num_perm=self.num_perm)
        shingles = self._shingles(text)
        
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))
        
        similar = self.lsh.query(m)
        return len(similar) > 0, similar
    
    def get_stats(self) -> dict:
        """Get near-deduplication statistics"""
        return {
            "total_processed": self.stats["total"],
            "near_duplicates": self.stats["near_duplicates"],
            "unique_count": len(self.documents),
            "threshold": self.threshold,
            "num_perm": self.num_perm,
        }


# ============================================================================
# LEVEL 3: SENTENCE-LEVEL DEDUPLICATION
# ============================================================================

class SentenceDeduplicator:
    """
    Sentence-level deduplication to remove repeated sentences within documents.
    
    Uses a sliding window approach to detect repeated 3-sentence spans.
    """
    
    def __init__(self, min_sentence_length: int = 20, window_size: int = 3):
        """
        Initialize sentence deduplicator.
        
        Args:
            min_sentence_length: Minimum characters for a valid sentence
            window_size: Number of sentences in sliding window
        """
        self.min_sentence_length = min_sentence_length
        self.window_size = window_size
        self.stats = {
            "total_sentences": 0,
            "duplicate_sentences": 0,
            "documents_processed": 0,
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split Arabic text into sentences.
        
        Handles Arabic punctuation (. ? ! ,)
        """
        # Arabic sentence delimiters
        delimiters = r'[.?!،؛]'
        
        sentences = re.split(delimiters, text)
        
        # Filter empty and too short sentences
        sentences = [
            s.strip() for s in sentences 
            if len(s.strip()) >= self.min_sentence_length
        ]
        
        return sentences
    
    def _sentence_hash(self, sentence: str) -> str:
        """Create hash for a sentence"""
        # Normalize whitespace and diacritics
        normalized = re.sub(r'\s+', ' ', sentence).strip()
        normalized = re.sub(r'[\u064B-\u065F]', '', normalized)  # Remove diacritics
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def deduplicate_document(self, text: str) -> str:
        """
        Remove duplicate sentences from a document.
        
        Args:
            text: Input text with potential duplicate sentences
            
        Returns:
            Text with duplicate sentences removed
        """
        self.stats["documents_processed"] += 1
        
        sentences = self._split_sentences(text)
        self.stats["total_sentences"] += len(sentences)
        
        if len(sentences) <= self.window_size:
            return text
        
        # Track seen sentence hashes
        seen_hashes: Set[str] = set()
        unique_sentences: List[str] = []
        
        # Sliding window approach
        for i in range(len(sentences)):
            # Create window hash
            window_start = max(0, i - self.window_size + 1)
            window = sentences[window_start:i+1]
            
            # Hash the window
            window_text = ' '.join(window)
            window_hash = self._sentence_hash(window_text)
            
            # Check if this window was seen before
            if window_hash not in seen_hashes:
                seen_hashes.add(window_hash)
                unique_sentences.append(sentences[i])
            else:
                self.stats["duplicate_sentences"] += 1
        
        # Reconstruct text
        return ' '.join(unique_sentences)
    
    def get_stats(self) -> dict:
        """Get sentence deduplication statistics"""
        return {
            "documents_processed": self.stats["documents_processed"],
            "total_sentences": self.stats["total_sentences"],
            "duplicate_sentences": self.stats["duplicate_sentences"],
            "duplicate_ratio": round(
                self.stats["duplicate_sentences"] / max(self.stats["total_sentences"], 1), 
                3
            ),
        }


# ============================================================================
# COMBINED DEDUPLICATION PIPELINE
# ============================================================================

class ArabicDeduplicationPipeline:
    """
    Complete 3-level deduplication pipeline for Arabic text.
    
    Combines:
    1. Exact deduplication (SHA-256)
    2. Near-duplicate detection (MinHash LSH)
    3. Sentence-level deduplication
    
    Usage:
        pipeline = ArabicDeduplicationPipeline()
        unique_docs = pipeline.deduplicate(documents)
    """
    
    def __init__(
        self,
        lsh_threshold: float = 0.8,
        lsh_num_perm: int = 128,
        min_sentence_length: int = 20,
    ):
        """
        Initialize deduplication pipeline.
        
        Args:
            lsh_threshold: Similarity threshold for near-duplicate detection
            lsh_num_perm: Number of permutations for MinHash
            min_sentence_length: Minimum sentence length for sentence dedup
        """
        self.exact_dedup = ExactDeduplicator()
        
        if DATASKETCH_AVAILABLE:
            self.near_dedup = NearDuplicateDeduplicator(
                threshold=lsh_threshold,
                num_perm=lsh_num_perm
            )
        else:
            self.near_dedup = None
        
        self.sentence_dedup = SentenceDeduplicator(
            min_sentence_length=min_sentence_length
        )
        
        self.stats = DeduplicationStats()
    
    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        Deduplicate a list of documents using all 3 levels.
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            List of unique documents
        """
        self.stats.start_time = datetime.now().isoformat()
        self.stats.total_documents = len(documents)
        
        unique_docs: List[Document] = []
        
        for doc in documents:
            self.stats.total_chars_before += len(doc.text)
            
            # Level 1: Exact deduplication
            if not self.exact_dedup.add(doc):
                self.stats.exact_duplicates += 1
                continue
            
            # Level 2: Near-duplicate detection
            if self.near_dedup:
                is_unique, similar_ids = self.near_dedup.add(doc)
                if not is_unique:
                    self.stats.near_duplicates += 1
                    logger.debug(f"Document {doc.id} is near-duplicate of {similar_ids}")
                    continue
            
            # Level 3: Sentence-level deduplication
            cleaned_text = self.sentence_dedup.deduplicate_document(doc.text)
            doc.text = cleaned_text
            
            # Add to unique list
            unique_docs.append(doc)
            self.stats.total_chars_after += len(doc.text)
        
        self.stats.unique_documents = len(unique_docs)
        self.stats.end_time = datetime.now().isoformat()
        
        logger.info(f"Deduplication complete: {len(unique_docs)}/{len(documents)} documents kept")
        
        return unique_docs
    
    def deduplicate_stream(
        self, 
        document_generator: Generator[Document, None, None]
    ) -> Generator[Document, None, None]:
        """
        Deduplicate documents from a generator (memory-efficient).
        
        Args:
            document_generator: Generator yielding documents
            
        Yields:
            Unique documents only
        """
        for doc in document_generator:
            self.stats.total_documents += 1
            self.stats.total_chars_before += len(doc.text)
            
            # Level 1: Exact deduplication
            if not self.exact_dedup.add(doc):
                self.stats.exact_duplicates += 1
                continue
            
            # Level 2: Near-duplicate detection
            if self.near_dedup:
                is_unique, similar_ids = self.near_dedup.add(doc)
                if not is_unique:
                    self.stats.near_duplicates += 1
                    continue
            
            # Level 3: Sentence-level deduplication
            cleaned_text = self.sentence_dedup.deduplicate_document(doc.text)
            doc.text = cleaned_text
            
            self.stats.total_chars_after += len(doc.text)
            self.stats.unique_documents += 1
            
            yield doc
    
    def get_stats(self) -> dict:
        """Get comprehensive deduplication statistics"""
        return {
            "summary": self.stats.to_dict(),
            "exact_dedup": self.exact_dedup.get_stats(),
            "near_dedup": self.near_dedup.get_stats() if self.near_dedup else None,
            "sentence_dedup": self.sentence_dedup.get_stats(),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def deduplicate_texts(
    texts: List[str],
    method: str = "all",
    threshold: float = 0.8
) -> Tuple[List[str], dict]:
    """
    Deduplicate a list of texts.
    
    Args:
        texts: List of texts to deduplicate
        method: "exact", "near", or "all"
        threshold: Similarity threshold for near-dedup
        
    Returns:
        Tuple of (unique_texts, statistics)
    """
    # Convert to documents
    documents = [
        Document(id=f"doc_{i}", text=text, source="input")
        for i, text in enumerate(texts)
    ]
    
    # Create pipeline
    pipeline = ArabicDeduplicationPipeline(lsh_threshold=threshold)
    
    # Deduplicate
    unique_docs = pipeline.deduplicate(documents)
    
    # Extract texts
    unique_texts = [doc.text for doc in unique_docs]
    
    # Get stats
    stats = pipeline.get_stats()
    
    return unique_texts, stats


def load_documents_from_jsonl(file_path: str) -> List[Document]:
    """
    Load documents from JSONL file.
    
    Args:
        file_path: Path to JSONL file with 'text' and 'id' fields
        
    Returns:
        List of Document objects
    """
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            doc = Document(
                id=data.get('id', f"doc_{i}"),
                text=data.get('text', data.get('content', data.get('input', ''))),
                source=data.get('source', str(file_path)),
                metadata=data
            )
            
            documents.append(doc)
    
    return documents


def save_documents_to_jsonl(documents: List[Document], output_path: str):
    """
    Save documents to JSONL file.
    
    Args:
        documents: List of Document objects
        output_path: Path to output file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            record = {
                'id': doc.id,
                'text': doc.text,
                'source': doc.source,
                'metadata': doc.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ============================================================================
# MAIN - CLI USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Example usage
    print("Arabic Text Deduplication Module - Balygh (بليغ)")
    print("=" * 60)
    
    if DATASKETCH_AVAILABLE:
        print("✓ datasketch library available - full deduplication enabled")
    else:
        print("✗ datasketch library not available - exact dedup only")
        print("  Install with: pip install datasketch")
    
    # Example
    sample_texts = [
        "هذا نص تجريبي أول",
        "هذا نص تجريبي أول",  # Exact duplicate
        "هذا نص تجريبي أول مع زيادة بسيطة",  # Near duplicate
        "هذا نص تجريبي ثاني مختلف تماماً",
        "هذا نص تجريبي أول. هذا نص تجريبي أول. هذا نص تجريبي أول.",  # Sentence duplicates
    ]
    
    print(f"\nInput: {len(sample_texts)} texts")
    
    unique_texts, stats = deduplicate_texts(sample_texts, method="all")
    
    print(f"Output: {len(unique_texts)} unique texts")
    print(f"\nStatistics:")
    print(f"  Exact duplicates: {stats['exact_dedup']['exact_duplicates']}")
    print(f"  Near duplicates: {stats['near_dedup']['near_duplicates'] if stats['near_dedup'] else 'N/A'}")
    print(f"  Sentence duplicates: {stats['sentence_dedup']['duplicate_sentences']}")
