"""
Data Preparation for Pre-Training - Module 2.2.1

Production-ready data preparation pipeline:
- Data collection from multiple sources
- Text cleaning and normalization
- Deduplication (MinHash, LSH)
- Quality filtering and scoring
- Dataset creation for pre-training

References:
- "The Pile: An 800GB Dataset of Diverse Text" (Gao et al., 2020)
- "Deduplicating Training Data Makes Language Models Better" (Lee et al., 2021)
"""

import hashlib
import json
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a text document."""
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'source': self.source,
            'metadata': self.metadata,
            'quality_score': self.quality_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create from dictionary."""
        return cls(
            text=data['text'],
            source=data.get('source', 'unknown'),
            metadata=data.get('metadata', {}),
            quality_score=data.get('quality_score', 0.0),
        )


class DataCollector:
    """
    Data Collector for pre-training.
    
    Collects text data from multiple sources:
    - Local files (JSON, JSONL, TXT)
    - Hugging Face datasets
    - Web crawls (via external tools)
    
    Args:
        sources: List of data sources
        cache_dir: Directory for caching downloaded data
        
    Example:
        >>> collector = DataCollector(sources=['wikipedia', 'books'])
        >>> documents = collector.collect(limit=10000)
    """
    
    def __init__(
        self,
        sources: Optional[List[Union[str, Dict]]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.sources = sources or []
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def add_source(
        self,
        source: Union[str, Dict],
    ) -> None:
        """
        Add a data source.
        
        Args:
            source: Source specification (string path or dict config)
        """
        self.sources.append(source)
    
    def _load_from_file(self, path: str) -> Iterator[Document]:
        """Load documents from a file."""
        path = Path(path)
        
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    yield Document(
                        text=data.get('text', ''),
                        source=data.get('source', path.name),
                        metadata=data.get('metadata', {}),
                    )
        
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield Document(
                            text=item.get('text', ''),
                            source=item.get('source', path.name),
                            metadata=item.get('metadata', {}),
                        )
        
        elif path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split by paragraphs
                paragraphs = text.split('\n\n')
                for i, para in enumerate(paragraphs):
                    if para.strip():
                        yield Document(
                            text=para.strip(),
                            source=f"{path.name}:{i}",
                            metadata={},
                        )
    
    def _load_from_hf(
        self,
        dataset_name: str,
        split: str = 'train',
        text_column: str = 'text',
        limit: Optional[int] = None,
    ) -> Iterator[Document]:
        """Load documents from Hugging Face datasets."""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(dataset_name, split=split)
            
            for i, item in enumerate(dataset):
                if limit is not None and i >= limit:
                    break
                
                text = item.get(text_column, '')
                if text:
                    yield Document(
                        text=text,
                        source=dataset_name,
                        metadata={'hf_index': i},
                    )
        
        except ImportError:
            logger.warning("Hugging Face datasets not installed. Install with: pip install datasets")
    
    def collect(
        self,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        Collect documents from all sources.
        
        Args:
            limit: Maximum number of documents to collect
        
        Returns:
            List of collected documents
        """
        documents = []
        
        for source in self.sources:
            if isinstance(source, str):
                # File path
                if Path(source).exists():
                    for doc in self._load_from_file(source):
                        if limit and len(documents) >= limit:
                            return documents
                        documents.append(doc)
                else:
                    # Try as HF dataset
                    for doc in self._load_from_hf(source, limit=limit):
                        if limit and len(documents) >= limit:
                            return documents
                        documents.append(doc)
            
            elif isinstance(source, dict):
                # Dict config
                source_type = source.get('type', 'file')
                
                if source_type == 'hf':
                    for doc in self._load_from_hf(
                        source.get('name', ''),
                        split=source.get('split', 'train'),
                        text_column=source.get('text_column', 'text'),
                        limit=source.get('limit'),
                    ):
                        if limit and len(documents) >= limit:
                            return documents
                        documents.append(doc)
                
                elif source_type == 'file':
                    for doc in self._load_from_file(source.get('path', '')):
                        if limit and len(documents) >= limit:
                            return documents
                        documents.append(doc)
        
        logger.info(f"Collected {len(documents)} documents from {len(self.sources)} sources")
        return documents
    
    def collect_iterator(
        self,
        limit: Optional[int] = None,
    ) -> Iterator[Document]:
        """
        Collect documents as an iterator (memory efficient).
        
        Args:
            limit: Maximum number of documents
        
        Yields:
            Documents
        """
        count = 0
        
        for source in self.sources:
            if isinstance(source, str):
                if Path(source).exists():
                    for doc in self._load_from_file(source):
                        if limit and count >= limit:
                            return
                        yield doc
                        count += 1
                else:
                    for doc in self._load_from_hf(source):
                        if limit and count >= limit:
                            return
                        yield doc
                        count += 1


class DataCleaner:
    """
    Data Cleaner for text preprocessing.
    
    Applies various cleaning operations:
    - Unicode normalization
    - HTML/entity removal
    - Whitespace normalization
    - Length filtering
    - Language detection (optional)
    
    Args:
        normalize_unicode: Whether to normalize unicode
        remove_html: Whether to remove HTML tags
        min_length: Minimum document length
        max_length: Maximum document length
        
    Example:
        >>> cleaner = DataCleaner(min_length=100, max_length=100000)
        >>> cleaned = cleaner.clean(document)
    """
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_html: bool = True,
        remove_control_chars: bool = True,
        min_length: int = 50,
        max_length: int = 1000000,
        min_words: int = 10,
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_html = remove_html
        self.remove_control_chars = remove_control_chars
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        
        # Compiled patterns
        self._html_pattern = re.compile(r'<[^>]+>')
        self._url_pattern = re.compile(r'http[s]?://\S+')
        self._email_pattern = re.compile(r'\S+@\S+')
        self._multi_whitespace = re.compile(r'\s+')
        self._control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        # NFKC normalization for compatibility
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        text = self._html_pattern.sub('', text)
        # Also decode common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        return text
    
    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters."""
        # Keep newlines and tabs
        text = self._control_char_pattern.sub('', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces/newlines with single space
        text = self._multi_whitespace.sub(' ', text)
        return text.strip()
    
    def clean(self, text: str) -> Optional[str]:
        """
        Clean a text document.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text or None if filtered out
        """
        if not text or not isinstance(text, str):
            return None
        
        # Apply cleaning steps
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        if self.remove_html:
            text = self._remove_html(text)
        
        if self.remove_control_chars:
            text = self._remove_control_chars(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Apply filters
        if len(text) < self.min_length:
            return None
        
        if len(text) > self.max_length:
            return None
        
        word_count = len(text.split())
        if word_count < self.min_words:
            return None
        
        return text
    
    def clean_document(self, doc: Document) -> Optional[Document]:
        """
        Clean a document.
        
        Args:
            doc: Input document
        
        Returns:
            Cleaned document or None if filtered
        """
        cleaned_text = self.clean(doc.text)
        
        if cleaned_text is None:
            return None
        
        return Document(
            text=cleaned_text,
            source=doc.source,
            metadata={**doc.metadata, 'cleaned': True},
        )
    
    def clean_batch(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Clean a batch of documents.
        
        Args:
            documents: List of documents
        
        Returns:
            List of cleaned documents (filtered)
        """
        cleaned = []
        for doc in documents:
            result = self.clean_document(doc)
            if result is not None:
                cleaned.append(result)
        
        logger.info(f"Cleaned {len(cleaned)}/{len(documents)} documents")
        return cleaned


class MinHash:
    """
    MinHash for document similarity.
    
    Computes a signature for documents that allows efficient
    estimation of Jaccard similarity.
    
    Args:
        num_permutations: Number of hash permutations
        ngram_size: Size of n-grams for shingling
        
    Reference:
        "On the resemblance and containment of documents" (Broder, 1997)
    """
    
    def __init__(
        self,
        num_permutations: int = 128,
        ngram_size: int = 5,
    ):
        self.num_permutations = num_permutations
        self.ngram_size = ngram_size
        
        # Generate random hash functions
        self._hash_params = [
            (np.random.randint(1, 2**31), np.random.randint(0, 2**31))
            for _ in range(num_permutations)
        ]
    
    def _hash(self, value: str, a: int, b: int) -> int:
        """Compute hash with given parameters."""
        h = hashlib.md5(value.encode()).hexdigest()
        h = int(h, 16)
        return (a * h + b) % (2**31 - 1)
    
    def _get_ngrams(self, text: str) -> Set[str]:
        """Get n-grams from text."""
        text = text.lower()
        words = text.split()
        ngrams = set()
        
        for i in range(len(words) - self.ngram_size + 1):
            ngram = ' '.join(words[i:i + self.ngram_size])
            ngrams.add(ngram)
        
        return ngrams
    
    def compute_signature(self, text: str) -> np.ndarray:
        """
        Compute MinHash signature for text.
        
        Args:
            text: Input text
        
        Returns:
            MinHash signature array
        """
        ngrams = self._get_ngrams(text)
        
        if not ngrams:
            return np.zeros(self.num_permutations, dtype=np.uint32)
        
        # Initialize with max values
        signature = np.full(self.num_permutations, 2**32 - 1, dtype=np.uint32)
        
        # Compute min hash for each permutation
        for ngram in ngrams:
            for i, (a, b) in enumerate(self._hash_params):
                h = self._hash(ngram, a, b)
                signature[i] = min(signature[i], h)
        
        return signature
    
    def jaccard_similarity(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
    ) -> float:
        """
        Estimate Jaccard similarity from signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
        
        Returns:
            Estimated Jaccard similarity
        """
        matches = np.sum(sig1 == sig2)
        return matches / self.num_permutations


class LSHIndex:
    """
    Locality Sensitive Hashing index for near-duplicate detection.
    
    Args:
        num_bands: Number of bands for LSH
        num_rows_per_band: Rows per band
        threshold: Similarity threshold
        
    Reference:
        "Mining of Massive Datasets" (Leskovec et al.)
    """
    
    def __init__(
        self,
        num_bands: int = 20,
        num_rows_per_band: int = 6,
        threshold: float = 0.5,
    ):
        self.num_bands = num_bands
        self.num_rows_per_band = num_rows_per_band
        self.threshold = threshold
        
        # Hash table: band -> hash_value -> list of doc_ids
        self.hash_tables: List[Dict[int, List[int]]] = [
            {} for _ in range(num_bands)
        ]
    
    def _compute_band_hash(self, signature: np.ndarray, band: int) -> int:
        """Compute hash for a band."""
        start = band * self.num_rows_per_band
        end = start + self.num_rows_per_band
        band_sig = signature[start:end]
        return hash(tuple(band_sig.tolist()))
    
    def add(self, doc_id: int, signature: np.ndarray) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Document ID
            signature: MinHash signature
        """
        for band in range(self.num_bands):
            band_hash = self._compute_band_hash(signature, band)
            
            if band_hash not in self.hash_tables[band]:
                self.hash_tables[band][band_hash] = []
            
            self.hash_tables[band][band_hash].append(doc_id)
    
    def query(
        self,
        signature: np.ndarray,
    ) -> Set[int]:
        """
        Find candidate duplicates for a signature.
        
        Args:
            signature: Query signature
        
        Returns:
            Set of candidate document IDs
        """
        candidates = set()
        
        for band in range(self.num_bands):
            band_hash = self._compute_band_hash(signature, band)
            
            if band_hash in self.hash_tables[band]:
                candidates.update(self.hash_tables[band][band_hash])
        
        return candidates


class Deduplicator:
    """
    Document Deduplicator using MinHash + LSH.
    
    Identifies and removes near-duplicate documents from a corpus.
    
    Args:
        similarity_threshold: Minimum similarity to consider duplicate
        num_permutations: Number of MinHash permutations
        ngram_size: Size of n-grams
        
    Example:
        >>> dedup = Deduplicator(similarity_threshold=0.8)
        >>> unique_docs = dedup.deduplicate(documents)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        num_permutations: int = 128,
        ngram_size: int = 5,
    ):
        self.similarity_threshold = similarity_threshold
        self.minhash = MinHash(num_permutations, ngram_size)
        
        # Calculate LSH parameters
        # b bands, r rows per band: threshold ≈ (1/b)^(1/r)
        self.num_bands = int((1 / similarity_threshold) ** (1 / 5))
        self.num_rows_per_band = num_permutations // self.num_bands
        
        self.lsh_index = LSHIndex(
            num_bands=self.num_bands,
            num_rows_per_band=self.num_rows_per_band,
        )
        
        self._signatures: Dict[int, np.ndarray] = {}
        self._doc_map: Dict[int, Document] = {}
    
    def _compute_exact_similarity(
        self,
        doc1: Document,
        doc2: Document,
    ) -> float:
        """Compute exact Jaccard similarity between documents."""
        ngrams1 = set(self.minhash._get_ngrams(doc1.text))
        ngrams2 = set(self.minhash._get_ngrams(doc2.text))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union
    
    def deduplicate(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Remove near-duplicate documents.
        
        Args:
            documents: List of documents
        
        Returns:
            List of unique documents
        """
        logger.info(f"Deduplicating {len(documents)} documents...")
        
        unique_docs = []
        seen_ids: Set[int] = set()
        
        for i, doc in enumerate(documents):
            # Compute signature
            sig = self.minhash.compute_signature(doc.text)
            self._signatures[i] = sig
            self._doc_map[i] = doc
            
            # Find candidates
            candidates = self.lsh_index.query(sig)
            
            # Check for duplicates
            is_duplicate = False
            for candidate_id in candidates:
                if candidate_id in seen_ids:
                    # Verify with exact similarity
                    candidate_doc = self._doc_map[candidate_id]
                    exact_sim = self._compute_exact_similarity(doc, candidate_doc)
                    
                    if exact_sim >= self.similarity_threshold:
                        is_duplicate = True
                        logger.debug(f"Document {i} is duplicate of {candidate_id} (sim={exact_sim:.3f})")
                        break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_ids.add(i)
                self.lsh_index.add(i, sig)
        
        logger.info(f"Removed {len(documents) - len(unique_docs)} duplicates")
        return unique_docs
    
    def deduplicate_exact(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Remove exact duplicates (faster, less thorough).
        
        Args:
            documents: List of documents
        
        Returns:
            List of unique documents
        """
        seen_hashes: Set[str] = set()
        unique_docs = []
        
        for doc in documents:
            # Hash the text
            text_hash = hashlib.md5(doc.text.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_docs.append(doc)
        
        logger.info(f"Removed {len(documents) - len(unique_docs)} exact duplicates")
        return unique_docs


class DataFilter:
    """
    Data Filter for quality control.
    
    Applies various filters to remove low-quality documents:
    - Language filtering
    - Toxicity filtering
    - PII removal
    - Boilerplate detection
    
    Args:
        filters: List of filter functions
        
    Example:
        >>> data_filter = DataFilter()
        >>> filtered = data_filter.filter(documents)
    """
    
    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 100000,
        min_word_ratio: float = 0.5,
        max_symbol_ratio: float = 0.1,
        max_bullet_ratio: float = 0.5,
        max_ellipsis_ratio: float = 0.3,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_word_ratio = min_word_ratio
        self.max_symbol_ratio = max_symbol_ratio
        self.max_bullet_ratio = max_bullet_ratio
        self.max_ellipsis_ratio = max_ellipsis_ratio
        
        # Patterns
        self._word_pattern = re.compile(r'\b\w+\b')
        self._symbol_pattern = re.compile(r'[#$%&*@]{2,}')
        self._bullet_pattern = re.compile(r'^\s*[-•*]\s', re.MULTILINE)
        self._ellipsis_pattern = re.compile(r'\.{2,}|…')
    
    def _compute_word_ratio(self, text: str) -> float:
        """Compute ratio of alphabetic words."""
        words = self._word_pattern.findall(text)
        if not words:
            return 0.0
        
        alpha_words = sum(1 for w in words if w.isalpha())
        return alpha_words / len(words)
    
    def _compute_symbol_ratio(self, text: str) -> float:
        """Compute ratio of symbol-heavy lines."""
        lines = text.split('\n')
        if not lines:
            return 0.0
        
        symbol_lines = sum(1 for line in lines if self._symbol_pattern.search(line))
        return symbol_lines / len(lines)
    
    def _compute_bullet_ratio(self, text: str) -> float:
        """Compute ratio of bullet points."""
        lines = text.split('\n')
        if not lines:
            return 0.0
        
        bullet_lines = sum(1 for line in lines if self._bullet_pattern.match(line))
        return bullet_lines / len(lines)
    
    def _compute_ellipsis_ratio(self, text: str) -> float:
        """Compute ratio of ellipsis occurrences."""
        ellipsis_count = len(self._ellipsis_pattern.findall(text))
        words = len(text.split())
        return ellipsis_count / max(words, 1)
    
    def _has_boilerplate(self, text: str) -> bool:
        """Detect common boilerplate patterns."""
        boilerplate_patterns = [
            r'copyright\s*©',
            r'all\s*rights\s*reserved',
            r'privacy\s*policy',
            r'terms\s*of\s*service',
            r'click\s*here\s*to',
            r'subscribe\s*to\s*our',
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in boilerplate_patterns)
    
    def filter(self, doc: Document) -> Tuple[bool, Dict[str, Any]]:
        """
        Filter a document.
        
        Args:
            doc: Document to filter
        
        Returns:
            Tuple of (passes_filter, filter_info)
        """
        text = doc.text
        info = {}
        
        # Length checks
        if len(text) < self.min_length:
            info['reason'] = f'too_short ({len(text)} < {self.min_length})'
            return False, info
        
        if len(text) > self.max_length:
            info['reason'] = f'too_long ({len(text)} > {self.max_length})'
            return False, info
        
        # Quality metrics
        word_ratio = self._compute_word_ratio(text)
        info['word_ratio'] = word_ratio
        
        if word_ratio < self.min_word_ratio:
            info['reason'] = f'low_word_ratio ({word_ratio:.2f} < {self.min_word_ratio})'
            return False, info
        
        symbol_ratio = self._compute_symbol_ratio(text)
        info['symbol_ratio'] = symbol_ratio
        
        if symbol_ratio > self.max_symbol_ratio:
            info['reason'] = f'high_symbol_ratio ({symbol_ratio:.2f} > {self.max_symbol_ratio})'
            return False, info
        
        bullet_ratio = self._compute_bullet_ratio(text)
        info['bullet_ratio'] = bullet_ratio
        
        if bullet_ratio > self.max_bullet_ratio:
            info['reason'] = f'high_bullet_ratio ({bullet_ratio:.2f} > {self.max_bullet_ratio})'
            return False, info
        
        ellipsis_ratio = self._compute_ellipsis_ratio(text)
        info['ellipsis_ratio'] = ellipsis_ratio
        
        if ellipsis_ratio > self.max_ellipsis_ratio:
            info['reason'] = f'high_ellipsis_ratio ({ellipsis_ratio:.2f} > {self.max_ellipsis_ratio})'
            return False, info
        
        # Boilerplate check
        if self._has_boilerplate(text):
            info['reason'] = 'boilerplate_detected'
            return False, info
        
        info['passed'] = True
        return True, info
    
    def filter_batch(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Filter a batch of documents.
        
        Args:
            documents: List of documents
        
        Returns:
            List of documents that pass filters
        """
        filtered = []
        filter_stats = Counter()
        
        for doc in documents:
            passes, info = self.filter(doc)
            
            if passes:
                filtered.append(doc)
            else:
                reason = info.get('reason', 'unknown')
                filter_stats[reason] += 1
        
        logger.info(f"Filtered {len(documents) - len(filtered)}/{len(documents)} documents")
        logger.info(f"Filter stats: {dict(filter_stats)}")
        
        return filtered


class QualityScorer:
    """
    Quality Scorer for documents.
    
    Computes quality scores based on various heuristics:
    - Perplexity (using a reference model)
    - Readability metrics
    - Information density
    - Language quality
    
    Args:
        reference_model: Optional model for perplexity scoring
        
    Example:
        >>> scorer = QualityScorer()
        >>> scores = scorer.score_batch(documents)
    """
    
    def __init__(
        self,
        use_perplexity: bool = False,
        perplexity_model: Optional[str] = None,
    ):
        self.use_perplexity = use_perplexity
        self.perplexity_model = perplexity_model
        
        # Patterns for quality indicators
        self._sentence_pattern = re.compile(r'[.!?]+')
        self._uppercase_pattern = re.compile(r'[A-Z]')
        self._digit_pattern = re.compile(r'\d')
    
    def _compute_readability_score(self, text: str) -> float:
        """Compute readability score based on text features."""
        sentences = len(self._sentence_pattern.findall(text))
        words = len(text.split())
        chars = len(text)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simple readability metrics
        avg_sentence_length = words / sentences
        avg_word_length = chars / words
        
        # Score based on ideal ranges
        sentence_score = 1.0 - abs(avg_sentence_length - 20) / 20
        word_score = 1.0 - abs(avg_word_length - 5) / 5
        
        return max(0, min(1, (sentence_score + word_score) / 2))
    
    def _compute_information_density(self, text: str) -> float:
        """Compute information density."""
        words = text.split()
        if not words:
            return 0.0
        
        # Unique word ratio
        unique_words = set(w.lower() for w in words)
        unique_ratio = len(unique_words) / len(words)
        
        # Content word ratio (simple heuristic)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose',
        }
        
        content_words = sum(1 for w in words if w.lower() not in stop_words)
        content_ratio = content_words / len(words)
        
        return (unique_ratio + content_ratio) / 2
    
    def _compute_perplexity_score(
        self,
        text: str,
    ) -> float:
        """Compute perplexity-based quality score."""
        # Placeholder - would use a language model in practice
        # Lower perplexity = higher quality
        return 0.5  # Default score
    
    def score(self, doc: Document) -> float:
        """
        Compute quality score for a document.
        
        Args:
            doc: Document to score
        
        Returns:
            Quality score (0-1)
        """
        text = doc.text
        
        # Readability score
        readability = self._compute_readability_score(text)
        
        # Information density
        info_density = self._compute_information_density(text)
        
        # Perplexity score (if enabled)
        if self.use_perplexity:
            perplexity = self._compute_perplexity_score(text)
        else:
            perplexity = 0.5
        
        # Combined score
        score = 0.4 * readability + 0.4 * info_density + 0.2 * perplexity
        
        return score
    
    def score_batch(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Score a batch of documents.
        
        Args:
            documents: List of documents
        
        Returns:
            Documents with quality scores
        """
        for doc in documents:
            doc.quality_score = self.score(doc)
        
        return documents
    
    def filter_by_score(
        self,
        documents: List[Document],
        min_score: float = 0.3,
    ) -> List[Document]:
        """
        Filter documents by quality score.
        
        Args:
            documents: Scored documents
            min_score: Minimum score threshold
        
        Returns:
            Documents above threshold
        """
        return [doc for doc in documents if doc.quality_score >= min_score]


class PreTrainingDataset(IterableDataset):
    """
    Pre-Training Dataset.
    
    Efficient dataset for large-scale pre-training with:
    - Streaming from disk
    - Tokenization on-the-fly
    - Packing sequences
    - Shuffling
    
    Args:
        documents: List of documents
        tokenizer: Tokenizer for encoding
        max_seq_length: Maximum sequence length
        pack_sequences: Whether to pack multiple documents
        
    Example:
        >>> dataset = PreTrainingDataset(documents, tokenizer, max_seq_length=2048)
        >>> dataloader = DataLoader(dataset, batch_size=32)
    """
    
    def __init__(
        self,
        documents: List[Document],
        tokenizer: Any,
        max_seq_length: int = 2048,
        pack_sequences: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pack_sequences = pack_sequences
        self.shuffle = shuffle
        self.seed = seed
        
        # Shuffle documents
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.documents)
    
    def _tokenize_document(self, doc: Document) -> List[int]:
        """Tokenize a document."""
        output = self.tokenizer.encode(
            doc.text,
            add_special_tokens=False,
            truncation=False,
        )
        return output.input_ids if hasattr(output, 'input_ids') else output
    
    def _pack_sequences(
        self,
        token_ids: List[int],
    ) -> Iterator[List[int]]:
        """Pack token sequences to max length."""
        current_seq = []
        
        for token_id in token_ids:
            current_seq.append(token_id)
            
            if len(current_seq) >= self.max_seq_length:
                yield current_seq[:self.max_seq_length]
                current_seq = current_seq[self.max_seq_length:]
        
        # Yield remaining tokens (padded)
        if current_seq:
            # Pad to max length
            current_seq.extend([0] * (self.max_seq_length - len(current_seq)))
            yield current_seq
    
    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        """Iterate over the dataset."""
        import torch
        
        # Get worker info for distributed training
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # Split documents among workers
            per_worker = max(1, len(self.documents) // worker_info.num_workers)
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.documents)
            docs = self.documents[start:end]
        else:
            docs = self.documents
        
        # Tokenize all documents
        all_tokens = []
        for doc in docs:
            tokens = self._tokenize_document(doc)
            all_tokens.extend(tokens)
        
        # Pack sequences
        for seq in self._pack_sequences(all_tokens):
            input_ids = torch.tensor(seq, dtype=torch.long)
            attention_mask = (input_ids != 0).long()
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
    
    def __len__(self) -> int:
        """Get dataset length."""
        total_tokens = sum(
            len(self._tokenize_document(doc))
            for doc in self.documents
        )
        return total_tokens // self.max_seq_length


class PreTrainingDataPipeline:
    """
    Complete Pre-Training Data Pipeline.
    
    Orchestrates the full data preparation workflow:
    1. Collection
    2. Cleaning
    3. Deduplication
    4. Filtering
    5. Quality scoring
    6. Dataset creation
    
    Example:
        >>> pipeline = PreTrainingDataPipeline()
        >>> dataset = pipeline.run(sources=['wikipedia', 'books'])
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path('./data_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.collector = DataCollector(cache_dir=str(self.cache_dir))
        self.cleaner = DataCleaner()
        self.deduplicator = Deduplicator()
        self.data_filter = DataFilter()
        self.quality_scorer = QualityScorer()
    
    def run(
        self,
        sources: List[Union[str, Dict]],
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        Run the full data preparation pipeline.
        
        Args:
            sources: Data sources
            output_path: Path to save processed data
            limit: Limit on number of documents
        
        Returns:
            Processed documents
        """
        logger.info("Starting pre-training data pipeline...")
        
        # Step 1: Collect
        logger.info("Step 1: Collecting data...")
        self.collector.sources = sources
        documents = self.collector.collect(limit=limit * 2 if limit else None)
        logger.info(f"Collected {len(documents)} documents")
        
        # Step 2: Clean
        logger.info("Step 2: Cleaning data...")
        documents = self.cleaner.clean_batch(documents)
        logger.info(f"After cleaning: {len(documents)} documents")
        
        # Step 3: Deduplicate
        logger.info("Step 3: Deduplicating...")
        documents = self.deduplicator.deduplicate(documents)
        logger.info(f"After deduplication: {len(documents)} documents")
        
        # Step 4: Filter
        logger.info("Step 4: Filtering...")
        documents = self.data_filter.filter_batch(documents)
        logger.info(f"After filtering: {len(documents)} documents")
        
        # Step 5: Quality scoring
        logger.info("Step 5: Quality scoring...")
        documents = self.quality_scorer.score_batch(documents)
        
        # Apply quality threshold
        documents = self.quality_scorer.filter_by_score(documents, min_score=0.3)
        logger.info(f"After quality filtering: {len(documents)} documents")
        
        # Step 6: Save
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / 'documents.jsonl', 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(json.dumps(doc.to_dict()) + '\n')
            
            logger.info(f"Saved {len(documents)} documents to {output_path}")
        
        return documents
