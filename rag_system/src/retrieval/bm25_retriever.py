"""
BM25 retrieval for keyword-based search.
"""

import os
import pickle
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass

from ..processing.arabic_processor import ArabicTextProcessor


@dataclass
class BM25Result:
    """BM25 search result."""

    doc_id: str
    score: float
    metadata: Dict[str, Any]


class BM25:
    """
    BM25 (Best Matching 25) retrieval algorithm.
    Implementation for Arabic text retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: Low frequency term weighting parameter
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Index data
        self.doc_count = 0
        self.avg_doc_length = 0
        self.doc_lengths = {}
        self.term_doc_freq = {}  # Document frequency for each term
        self.term_idf = {}  # IDF for each term
        self.doc_term_freq = {}  # Term frequencies per document
        self.documents = {}  # Original documents

        # Text processor
        self.text_processor = ArabicTextProcessor(remove_stopwords=False)

        # Tokenizer
        self._tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):
        """Get tokenizer function."""

        def tokenizer(text: str) -> List[str]:
            # Use Arabic text processor
            return self.text_processor.tokenize(text)

        return tokenizer

    def index(self, documents: List[Dict[str, Any]]):
        """
        Index documents for BM25 retrieval.

        Args:
            documents: List of documents with 'id' and 'content' keys
        """
        print(f"Indexing {len(documents)} documents...")

        self.doc_count = len(documents)
        self.doc_lengths = {}
        self.doc_term_freq = {}
        self.term_doc_freq = defaultdict(int)

        total_length = 0

        for doc in documents:
            doc_id = doc.get("id", doc.get("chunk_id", str(hash(doc["content"]))))
            content = doc["content"]

            # Tokenize
            tokens = self._tokenizer(content)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Term frequencies
            term_freq = Counter(tokens)
            self.doc_term_freq[doc_id] = term_freq

            # Document frequency
            for term in set(tokens):
                self.term_doc_freq[term] += 1

            # Store document
            self.documents[doc_id] = doc

        # Calculate average document length
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 1

        # Calculate IDF for all terms
        self._calculate_idf()

        print(f"Indexed {self.doc_count} documents")
        print(f"Average document length: {self.avg_doc_length:.1f} tokens")
        print(f"Vocabulary size: {len(self.term_idf)} unique terms")

    def _calculate_idf(self):
        """Calculate IDF values for all terms."""
        for term, doc_freq in self.term_doc_freq.items():
            # BM25 IDF formula
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            self.term_idf[term] = idf

    def get_scores(self, query: str) -> Dict[str, float]:
        """
        Calculate BM25 scores for a query against all documents.

        Args:
            query: Query string

        Returns:
            Dictionary mapping document IDs to BM25 scores
        """
        query_tokens = self._tokenizer(query)

        if not query_tokens:
            return {}

        scores = {}

        for doc_id, term_freq in self.doc_term_freq.items():
            doc_length = self.doc_lengths[doc_id]

            score = 0.0

            for term in query_tokens:
                if term not in self.term_idf:
                    continue

                tf = term_freq.get(term, 0)

                if tf == 0:
                    continue

                idf = self.term_idf[term]

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )

                score += idf * (numerator / denominator)

            scores[doc_id] = score

        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BM25Result]:
        """
        Search for documents matching a query.

        Args:
            query: Query string
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of BM25 results sorted by score
        """
        scores = self.get_scores(query)

        # Apply filters
        if filters:
            filtered_scores = {}
            for doc_id, score in scores.items():
                doc = self.documents.get(doc_id, {})
                metadata = doc.get("metadata", {})

                if all(metadata.get(k) == v for k, v in filters.items()):
                    filtered_scores[doc_id] = score

            scores = filtered_scores

        # Sort by score descending
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self.documents.get(doc_id, {})

            # Extract metadata
            metadata = {
                "content": doc.get("content", ""),
                "book_id": doc.get("book_id"),
                "book_title": doc.get("book_title", ""),
                "author": doc.get("author", ""),
                "category": doc.get("category", ""),
                "chunk_id": doc.get("chunk_id", doc_id),
            }

            results.append(BM25Result(doc_id=doc_id, score=score, metadata=metadata))

        return results

    def save(self, filepath: str):
        """Save BM25 index to file."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "epsilon": self.epsilon,
                    "doc_count": self.doc_count,
                    "avg_doc_length": self.avg_doc_length,
                    "doc_lengths": self.doc_lengths,
                    "term_doc_freq": dict(self.term_doc_freq),
                    "term_idf": self.term_idf,
                    "doc_term_freq": self.doc_term_freq,
                    "documents": self.documents,
                },
                f,
            )
        print(f"Saved BM25 index to {filepath}")

    def load(self, filepath: str):
        """Load BM25 index from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.k1 = data["k1"]
        self.b = data["b"]
        self.epsilon = data["epsilon"]
        self.doc_count = data["doc_count"]
        self.avg_doc_length = data["avg_doc_length"]
        self.doc_lengths = data["doc_lengths"]
        self.term_doc_freq = data["term_doc_freq"]
        self.term_idf = data["term_idf"]
        self.doc_term_freq = data["doc_term_freq"]
        self.documents = data["documents"]

        print(f"Loaded BM25 index from {filepath}")
        print(f"Documents: {self.doc_count}, Vocabulary: {len(self.term_idf)}")

    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        return len(self.term_idf)

    def get_document_count(self) -> int:
        """Get number of indexed documents."""
        return self.doc_count


class BM25Retriever:
    """High-level BM25 retriever with caching and management."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        index_path: Optional[str] = None,
    ):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
            index_path: Path to save/load index
        """
        self.bm25 = BM25(k1=k1, b=b)
        self.index_path = index_path

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for retrieval."""
        self.bm25.index(documents)

        # Save if path provided
        if self.index_path:
            self.bm25.save(self.index_path)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents.

        Args:
            query: Query string
            top_k: Number of results
            filters: Optional filters

        Returns:
            List of search results
        """
        results = self.bm25.search(query, top_k=top_k, filters=filters)

        return [
            {
                "doc_id": r.doc_id,
                "score": r.score,
                "book_id": r.metadata.get("book_id"),
                "book_title": r.metadata.get("book_title"),
                "author": r.metadata.get("author"),
                "category": r.metadata.get("category"),
                "content": r.metadata.get("content", ""),
                "chunk_id": r.metadata.get("chunk_id"),
            }
            for r in results
        ]

    def load_index(self, filepath: str):
        """Load index from file."""
        self.bm25.load(filepath)

    def save_index(self, filepath: str):
        """Save index to file."""
        self.bm25.save(filepath)


def create_bm25_retriever(
    k1: float = 1.5,
    b: float = 0.75,
    index_path: Optional[str] = None,
) -> BM25Retriever:
    """
    Factory function to create a BM25 retriever.

    Args:
        k1: BM25 k1 parameter
        b: BM25 b parameter
        index_path: Path for index storage

    Returns:
        BM25Retriever instance
    """
    return BM25Retriever(k1=k1, b=b, index_path=index_path)
