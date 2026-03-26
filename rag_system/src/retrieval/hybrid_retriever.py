"""
Hybrid Retrieval - Combining Vector Search + BM25 + Reranking

Following RAG Pipeline Guide 2026 - Phase 5: Retrieval & Reranking
"""

import os
import json
import pickle
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from the retrieval system."""

    chunk_id: str
    content: str
    score: float
    source: str  # 'bm25', 'semantic', or 'hybrid'
    metadata: Dict[str, Any]


class BM25Index:
    """
    BM25 (Best Matching 25) retrieval algorithm implementation.
    Optimized for Arabic text retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Index data structures
        self.doc_count = 0
        self.avg_doc_length = 0
        self.doc_lengths: Dict[str, int] = {}
        self.term_doc_freq: Dict[str, int] = {}
        self.term_idf: Dict[str, float] = {}
        self.doc_term_freq: Dict[str, Dict[str, int]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}

        # Arabic tokenizer
        self._tokenizer = self._arabic_tokenizer()

    def _arabic_tokenizer(self):
        """Create Arabic tokenizer."""
        import re

        def tokenize(text: str) -> List[str]:
            if not text:
                return []

            # Normalize Arabic text
            # Remove diacritics
            text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
            # Normalize alef variants
            text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            # Normalize teh marbuta
            text = text.replace("ة", "ه")
            # Normalize yeh
            text = text.replace("ى", "ي")
            # Remove tatweel
            text = re.sub(r"ـ+", "", text)

            # Split on non-Arabic and non-word characters
            tokens = re.findall(r"[\u0600-\u06FF]+", text)

            # Lowercase for consistency
            tokens = [t.lower() for t in tokens]

            return tokens

        return tokenize

    def index(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents with 'id' and 'content' keys
        """

        logger.info(f"Building BM25 index for {len(documents)} documents...")

        self.doc_count = len(documents)

        total_length = 0

        for doc in documents:
            doc_id = doc.get(
                "id", doc.get("chunk_id", str(hash(doc.get("content", ""))))
            )
            content = doc.get("content", "")

            # Tokenize
            tokens = self._tokenizer(content)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Term frequencies
            term_freq = Counter(tokens)
            self.doc_term_freq[doc_id] = term_freq

            # Document frequency
            for term in set(tokens):
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1

            # Store document
            self.documents[doc_id] = doc

        # Calculate average document length
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 1

        # Calculate IDF for all terms
        self._calculate_idf()

        logger.info(
            f"BM25 index built: {self.doc_count} documents, "
            f"{len(self.term_idf)} unique terms"
        )

    def _calculate_idf(self):
        """Calculate IDF values for all terms."""

        for term, doc_freq in self.term_doc_freq.items():
            # Standard BM25 IDF formula
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            self.term_idf[term] = idf

    def get_scores(self, query: str) -> Dict[str, float]:
        """Calculate BM25 scores for a query."""

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
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Search for documents matching a query."""

        scores = self.get_scores(query)

        # Apply filters (if provided)
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

        # Normalize scores to 0-1 range
        max_score = sorted_docs[0][1] if sorted_docs else 1.0

        # Return top k results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            doc = self.documents.get(doc_id, {})

            normalized_score = score / max_score if max_score > 0 else 0

            results.append(
                RetrievalResult(
                    chunk_id=doc_id,
                    content=doc.get("content", ""),
                    score=normalized_score,
                    source="bm25",
                    metadata={
                        "book_id": doc.get("book_id"),
                        "book_title": doc.get("book_title", ""),
                        "author": doc.get("author", ""),
                        "category": doc.get("category", ""),
                    },
                )
            )

        return results

    def save(self, filepath: str):
        """Save index to file."""

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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

        logger.info(f"BM25 index saved to {filepath}")

    def load(self, filepath: str):
        """Load index from file."""

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

        logger.info(f"BM25 index loaded: {self.doc_count} documents")


class HybridRetriever:
    """
    Hybrid retrieval combining vector search + BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        vector_store: Any = None,
        embedding_model: Any = None,
        bm25: Optional[BM25Index] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25 = bm25

        # Default weights: favor semantic search
        self.weights = weights or {
            "semantic": 0.6,
            "bm25": 0.4,
        }

    async def search(
        self,
        query: str,
        top_k: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            top_k: Number of initial candidates to retrieve
            filters: Optional metadata filters

        Returns:
            List of retrieval results sorted by combined score
        """

        # Step 1: Vector search (semantic)
        semantic_results = []
        if self.vector_store and self.embedding_model:
            semantic_results = await self._semantic_search(query, top_k, filters)

        # Step 2: BM25 keyword search
        bm25_results = []
        if self.bm25:
            bm25_results = self.bm25.search(query, top_k, filters)

        # Step 3: Combine with Reciprocal Rank Fusion
        combined = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results,
            top_k=top_k,
        )

        return combined

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Perform vector search."""

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        # Convert to RetrievalResult
        retrieval_results = []
        for r in results:
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=r.get("id", ""),
                    content=r.get("payload", {}).get("content", ""),
                    score=r.get("score", 0),
                    source="semantic",
                    metadata=r.get("payload", {}),
                )
            )

        return retrieval_results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        Formula: RRF_score = Σ (weight / (rank + k))

        Args:
            semantic_results: Results from vector search
            bm25_results: Results from BM25
            top_k: Number of results to return
            k: RRF constant (higher = more weight to lower ranks)

        Returns:
            Combined and ranked results
        """

        scores = defaultdict(float)
        doc_map = {}

        # Score from semantic results
        for rank, result in enumerate(semantic_results):
            scores[result.chunk_id] += self.weights.get("semantic", 0.6) / (rank + k)
            if result.chunk_id not in doc_map:
                doc_map[result.chunk_id] = result

        # Score from BM25 results
        for rank, result in enumerate(bm25_results):
            scores[result.chunk_id] += self.weights.get("bm25", 0.4) / (rank + k)
            if result.chunk_id not in doc_map:
                doc_map[result.chunk_id] = result

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        final_results = []
        for chunk_id, score in sorted_results[:top_k]:
            result = doc_map[chunk_id]
            final_results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=score,
                    source="hybrid",
                    metadata=result.metadata,
                )
            )

        return final_results


class Reranker:
    """
    Cross-encoder reranker for improved retrieval quality.

    Uses a cross-encoder model to re-score retrieved candidates.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded reranker model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed, using score-based reranking"
                )
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Rerank candidates by relevance to query.

        Args:
            query: Original query
            candidates: List of candidate results
            top_k: Number of results to return

        Returns:
            Reranked results
        """

        if not candidates:
            return []

        if self.model is None:
            # Fallback: return candidates sorted by original score
            return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]

        # Prepare pairs for cross-encoder
        pairs = [[query, candidate.content] for candidate in candidates]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Update candidate scores and sort
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)

        return reranked[:top_k]

    def rerank_with_cohere(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Alternative: Use Cohere's Rerank API."""

        try:
            import cohere

            client = cohere.Client()

            documents = [c.content for c in candidates]

            response = client.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=documents,
                top_n=top_k,
            )

            reranked = []
            for result in response.results:
                candidate = candidates[result.index]
                candidate.score = result.relevance_score
                reranked.append(candidate)

            return reranked

        except ImportError:
            logger.warning("cohere not installed, using local reranker")
            return self.rerank(query, candidates, top_k)


class AdaptiveRetriever:
    """
    Advanced retriever that adapts strategy based on query type.

    Features:
    - Query classification (factual, how-to, comparison, etc.)
    - Strategy selection based on query type
    - Multi-stage retrieval
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def _classify_query(self, query: str) -> str:
        """Classify query intent."""

        query_lower = query.lower()

        # Factual questions
        factual_words = [
            "ما",
            "من",
            "متى",
            "أين",
            "كم",
            "who",
            "what",
            "when",
            "where",
            "how many",
        ]
        if any(word in query_lower for word in factual_words):
            return "factual"

        # How-to questions
        howto_words = ["كيف", "طريقة", "كيفية", "how to", "how do"]
        if any(word in query_lower for word in howto_words):
            return "howto"

        # Comparison questions
        compare_words = ["versus", "vs", "مقارنة", "between", "افضل", "أفضل"]
        if any(word in query_lower for word in compare_words):
            return "comparison"

        # Definition questions
        defin_words = ["تعريف", "meaning", "definition", "ما هو", "ما هي"]
        if any(word in query_lower for word in defin_words):
            return "definition"

        # Default to exploratory
        return "exploratory"

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_top_k: int = 50,
    ) -> List[RetrievalResult]:
        """
        Retrieve with adaptive strategy.

        Args:
            query: User query
            top_k: Final number of results
            retrieval_top_k: Number of candidates to retrieve

        Returns:
            Final reranked results
        """

        # Classify query
        query_type = self._classify_query(query)

        # Adjust parameters based on query type
        if query_type == "factual":
            # Factual: more weight on exact matches
            self.hybrid_retriever.weights = {"semantic": 0.3, "bm25": 0.7}
        elif query_type == "howto":
            # How-to: balance semantic and keyword
            self.hybrid_retriever.weights = {"semantic": 0.5, "bm25": 0.5}
        elif query_type == "comparison":
            # Comparison: favor semantic for broader context
            self.hybrid_retriever.weights = {"semantic": 0.7, "bm25": 0.3}
        else:
            # Default balanced
            self.hybrid_retriever.weights = {"semantic": 0.6, "bm25": 0.4}

        # Stage 1: Retrieve candidates
        candidates = await self.hybrid_retriever.search(
            query=query,
            top_k=retrieval_top_k,
        )

        # Stage 2: Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return reranked
