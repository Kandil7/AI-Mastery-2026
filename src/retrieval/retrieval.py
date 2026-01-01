"""
Retrieval Module - AI-Mastery-2026

This module implements various retrieval strategies for document search,
including sparse (BM25), dense (vector), and hybrid approaches.

Key Components:
- BM25Retriever: Classic sparse retrieval using BM25 algorithm
- DenseRetriever: Vector-based retrieval using embeddings
- ColBERTRetriever: Late interaction retrieval for better accuracy
- HybridRetriever: Combines sparse and dense methods
- RetrievalPipeline: Orchestrates multiple retrievers

Author: AI-Mastery-2026
License: MIT
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
import math
import re
import logging
from heapq import nlargest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Document:
    """
    Represents a document in the retrieval system.
    
    Attributes:
        id: Unique identifier for the document
        content: Text content of the document
        metadata: Additional metadata (source, date, etc.)
        embedding: Optional precomputed embedding vector
    
    Example:
        >>> doc = Document(
        ...     id="doc_001",
        ...     content="Machine learning is a subset of AI.",
        ...     metadata={"source": "wikipedia"}
        ... )
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(hash(self.content))[:8]


@dataclass
class RetrievalResult:
    """
    Represents a retrieval result with score and metadata.
    
    Attributes:
        document: The retrieved document
        score: Relevance score (higher is better)
        rank: Position in the result list (1-indexed)
        retriever: Name of the retriever that found this result
    """
    document: Document
    score: float
    rank: int = 0
    retriever: str = ""


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval systems.
    
    Attributes:
        top_k: Number of results to return
        min_score: Minimum score threshold
        use_reranking: Whether to apply reranking
        hybrid_alpha: Weight for dense retriever in hybrid search (0-1)
    """
    top_k: int = 10
    min_score: float = 0.0
    use_reranking: bool = False
    hybrid_alpha: float = 0.5
    

# =============================================================================
# Text Preprocessing
# =============================================================================

class TextPreprocessor:
    """
    Handles text preprocessing for retrieval.
    
    Includes tokenization, lowercasing, stopword removal, and stemming.
    
    Example:
        >>> preprocessor = TextPreprocessor()
        >>> tokens = preprocessor.tokenize("The quick brown fox!")
        >>> print(tokens)  # ['quick', 'brown', 'fox']
    """
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
    }
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        min_token_length: int = 2
    ):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_stopwords: Remove common stopwords
            min_token_length: Minimum token length to keep
        """
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Extract words (alphanumeric sequences)
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter tokens
        filtered = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue
            # Skip stopwords
            if self.remove_stopwords and token in self.STOPWORDS:
                continue
            # Skip pure numbers
            if token.isdigit():
                continue
            filtered.append(token)
        
        return filtered


# =============================================================================
# Base Retriever
# =============================================================================

class BaseRetriever(ABC):
    """
    Abstract base class for all retrievers.
    
    All retriever implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.documents: List[Document] = []
        self.is_indexed = False
    
    @abstractmethod
    def index(self, documents: List[Document]) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of documents to index
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        pass
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents and reindex."""
        self.documents.extend(documents)
        self.index(self.documents)


# =============================================================================
# BM25 Retriever (Sparse)
# =============================================================================

class BM25Retriever(BaseRetriever):
    """
    BM25 (Best Matching 25) retriever - Classic sparse retrieval.
    
    BM25 is a bag-of-words retrieval function that ranks documents
    based on query term frequency and document length.
    
    The BM25 scoring formula:
    
    score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1 + 1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl))
    
    Where:
        - f(qi,D): Term frequency of qi in document D
        - |D|: Length of document D
        - avgdl: Average document length
        - k1, b: Tuning parameters
    
    Attributes:
        k1: Term frequency saturation parameter (default: 1.5)
        b: Length normalization parameter (default: 0.75)
    
    Example:
        >>> retriever = BM25Retriever(k1=1.5, b=0.75)
        >>> retriever.index(documents)
        >>> results = retriever.retrieve("machine learning")
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation (higher = more weight to term freq)
            b: Length normalization (0 = no normalization, 1 = full normalization)
            preprocessor: Text preprocessor for tokenization
        """
        super().__init__(name="bm25")
        self.k1 = k1
        self.b = b
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Index data structures
        self.doc_freqs: Dict[str, int] = {}  # Document frequency per term
        self.doc_lengths: List[int] = []      # Length of each document
        self.avg_doc_length: float = 0.0      # Average document length
        self.doc_term_freqs: List[Dict[str, int]] = []  # Term freq per doc
        self.idf: Dict[str, float] = {}       # Inverse document frequency
        
        logger.info(f"Initialized BM25Retriever with k1={k1}, b={b}")
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents using BM25.
        
        Computes term frequencies, document frequencies, and IDF values.
        
        Args:
            documents: List of documents to index
            
        Time Complexity: O(N * L) where N is number of docs, L is avg length
        """
        self.documents = documents
        self.doc_freqs = {}
        self.doc_lengths = []
        self.doc_term_freqs = []
        
        # First pass: compute term frequencies and document lengths
        for doc in documents:
            tokens = self.preprocessor.tokenize(doc.content)
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this document
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(dict(term_freqs))
            
            # Update document frequencies
            for term in term_freqs:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        # Compute average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if documents else 0
        
        # Compute IDF for each term
        n_docs = len(documents)
        for term, df in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5))
            self.idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        
        self.is_indexed = True
        logger.info(f"Indexed {len(documents)} documents with {len(self.doc_freqs)} unique terms")
    
    def _compute_score(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a query-document pair.
        
        Args:
            query_terms: Tokenized query
            doc_idx: Index of the document
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        for term in query_terms:
            if term not in self.idf:
                continue
            
            # Get term frequency in document
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            
            # BM25 scoring formula
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve documents matching the query using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects sorted by score
            
        Example:
            >>> results = retriever.retrieve("neural networks", top_k=5)
            >>> for r in results:
            ...     print(f"{r.score:.3f}: {r.document.content[:50]}")
        """
        if not self.is_indexed:
            raise ValueError("Retriever not indexed. Call index() first.")
        
        # Tokenize query
        query_terms = self.preprocessor.tokenize(query)
        if not query_terms:
            return []
        
        # Score all documents
        scored_docs = []
        for idx, doc in enumerate(self.documents):
            score = self._compute_score(query_terms, idx)
            if score > 0:
                scored_docs.append((score, idx))
        
        # Get top-k results
        top_results = nlargest(top_k, scored_docs, key=lambda x: x[0])
        
        # Create RetrievalResult objects
        results = []
        for rank, (score, idx) in enumerate(top_results, 1):
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=score,
                rank=rank,
                retriever=self.name
            ))
        
        return results
    
    def explain_score(self, query: str, doc_idx: int) -> Dict[str, float]:
        """
        Explain the BM25 score breakdown for a query-document pair.
        
        Args:
            query: Search query
            doc_idx: Document index
            
        Returns:
            Dictionary with per-term score contributions
        """
        query_terms = self.preprocessor.tokenize(query)
        contributions = {}
        
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        for term in query_terms:
            if term not in self.idf:
                contributions[term] = 0.0
                continue
            
            tf = term_freqs.get(term, 0)
            if tf == 0:
                contributions[term] = 0.0
                continue
            
            idf = self.idf[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            
            contributions[term] = idf * (numerator / denominator)
        
        return contributions


# =============================================================================
# Dense Retriever (Vector)
# =============================================================================

class DenseRetriever(BaseRetriever):
    """
    Dense retriever using embedding vectors.
    
    Uses pre-trained language models to encode documents and queries
    into dense vectors, then retrieves by vector similarity.
    
    Supports:
        - Sentence transformers (all-MiniLM-L6-v2, etc.)
        - FAISS index for efficient search
        - Cosine and dot product similarity
    
    Example:
        >>> retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
        >>> retriever.index(documents)
        >>> results = retriever.retrieve("machine learning", top_k=5)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True,
        similarity: str = "cosine"
    ):
        """
        Initialize dense retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            use_faiss: Whether to use FAISS for efficient search
            similarity: Similarity metric ('cosine' or 'dot')
        """
        super().__init__(name="dense")
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.similarity = similarity
        
        self._encoder = None
        self._index = None
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f"Initialized DenseRetriever with model: {model_name}")
    
    def _load_encoder(self):
        """Lazy load the sentence transformer model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, using TF-IDF fallback")
                self._encoder = "fallback"
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the model."""
        self._load_encoder()
        
        if self._encoder == "fallback":
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=384)
            return vectorizer.fit_transform(texts).toarray().astype(np.float32)
        
        return self._encoder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=(self.similarity == "cosine"),
            show_progress_bar=False
        ).astype(np.float32)
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents by computing their embeddings.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        
        # Encode all documents
        texts = [doc.content for doc in documents]
        self.embeddings = self._encode_texts(texts)
        
        # Build FAISS index if enabled
        if self.use_faiss:
            try:
                import faiss
                dim = self.embeddings.shape[1]
                
                if self.similarity == "cosine":
                    # Use inner product for normalized vectors
                    self._index = faiss.IndexFlatIP(dim)
                else:
                    # Use L2 distance
                    self._index = faiss.IndexFlatL2(dim)
                
                self._index.add(self.embeddings)
                logger.info(f"Built FAISS index with {len(documents)} vectors")
            except ImportError:
                logger.warning("FAISS not installed, using brute-force search")
                self.use_faiss = False
        
        self.is_indexed = True
        logger.info(f"Indexed {len(documents)} documents with dense embeddings")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Retriever not indexed. Call index() first.")
        
        # Encode query
        query_embedding = self._encode_texts([query])[0]
        
        if self.use_faiss and self._index is not None:
            # Use FAISS for efficient search
            scores, indices = self._index.search(
                query_embedding.reshape(1, -1),
                min(top_k, len(self.documents))
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Brute-force search
            if self.similarity == "cosine":
                scores = np.dot(self.embeddings, query_embedding)
            else:
                # Negative L2 distance (higher is better)
                scores = -np.linalg.norm(self.embeddings - query_embedding, axis=1)
            
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        # Create results
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            if idx >= 0:  # FAISS may return -1 for empty results
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank,
                    retriever=self.name
                ))
        
        return results


# =============================================================================
# ColBERT Retriever (Late Interaction)
# =============================================================================

class ColBERTRetriever(BaseRetriever):
    """
    ColBERT-style late interaction retriever.
    
    ColBERT computes token-level embeddings for both queries and documents,
    then uses late interaction (MaxSim) for scoring.
    
    Benefits:
        - More accurate than single-vector retrieval
        - Captures fine-grained semantic matches
        - Better handling of multi-aspect queries
    
    Example:
        >>> retriever = ColBERTRetriever()
        >>> retriever.index(documents)
        >>> results = retriever.retrieve("machine learning applications")
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ColBERT retriever.
        
        Args:
            model_name: Name of the transformer model for encoding
        """
        super().__init__(name="colbert")
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        
        # Store token embeddings for each document
        self.doc_token_embeddings: List[np.ndarray] = []
        
        logger.info(f"Initialized ColBERTRetriever with model: {model_name}")
    
    def _load_model(self):
        """Load the transformer model for token embeddings."""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    f"sentence-transformers/{self.model_name}"
                )
                self._model = AutoModel.from_pretrained(
                    f"sentence-transformers/{self.model_name}"
                )
                self._model.eval()
            except ImportError:
                logger.warning("transformers not installed, using fallback")
                self._model = "fallback"
    
    def _get_token_embeddings(self, text: str) -> np.ndarray:
        """
        Get token-level embeddings for text.
        
        Returns:
            Array of shape (num_tokens, embedding_dim)
        """
        import torch
        
        self._load_model()
        
        if self._model == "fallback":
            # Simple fallback using random embeddings
            words = text.split()[:20]  # Limit tokens
            return np.random.randn(len(words), 384).astype(np.float32)
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state[0].numpy()
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings.astype(np.float32)
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents with token-level embeddings.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.doc_token_embeddings = []
        
        for doc in documents:
            token_embs = self._get_token_embeddings(doc.content)
            self.doc_token_embeddings.append(token_embs)
        
        self.is_indexed = True
        logger.info(f"Indexed {len(documents)} documents with token embeddings")
    
    def _maxsim_score(
        self,
        query_embs: np.ndarray,
        doc_embs: np.ndarray
    ) -> float:
        """
        Compute MaxSim score between query and document.
        
        For each query token, find the maximum similarity to any document token,
        then sum these maximum similarities.
        
        Args:
            query_embs: Query token embeddings (Q, dim)
            doc_embs: Document token embeddings (D, dim)
            
        Returns:
            MaxSim score
        """
        # Compute similarity matrix (Q x D)
        similarity_matrix = np.dot(query_embs, doc_embs.T)
        
        # Take max over document tokens for each query token
        max_sims = np.max(similarity_matrix, axis=1)
        
        # Sum the max similarities
        return float(np.sum(max_sims))
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using late interaction (MaxSim).
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Retriever not indexed. Call index() first.")
        
        # Get query token embeddings
        query_embs = self._get_token_embeddings(query)
        
        # Score all documents
        scored_docs = []
        for idx, doc_embs in enumerate(self.doc_token_embeddings):
            score = self._maxsim_score(query_embs, doc_embs)
            scored_docs.append((score, idx))
        
        # Get top-k
        top_results = nlargest(top_k, scored_docs, key=lambda x: x[0])
        
        # Create results
        results = []
        for rank, (score, idx) in enumerate(top_results, 1):
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=score,
                rank=rank,
                retriever=self.name
            ))
        
        return results


# =============================================================================
# Hybrid Retriever
# =============================================================================

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining sparse and dense methods.
    
    Combines BM25 (keyword matching) with dense retrieval (semantic matching)
    for better coverage and accuracy.
    
    Fusion Methods:
        - 'weighted': Weighted combination of normalized scores
        - 'rrf': Reciprocal Rank Fusion
    
    Example:
        >>> retriever = HybridRetriever(alpha=0.5, fusion='rrf')
        >>> retriever.index(documents)
        >>> results = retriever.retrieve("neural networks", top_k=10)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        fusion: str = "rrf",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        dense_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            alpha: Weight for dense scores (1-alpha for sparse)
            fusion: Fusion method ('weighted' or 'rrf')
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            dense_model: Model for dense retrieval
        """
        super().__init__(name="hybrid")
        self.alpha = alpha
        self.fusion = fusion
        
        # Initialize sub-retrievers
        self.sparse_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.dense_retriever = DenseRetriever(model_name=dense_model)
        
        logger.info(f"Initialized HybridRetriever with alpha={alpha}, fusion={fusion}")
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents in both retrievers.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        
        # Index in both retrievers
        self.sparse_retriever.index(documents)
        self.dense_retriever.index(documents)
        
        self.is_indexed = True
        logger.info(f"Indexed {len(documents)} documents in hybrid retriever")
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return {}
        
        scores = {r.document.id: r.score for r in results}
        
        min_score = min(scores.values())
        max_score = max(scores.values())
        
        if max_score == min_score:
            return {doc_id: 1.0 for doc_id in scores}
        
        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }
    
    def _reciprocal_rank_fusion(
        self,
        sparse_results: List[RetrievalResult],
        dense_results: List[RetrievalResult],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Apply Reciprocal Rank Fusion (RRF) to combine results.
        
        RRF score = Σ 1 / (k + rank)
        
        Args:
            sparse_results: Results from sparse retriever
            dense_results: Results from dense retriever
            k: RRF constant (default 60)
            
        Returns:
            List of (doc_id, fused_score) tuples
        """
        scores = {}
        
        # Add sparse scores
        for result in sparse_results:
            doc_id = result.document.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + result.rank)
        
        # Add dense scores
        for result in dense_results:
            doc_id = result.document.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + result.rank)
        
        # Sort by fused score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed:
            raise ValueError("Retriever not indexed. Call index() first.")
        
        # Get more results from each retriever for better fusion
        k_per_retriever = top_k * 2
        
        sparse_results = self.sparse_retriever.retrieve(query, k_per_retriever)
        dense_results = self.dense_retriever.retrieve(query, k_per_retriever)
        
        if self.fusion == "rrf":
            # Reciprocal Rank Fusion
            fused = self._reciprocal_rank_fusion(sparse_results, dense_results)
        else:
            # Weighted fusion
            sparse_scores = self._normalize_scores(sparse_results)
            dense_scores = self._normalize_scores(dense_results)
            
            # Combine scores
            all_doc_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
            fused = []
            for doc_id in all_doc_ids:
                sparse = sparse_scores.get(doc_id, 0)
                dense = dense_scores.get(doc_id, 0)
                combined = (1 - self.alpha) * sparse + self.alpha * dense
                fused.append((doc_id, combined))
            
            fused = sorted(fused, key=lambda x: x[1], reverse=True)
        
        # Create results
        doc_map = {doc.id: doc for doc in self.documents}
        results = []
        for rank, (doc_id, score) in enumerate(fused[:top_k], 1):
            results.append(RetrievalResult(
                document=doc_map[doc_id],
                score=score,
                rank=rank,
                retriever=self.name
            ))
        
        return results


# =============================================================================
# Retrieval Pipeline
# =============================================================================

class RetrievalPipeline:
    """
    Orchestrates multiple retrievers for complex retrieval scenarios.
    
    Supports:
        - Sequential retrieval (one retriever feeds into another)
        - Ensemble retrieval (combine multiple retrievers)
        - Filtering and post-processing
    
    Example:
        >>> pipeline = RetrievalPipeline()
        >>> pipeline.add_retriever(bm25_retriever, stage="initial")
        >>> pipeline.add_retriever(dense_retriever, stage="rerank")
        >>> results = pipeline.retrieve("machine learning", top_k=10)
    """
    
    def __init__(self):
        """Initialize the retrieval pipeline."""
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.stages: List[str] = []
        self.config = RetrievalConfig()
        
    def add_retriever(
        self,
        retriever: BaseRetriever,
        stage: str = "default"
    ) -> None:
        """
        Add a retriever to the pipeline.
        
        Args:
            retriever: The retriever to add
            stage: Pipeline stage name
        """
        self.retrievers[stage] = retriever
        if stage not in self.stages:
            self.stages.append(stage)
    
    def index(self, documents: List[Document]) -> None:
        """
        Index documents in all retrievers.
        
        Args:
            documents: List of documents to index
        """
        for retriever in self.retrievers.values():
            retriever.index(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        config: Optional[RetrievalConfig] = None
    ) -> List[RetrievalResult]:
        """
        Run the retrieval pipeline.
        
        Args:
            query: Search query
            top_k: Number of results to return
            config: Optional configuration override
            
        Returns:
            List of RetrievalResult objects
        """
        config = config or self.config
        
        results = []
        for stage in self.stages:
            retriever = self.retrievers[stage]
            stage_results = retriever.retrieve(query, top_k)
            
            # For simplicity, we use the last stage's results
            # More sophisticated pipelines could combine results
            results = stage_results
        
        # Apply minimum score filter
        results = [r for r in results if r.score >= config.min_score]
        
        return results[:top_k]


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Retrieval Module Demo")
    print("=" * 60)
    
    # Sample documents
    docs = [
        Document(id="1", content="Machine learning is a subset of artificial intelligence."),
        Document(id="2", content="Deep learning uses neural networks with many layers."),
        Document(id="3", content="Natural language processing helps computers understand text."),
        Document(id="4", content="Computer vision enables machines to interpret images."),
        Document(id="5", content="Reinforcement learning trains agents through rewards."),
    ]
    
    # Test BM25
    print("\n1. BM25 Retrieval")
    print("-" * 40)
    
    bm25 = BM25Retriever()
    bm25.index(docs)
    
    results = bm25.retrieve("neural networks learning", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.content}")
    
    # Test Dense
    print("\n2. Dense Retrieval")
    print("-" * 40)
    
    dense = DenseRetriever()
    dense.index(docs)
    
    results = dense.retrieve("understanding text and language", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.content}")
    
    # Test Hybrid
    print("\n3. Hybrid Retrieval")
    print("-" * 40)
    
    hybrid = HybridRetriever(alpha=0.5, fusion="rrf")
    hybrid.index(docs)
    
    results = hybrid.retrieve("AI and neural networks", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.document.content}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
