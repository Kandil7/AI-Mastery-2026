"""
Legal Document RAG - Query Engine
=================================
Hybrid search and generation pipeline for legal document QA.

Features:
- Dense + Sparse hybrid search
- Cross-encoder reranking
- Citation-aware response generation
- Confidence scoring

Author: AI-Mastery-2026
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SearchResult:
    """A single search result with scoring information."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Complete query result with answer and sources."""
    query: str
    answer: str
    sources: List[SearchResult]
    confidence: float
    citations: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ============================================================
# RETRIEVAL COMPONENTS
# ============================================================

class Retriever(ABC):
    """Base class for retrieval strategies."""
    
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search for relevant documents."""
        pass


class DenseRetriever(Retriever):
    """
    Dense embedding-based retrieval using vector similarity.
    
    Uses sentence transformers for semantic similarity.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        vector_index: Any = None  # FAISS index
    ):
        self.model_name = embedding_model
        self.vector_index = vector_index
        self._encoder = None
    
    def _get_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers required")
        return self._encoder
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        encoder = self._get_encoder()
        return encoder.encode(texts).tolist()
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using dense embeddings."""
        if self.vector_index is None:
            raise ValueError("Vector index not initialized")
        
        query_embedding = self.encode([query])[0]
        
        # Assuming FAISS-like interface
        distances, indices = self.vector_index.search(
            [query_embedding], k
        )
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # Valid index
                results.append(SearchResult(
                    chunk_id=str(idx),
                    document_id="",  # To be filled from metadata
                    content="",  # To be filled from metadata
                    score=float(1 / (1 + dist)),  # Convert distance to similarity
                    dense_score=float(1 / (1 + dist)),
                ))
        
        return results


class SparseRetriever(Retriever):
    """
    Sparse BM25-based retrieval for keyword matching.
    
    Important for legal terms that must match exactly.
    """
    
    def __init__(self, corpus: Optional[List[str]] = None):
        self.corpus = corpus or []
        self._index = None
        self._tokenized_corpus = None
    
    def _build_index(self):
        """Build BM25 index from corpus."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 required: pip install rank-bm25")
        
        # Simple tokenization
        self._tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self._index = BM25Okapi(self._tokenized_corpus)
    
    def index(self, corpus: List[str]):
        """Index a corpus of documents."""
        self.corpus = corpus
        self._build_index()
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search using BM25."""
        if self._index is None:
            self._build_index()
        
        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)
        
        # Get top k
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    chunk_id=str(idx),
                    document_id="",
                    content=self.corpus[idx] if idx < len(self.corpus) else "",
                    score=float(scores[idx]),
                    sparse_score=float(scores[idx]),
                ))
        
        return results


class HybridRetriever(Retriever):
    """
    Hybrid retrieval combining dense and sparse methods.
    
    Fusion Methods:
        - RRF (Reciprocal Rank Fusion): Rank-based combination
        - Weighted: Score-based combination with tunable weight
    """
    
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        alpha: float = 0.5,  # Weight for dense (1-alpha for sparse)
        fusion_method: str = "rrf"  # "rrf" or "weighted"
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha
        self.fusion_method = fusion_method
    
    def _rrf_fusion(
        self, 
        dense_results: List[SearchResult], 
        sparse_results: List[SearchResult],
        k: int = 60
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion.
        
        RRF(d) = Σ 1 / (k + rank(d))
        """
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}
        
        for rank, result in enumerate(dense_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (k + rank + 1)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
        
        for rank, result in enumerate(sparse_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1 / (k + rank + 1)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            result.score = rrf_scores[chunk_id]
            results.append(result)
        
        return results
    
    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Weighted score combination."""
        score_map: Dict[str, Tuple[float, float]] = {}
        result_map: Dict[str, SearchResult] = {}
        
        for result in dense_results:
            score_map[result.chunk_id] = (result.score, 0)
            result_map[result.chunk_id] = result
        
        for result in sparse_results:
            dense_score = score_map.get(result.chunk_id, (0, 0))[0]
            score_map[result.chunk_id] = (dense_score, result.score)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
        
        # Normalize and combine
        max_dense = max([s[0] for s in score_map.values()]) or 1
        max_sparse = max([s[1] for s in score_map.values()]) or 1
        
        combined_scores = {}
        for chunk_id, (dense, sparse) in score_map.items():
            combined = self.alpha * (dense / max_dense) + (1 - self.alpha) * (sparse / max_sparse)
            combined_scores[chunk_id] = combined
        
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        results = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            result.score = combined_scores[chunk_id]
            results.append(result)
        
        return results
    
    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Perform hybrid search."""
        dense_results = self.dense.search(query, k * 2)
        sparse_results = self.sparse.search(query, k * 2)
        
        if self.fusion_method == "rrf":
            fused = self._rrf_fusion(dense_results, sparse_results)
        else:
            fused = self._weighted_fusion(dense_results, sparse_results)
        
        return fused[:k]


# ============================================================
# RERANKING
# ============================================================

class CrossEncoderReranker:
    """
    Cross-encoder reranking for improved precision.
    
    More accurate than bi-encoder but slower.
    Use after initial retrieval to rerank top candidates.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """Lazy load cross-encoder."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers required")
        return self._model
    
    def rerank(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of results to return
        
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        model = self._get_model()
        
        # Create query-document pairs
        pairs = [(query, r.content) for r in results]
        
        # Score with cross-encoder
        scores = model.predict(pairs)
        
        # Update results with rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.rerank_score or 0, reverse=True)
        
        if top_k:
            return reranked[:top_k]
        return reranked


# ============================================================
# QUERY ENGINE
# ============================================================

class LegalQueryEngine:
    """
    Complete query engine for legal document QA.
    
    Pipeline:
        Query -> Hybrid Search -> Rerank -> Generate Answer
    
    Example:
        >>> engine = LegalQueryEngine(retriever, generator)
        >>> result = engine.query("What are the termination clauses?")
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        retriever: Retriever,
        generator: Optional[Any] = None,  # LLM generator
        reranker: Optional[CrossEncoderReranker] = None,
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5
    ):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
    
    def _build_prompt(self, query: str, sources: List[SearchResult]) -> str:
        """Build prompt for answer generation."""
        context = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{s.content}"
            for i, s in enumerate(sources)
        ])
        
        prompt = f"""You are a legal research assistant. Answer the question based ONLY on the provided legal documents.

IMPORTANT:
- Only use information from the sources below
- Cite source numbers when making claims [Source N]
- If the answer cannot be found in the sources, say so
- Be precise with legal terminology

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def _estimate_confidence(self, sources: List[SearchResult]) -> float:
        """Estimate answer confidence based on retrieval scores."""
        if not sources:
            return 0.0
        
        # Use rerank scores if available, otherwise use regular scores
        scores = [s.rerank_score if s.rerank_score else s.score for s in sources]
        
        # High confidence if top result is strong and others are close
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range (heuristic)
        confidence = min(1.0, (top_score + avg_score) / 2)
        
        return confidence
    
    def query(self, query: str) -> QueryResult:
        """
        Execute query and generate answer.
        
        Args:
            query: User's question
        
        Returns:
            QueryResult with answer and sources
        """
        import time
        start_time = time.time()
        
        # Retrieve
        results = self.retriever.search(query, self.top_k_retrieval)
        
        # Rerank
        if self.reranker and results:
            results = self.reranker.rerank(query, results, self.top_k_rerank)
        else:
            results = results[:self.top_k_rerank]
        
        # Generate answer
        if self.generator and results:
            prompt = self._build_prompt(query, results)
            answer = self.generator(prompt)
        else:
            answer = "No generator configured. Top sources retrieved."
        
        # Extract citations from sources
        citations = []
        for source in results:
            if 'citations' in source.metadata:
                citations.extend(source.metadata['citations'])
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResult(
            query=query,
            answer=answer,
            sources=results,
            confidence=self._estimate_confidence(results),
            citations=list(set(citations)),
            processing_time_ms=processing_time
        )
    
    def search_only(self, query: str, k: int = 10) -> List[SearchResult]:
        """Search without generation."""
        results = self.retriever.search(query, k)
        
        if self.reranker:
            results = self.reranker.rerank(query, results, k)
        
        return results


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'SearchResult', 'QueryResult',
    'Retriever', 'DenseRetriever', 'SparseRetriever', 'HybridRetriever',
    'CrossEncoderReranker',
    'LegalQueryEngine',
]
