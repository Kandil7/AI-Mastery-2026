"""
Reranking Module - AI-Mastery-2026

This module implements second-stage reranking strategies to improve
retrieval quality beyond initial retrieval.

Key Components:
- CrossEncoderReranker: BERT-based cross-encoder for accurate scoring
- LLMReranker: Use LLM prompts for semantic reranking
- ReciprocalRankFusion: Combine multiple ranked lists
- DiversityReranker: MMR-style diversity optimization

Why Reranking?
- Initial retrieval is fast but may miss nuances
- Reranking is slower but more accurate
- Two-stage approach: fast retrieval → accurate reranking

Author: AI-Mastery-2026
License: MIT
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Document:
    """Represents a document for reranking."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RerankResult:
    """
    Represents a reranking result.
    
    Attributes:
        document: The reranked document
        original_score: Score from initial retrieval
        rerank_score: Score from reranking
        final_score: Combined final score
        original_rank: Position before reranking
        final_rank: Position after reranking
    """
    document: Document
    original_score: float
    rerank_score: float
    final_score: float
    original_rank: int
    final_rank: int


@dataclass
class RerankConfig:
    """
    Configuration for reranking.
    
    Attributes:
        top_k: Number of documents to rerank
        batch_size: Batch size for processing
        score_threshold: Minimum score to include
        diversity_lambda: Weight for diversity (0-1)
        combine_method: How to combine scores ('replace', 'linear', 'harmonic')
        alpha: Weight for rerank score in linear combination
    """
    top_k: int = 10
    batch_size: int = 32
    score_threshold: float = 0.0
    diversity_lambda: float = 0.0
    combine_method: str = "replace"  # 'replace', 'linear', 'harmonic'
    alpha: float = 0.7  # Weight for rerank score


# =============================================================================
# Base Reranker
# =============================================================================

class BaseReranker(ABC):
    """
    Abstract base class for all rerankers.
    
    Rerankers take initial retrieval results and produce improved rankings
    by applying more sophisticated scoring methods.
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.config = RerankConfig()
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: User query
            documents: Documents to rerank
            top_k: Number of results to return
            
        Returns:
            List of RerankResult objects
        """
        pass
    
    def _combine_scores(
        self,
        original_score: float,
        rerank_score: float
    ) -> float:
        """Combine original and rerank scores based on config."""
        method = self.config.combine_method
        alpha = self.config.alpha
        
        if method == "replace":
            return rerank_score
        elif method == "linear":
            return alpha * rerank_score + (1 - alpha) * original_score
        elif method == "harmonic":
            if original_score + rerank_score == 0:
                return 0.0
            return 2 * original_score * rerank_score / (original_score + rerank_score)
        else:
            return rerank_score


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================

class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder based reranker using BERT-style models.
    
    Unlike bi-encoders (separate query/document encoding), cross-encoders
    process query and document together, enabling richer interaction.
    
    Architecture:
        Input: [CLS] query [SEP] document [SEP]
        Output: Single relevance score
    
    Pros:
        - More accurate than bi-encoders
        - Captures cross-attention between query and document
    
    Cons:
        - Slower (cannot pre-compute document embeddings)
        - Not suitable for initial retrieval at scale
    
    Example:
        >>> reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> results = reranker.rerank(query, documents, top_k=10)
    
    Attributes:
        model_name: HuggingFace model name for cross-encoder
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: Optional[RerankConfig] = None
    ):
        """
        Initialize CrossEncoderReranker.
        
        Args:
            model_name: Name of the cross-encoder model
            config: Reranking configuration
            
        Popular models:
            - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
            - cross-encoder/ms-marco-TinyBERT-L-6 (faster)
            - cross-encoder/ms-marco-electra-base (accurate)
        """
        super().__init__(name="cross_encoder")
        self.model_name = model_name
        self.config = config or RerankConfig()
        self._model = None
        
        logger.info(f"Initialized CrossEncoderReranker with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed, using fallback")
                self._model = "fallback"
    
    def _fallback_score(self, query: str, document: str) -> float:
        """
        Fallback scoring when model is unavailable.
        
        Uses simple word overlap as a proxy for relevance.
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        overlap = len(query_words & doc_words)
        union = len(query_words | doc_words)
        
        return overlap / union if union > 0 else 0.0
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: User query
            documents: Documents to rerank (with original scores)
            top_k: Number of results to return
            
        Returns:
            List of RerankResult objects sorted by final score
            
        Example:
            >>> docs = [Document(id="1", content="ML is AI", score=0.8), ...]
            >>> results = reranker.rerank("machine learning", docs, top_k=5)
        """
        top_k = top_k or self.config.top_k
        self._load_model()
        
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Get scores
        if self._model == "fallback":
            scores = [self._fallback_score(query, doc.content) for doc in documents]
        else:
            scores = self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
        
        # Create results with combined scores
        results = []
        for i, (doc, rerank_score) in enumerate(zip(documents, scores)):
            final_score = self._combine_scores(doc.score, float(rerank_score))
            results.append(RerankResult(
                document=doc,
                original_score=doc.score,
                rerank_score=float(rerank_score),
                final_score=final_score,
                original_rank=i + 1,
                final_rank=0  # Will be set after sorting
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign final ranks and filter
        final_results = []
        for rank, result in enumerate(results[:top_k], 1):
            result.final_rank = rank
            if result.final_score >= self.config.score_threshold:
                final_results.append(result)
        
        return final_results
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: User query
            document: Document text
            
        Returns:
            Relevance score
        """
        self._load_model()
        
        if self._model == "fallback":
            return self._fallback_score(query, document)
        
        return float(self._model.predict([(query, document)])[0])


# =============================================================================
# LLM Reranker
# =============================================================================

class LLMReranker(BaseReranker):
    """
    LLM-based reranker using prompt engineering.
    
    Uses a language model to assess relevance through prompting,
    enabling zero-shot reranking with natural language understanding.
    
    Strategies:
        - 'pointwise': Score each document independently
        - 'pairwise': Compare documents pairwise
        - 'listwise': Rank entire list at once
    
    Example:
        >>> reranker = LLMReranker(strategy="pointwise")
        >>> results = reranker.rerank(query, documents, top_k=5)
    
    Attributes:
        strategy: Reranking strategy ('pointwise', 'pairwise', 'listwise')
        model_name: LLM model to use
    """
    
    # Prompt templates for different strategies
    POINTWISE_PROMPT = """Given the following query and document, rate the relevance on a scale of 0-10.

Query: {query}

Document: {document}

Relevance score (0-10):"""

    PAIRWISE_PROMPT = """Given the query, which document is more relevant? Answer A or B.

Query: {query}

Document A: {doc_a}

Document B: {doc_b}

More relevant (A or B):"""
    
    def __init__(
        self,
        strategy: str = "pointwise",
        model_name: str = "gpt2",
        config: Optional[RerankConfig] = None
    ):
        """
        Initialize LLMReranker.
        
        Args:
            strategy: Reranking strategy
            model_name: LLM model name
            config: Reranking configuration
        """
        super().__init__(name="llm_reranker")
        self.strategy = strategy
        self.model_name = model_name
        self.config = config or RerankConfig()
        self._model = None
        self._tokenizer = None
        
        logger.info(f"Initialized LLMReranker with strategy: {strategy}")
    
    def _load_model(self):
        """Lazy load the language model."""
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self._model.eval()
            except ImportError:
                logger.warning("transformers not installed, using fallback")
                self._model = "fallback"
    
    def _pointwise_score(self, query: str, document: str) -> float:
        """
        Score document using pointwise strategy.
        
        Asks the LLM to rate relevance from 0-10.
        """
        import torch
        
        self._load_model()
        
        if self._model == "fallback":
            # Fallback: word overlap
            q_words = set(query.lower().split())
            d_words = set(document.lower().split())
            return len(q_words & d_words) / max(len(q_words), 1)
        
        # Create prompt
        prompt = self.POINTWISE_PROMPT.format(
            query=query,
            document=document[:500]  # Truncate long documents
        )
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate score
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False
            )
        
        # Extract score from output
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        try:
            score = float(response.split()[0])
            return min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
        except (ValueError, IndexError):
            return 0.5  # Default score if parsing fails
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents using LLM.
        
        Args:
            query: User query
            documents: Documents to rerank
            top_k: Number of results to return
            
        Returns:
            List of RerankResult objects
        """
        top_k = top_k or self.config.top_k
        
        if not documents:
            return []
        
        # Score each document
        results = []
        for i, doc in enumerate(documents):
            rerank_score = self._pointwise_score(query, doc.content)
            final_score = self._combine_scores(doc.score, rerank_score)
            
            results.append(RerankResult(
                document=doc,
                original_score=doc.score,
                rerank_score=rerank_score,
                final_score=final_score,
                original_rank=i + 1,
                final_rank=0
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign final ranks
        for rank, result in enumerate(results, 1):
            result.final_rank = rank
        
        return results[:top_k]


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================

class ReciprocalRankFusion(BaseReranker):
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.
    
    RRF is a simple but effective method for combining rankings from
    different retrieval systems without requiring score normalization.
    
    Formula:
        RRF(d) = Σ 1 / (k + rank(d, r))
        
    Where:
        - d: document
        - r: ranking from retriever
        - k: constant (typically 60)
    
    Benefits:
        - No score normalization needed
        - Handles different score scales
        - Simple and effective
        - Works well with diverse retrievers
    
    Example:
        >>> rrf = ReciprocalRankFusion(k=60)
        >>> combined = rrf.fuse([bm25_results, dense_results])
    
    Attributes:
        k: RRF constant (higher = more weight to lower ranks)
    """
    
    def __init__(
        self,
        k: int = 60,
        config: Optional[RerankConfig] = None
    ):
        """
        Initialize RRF reranker.
        
        Args:
            k: RRF constant (typical values: 60, 100)
            config: Reranking configuration
        """
        super().__init__(name="rrf")
        self.k = k
        self.config = config or RerankConfig()
        
        logger.info(f"Initialized ReciprocalRankFusion with k={k}")
    
    def fuse(
        self,
        ranked_lists: List[List[Document]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Fuse multiple ranked lists using RRF.
        
        Args:
            ranked_lists: List of ranked document lists
            top_k: Number of results to return
            
        Returns:
            Fused list of RerankResult objects
            
        Example:
            >>> bm25_docs = retriever1.retrieve(query)
            >>> dense_docs = retriever2.retrieve(query)
            >>> fused = rrf.fuse([bm25_docs, dense_docs], top_k=10)
        """
        top_k = top_k or self.config.top_k
        
        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, 1):
                doc_map[doc.id] = doc
                rrf_score = 1.0 / (self.k + rank)
                rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + rrf_score
        
        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create results
        results = []
        for rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], 1):
            doc = doc_map[doc_id]
            results.append(RerankResult(
                document=doc,
                original_score=doc.score,
                rerank_score=rrf_score,
                final_score=rrf_score,
                original_rank=0,  # Multiple original ranks
                final_rank=rank
            ))
        
        return results
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using RRF (treats input as single list).
        
        For RRF, use fuse() with multiple lists instead.
        """
        # Single list, just return with assigned ranks
        results = []
        for rank, doc in enumerate(documents[:top_k or self.config.top_k], 1):
            results.append(RerankResult(
                document=doc,
                original_score=doc.score,
                rerank_score=1.0 / (self.k + rank),
                final_score=1.0 / (self.k + rank),
                original_rank=rank,
                final_rank=rank
            ))
        return results


# =============================================================================
# Diversity Reranker (MMR)
# =============================================================================

class DiversityReranker(BaseReranker):
    """
    Diversity-aware reranker using Maximal Marginal Relevance (MMR).
    
    MMR balances relevance with diversity by penalizing documents
    that are too similar to already selected documents.
    
    Formula:
        MMR = arg max [λ · Sim(d, q) - (1-λ) · max Sim(d, d_i)]
        
    Where:
        - d: candidate document
        - q: query
        - d_i: already selected documents
        - λ: trade-off parameter (0 = max diversity, 1 = max relevance)
    
    Benefits:
        - Reduces redundancy in results
        - Covers multiple aspects of the query
        - Useful for question answering, summarization
    
    Example:
        >>> reranker = DiversityReranker(lambda_param=0.7)
        >>> diversified = reranker.rerank(query, documents, top_k=10)
    
    Attributes:
        lambda_param: Relevance/diversity trade-off (0-1)
    """
    
    def __init__(
        self,
        lambda_param: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
        config: Optional[RerankConfig] = None
    ):
        """
        Initialize MMR diversity reranker.
        
        Args:
            lambda_param: Relevance weight (1 = no diversity consideration)
            embedding_model: Model for computing document similarity
            config: Reranking configuration
        """
        super().__init__(name="diversity_mmr")
        self.lambda_param = lambda_param
        self.embedding_model = embedding_model
        self.config = config or RerankConfig()
        self._encoder = None
        
        logger.info(f"Initialized DiversityReranker with λ={lambda_param}")
    
    def _load_encoder(self):
        """Load sentence encoder for computing similarities."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model)
            except ImportError:
                self._encoder = "fallback"
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for texts."""
        self._load_encoder()
        
        if self._encoder == "fallback":
            # Fallback: random embeddings (for demo purposes)
            return np.random.randn(len(texts), 384).astype(np.float32)
        
        return self._encoder.encode(texts, normalize_embeddings=True)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents using MMR for diversity.
        
        Args:
            query: User query
            documents: Documents to rerank (should have relevance scores)
            top_k: Number of results to return
            
        Returns:
            Diversified list of RerankResult objects
            
        Algorithm:
            1. Start with empty selected set
            2. While |selected| < top_k:
                a. For each unselected doc, compute MMR score
                b. Select doc with highest MMR score
            3. Return selected documents
        """
        top_k = top_k or self.config.top_k
        
        if not documents:
            return []
        
        # Compute embeddings for query and documents
        texts = [query] + [doc.content for doc in documents]
        embeddings = self._compute_embeddings(texts)
        
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Normalize relevance scores to [0, 1]
        max_score = max(doc.score for doc in documents) or 1.0
        relevance_scores = [doc.score / max_score for doc in documents]
        
        # MMR selection
        selected_indices = []
        selected_embeddings = []
        remaining_indices = list(range(len(documents)))
        
        results = []
        
        while len(selected_indices) < top_k and remaining_indices:
            best_idx = None
            best_mmr = float('-inf')
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = relevance_scores[idx]
                
                # Max similarity to already selected documents
                if selected_embeddings:
                    max_sim_to_selected = max(
                        self._cosine_similarity(doc_embeddings[idx], sel_emb)
                        for sel_emb in selected_embeddings
                    )
                else:
                    max_sim_to_selected = 0
                
                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim_to_selected
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(doc_embeddings[best_idx])
                remaining_indices.remove(best_idx)
                
                doc = documents[best_idx]
                results.append(RerankResult(
                    document=doc,
                    original_score=doc.score,
                    rerank_score=best_mmr,
                    final_score=best_mmr,
                    original_rank=best_idx + 1,
                    final_rank=len(results) + 1
                ))
        
        return results


# =============================================================================
# Reranking Pipeline
# =============================================================================

class RerankingPipeline:
    """
    Orchestrates multiple reranking stages.
    
    Allows chaining different rerankers for sophisticated
    multi-stage reranking pipelines.
    
    Example:
        >>> pipeline = RerankingPipeline()
        >>> pipeline.add_stage(cross_encoder_reranker)
        >>> pipeline.add_stage(diversity_reranker)
        >>> results = pipeline.rerank(query, documents, top_k=10)
    """
    
    def __init__(self):
        """Initialize reranking pipeline."""
        self.stages: List[BaseReranker] = []
        self.config = RerankConfig()
    
    def add_stage(self, reranker: BaseReranker) -> "RerankingPipeline":
        """
        Add a reranking stage to the pipeline.
        
        Args:
            reranker: Reranker to add
            
        Returns:
            Self for chaining
        """
        self.stages.append(reranker)
        return self
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Run the reranking pipeline.
        
        Args:
            query: User query
            documents: Documents to rerank
            top_k: Number of final results
            
        Returns:
            Final reranked results
        """
        top_k = top_k or self.config.top_k
        
        if not self.stages:
            # No rerankers, return as-is
            return [
                RerankResult(
                    document=doc,
                    original_score=doc.score,
                    rerank_score=doc.score,
                    final_score=doc.score,
                    original_rank=i + 1,
                    final_rank=i + 1
                )
                for i, doc in enumerate(documents[:top_k])
            ]
        
        # Run through each stage
        current_docs = documents
        
        for stage in self.stages:
            results = stage.rerank(query, current_docs, top_k=len(current_docs))
            
            # Update documents with new scores for next stage
            current_docs = [
                Document(
                    id=r.document.id,
                    content=r.document.content,
                    metadata=r.document.metadata,
                    score=r.final_score
                )
                for r in results
            ]
        
        # Final reranking
        final_results = self.stages[-1].rerank(query, current_docs, top_k=top_k)
        
        return final_results


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1] range using min-max normalization.
    
    Args:
        scores: List of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


def compute_rank_changes(results: List[RerankResult]) -> Dict[str, int]:
    """
    Compute rank changes from reranking.
    
    Args:
        results: Reranked results
        
    Returns:
        Dictionary mapping document IDs to rank change
    """
    return {
        r.document.id: r.original_rank - r.final_rank
        for r in results
    }


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reranking Module Demo")
    print("=" * 60)
    
    # Sample documents (simulating retrieval results)
    docs = [
        Document(id="1", content="Machine learning is a subset of AI.", score=0.85),
        Document(id="2", content="Deep learning uses neural networks.", score=0.82),
        Document(id="3", content="Python is great for data science.", score=0.75),
        Document(id="4", content="Neural networks learn from data.", score=0.70),
        Document(id="5", content="AI transforms many industries.", score=0.65),
    ]
    
    query = "How do neural networks learn?"
    
    # Test Cross-Encoder
    print("\n1. Cross-Encoder Reranking")
    print("-" * 40)
    
    cross_encoder = CrossEncoderReranker()
    results = cross_encoder.rerank(query, docs, top_k=3)
    
    for r in results:
        print(f"  Rank {r.final_rank}: [{r.rerank_score:.3f}] {r.document.content}")
        print(f"    (was rank {r.original_rank}, original score: {r.original_score:.3f})")
    
    # Test RRF
    print("\n2. Reciprocal Rank Fusion")
    print("-" * 40)
    
    rrf = ReciprocalRankFusion(k=60)
    
    # Simulate two different rankings
    list1 = [docs[0], docs[1], docs[3], docs[2], docs[4]]
    list2 = [docs[3], docs[1], docs[0], docs[4], docs[2]]
    
    fused = rrf.fuse([list1, list2], top_k=3)
    
    for r in fused:
        print(f"  Rank {r.final_rank}: [RRF: {r.rerank_score:.4f}] {r.document.content}")
    
    # Test Diversity
    print("\n3. Diversity Reranking (MMR)")
    print("-" * 40)
    
    diversity = DiversityReranker(lambda_param=0.5)
    diversified = diversity.rerank(query, docs, top_k=3)
    
    for r in diversified:
        print(f"  Rank {r.final_rank}: [MMR: {r.rerank_score:.3f}] {r.document.content}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
