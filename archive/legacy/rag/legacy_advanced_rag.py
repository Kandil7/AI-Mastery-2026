"""
Advanced RAG Module
===================

Enterprise-grade Retrieval-Augmented Generation system.

Inspired by Notion AI architecture:
- Semantic Chunking: Structure-aware document splitting
- Hybrid Retrieval: Vector similarity + BM25 keyword search
- Model Router: Cost-efficient model selection
- LLM-as-Judge: Automated quality evaluation

Features:
- Hierarchical document chunking
- Dense + sparse retrieval with score fusion
- Intelligent model routing based on task complexity
- Automated evaluation with LLM judges

References:
- Notion AI architecture case study
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod
import re
import logging
from enum import Enum
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    """A document in the RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    """A chunk of a document."""
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    start_char: int = 0
    end_char: int = 0
    level: int = 0  # Hierarchical level (0 = top level)
    parent_id: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    chunk: Chunk
    score: float
    retrieval_method: str  # "dense", "sparse", or "hybrid"


@dataclass
class GenerationResult:
    """Result from generation."""
    answer: str
    sources: List[Chunk]
    model_used: str
    confidence: float
    latency_ms: float
    token_count: int


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

class ChunkingStrategy(Enum):
    """Chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class SemanticChunker:
    """
    Structure-aware document chunking.
    
    Preserves semantic units (paragraphs, sections) rather than
    splitting arbitrarily at character boundaries.
    
    Implements:
    - Sentence-based chunking with overlap
    - Paragraph-aware splitting
    - Hierarchical chunking (sections > paragraphs > sentences)
    - Metadata preservation
    """
    
    def __init__(self, 
                 strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
                 chunk_size: int = 512,
                 overlap: int = 50,
                 min_chunk_size: int = 100):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split document into chunks based on strategy.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(document)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunking(document)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._paragraph_chunking(document)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(document)
        elif self.strategy == ChunkingStrategy.HIERARCHICAL:
            return self._hierarchical_chunking(document)
        else:
            return self._fixed_size_chunking(document)
            
    def _fixed_size_chunking(self, document: Document) -> List[Chunk]:
        """Simple fixed-size chunking with overlap."""
        chunks = []
        content = document.content
        
        start = 0
        chunk_idx = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence end
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = content[start:end].rfind(punct)
                    if last_punct != -1:
                        end = start + last_punct + len(punct)
                        break
                        
            chunk_content = content[start:end].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_content,
                    metadata={**document.metadata, "chunk_idx": chunk_idx},
                    start_char=start,
                    end_char=end,
                    level=0
                ))
                chunk_idx += 1
                
            start = end - self.overlap
            
        return chunks
    
    def _sentence_chunking(self, document: Document) -> List[Chunk]:
        """Chunk at sentence boundaries."""
        # Simple sentence splitting
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, document.content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_size + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = ' '.join(current_chunk)
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_content,
                    metadata={**document.metadata, "chunk_idx": chunk_idx},
                    level=0
                ))
                chunk_idx += 1
                
                # Keep last sentence for overlap
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_size = len(current_chunk[0]) if current_chunk else 0
                
            current_chunk.append(sentence)
            current_size += len(sentence)
            
        # Add remaining
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_content,
                    metadata={**document.metadata, "chunk_idx": chunk_idx},
                    level=0
                ))
                
        return chunks
    
    def _paragraph_chunking(self, document: Document) -> List[Chunk]:
        """Chunk at paragraph boundaries."""
        paragraphs = document.content.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if current_size + len(para) > self.chunk_size and current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_content,
                    metadata={**document.metadata, "chunk_idx": chunk_idx},
                    level=0
                ))
                chunk_idx += 1
                current_chunk = []
                current_size = 0
                
            current_chunk.append(para)
            current_size += len(para)
            
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=chunk_content,
                    metadata={**document.metadata, "chunk_idx": chunk_idx},
                    level=0
                ))
                
        return chunks
    
    def _semantic_chunking(self, document: Document) -> List[Chunk]:
        """
        Semantic chunking based on content structure.
        
        Detects:
        - Headers/sections
        - Lists
        - Code blocks
        - Tables
        """
        chunks = []
        content = document.content
        
        # Split by markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        
        chunk_idx = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Extract header if present
            header_match = re.match(header_pattern, section, re.MULTILINE)
            header = header_match.group(2) if header_match else None
            level = len(header_match.group(1)) if header_match else 0
            
            # If section is too long, split into paragraphs
            if len(section) > self.chunk_size:
                sub_chunks = self._paragraph_chunking(Document(
                    id=document.id,
                    content=section,
                    metadata=document.metadata
                ))
                for sub_chunk in sub_chunks:
                    sub_chunk.id = f"{document.id}_chunk_{chunk_idx}"
                    sub_chunk.metadata["section_header"] = header
                    sub_chunk.level = level
                    chunks.append(sub_chunk)
                    chunk_idx += 1
            else:
                chunks.append(Chunk(
                    id=f"{document.id}_chunk_{chunk_idx}",
                    document_id=document.id,
                    content=section,
                    metadata={**document.metadata, "section_header": header},
                    level=level
                ))
                chunk_idx += 1
                
        return chunks
    
    def _hierarchical_chunking(self, document: Document) -> List[Chunk]:
        """
        Hierarchical chunking creating parent-child relationships.
        
        Creates multiple levels:
        - Level 0: Document summary
        - Level 1: Sections
        - Level 2: Paragraphs
        - Level 3: Sentences
        """
        all_chunks = []
        
        # Level 0: Full document (as summary placeholder)
        doc_summary = document.content[:500] + "..." if len(document.content) > 500 else document.content
        root_chunk = Chunk(
            id=f"{document.id}_root",
            document_id=document.id,
            content=doc_summary,
            metadata={**document.metadata, "is_summary": True},
            level=0
        )
        all_chunks.append(root_chunk)
        
        # Level 1: Sections
        sections = re.split(r'\n(?=#{1,3}\s)', document.content)
        
        section_idx = 0
        for section in sections:
            if not section.strip():
                continue
                
            section_chunk = Chunk(
                id=f"{document.id}_section_{section_idx}",
                document_id=document.id,
                content=section[:self.chunk_size] if len(section) > self.chunk_size else section,
                metadata={**document.metadata},
                level=1,
                parent_id=root_chunk.id
            )
            all_chunks.append(section_chunk)
            
            # Level 2: Paragraphs within section
            paragraphs = section.split('\n\n')
            para_idx = 0
            for para in paragraphs:
                if len(para.strip()) < self.min_chunk_size:
                    continue
                    
                para_chunk = Chunk(
                    id=f"{document.id}_section_{section_idx}_para_{para_idx}",
                    document_id=document.id,
                    content=para.strip(),
                    metadata={**document.metadata},
                    level=2,
                    parent_id=section_chunk.id
                )
                all_chunks.append(para_chunk)
                para_idx += 1
                
            section_idx += 1
            
        return all_chunks


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

class DenseRetriever:
    """Dense retrieval using embeddings."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks with their embeddings."""
        self.chunks.extend(chunks)
        
        # Collect embeddings
        new_embeddings = []
        for chunk in chunks:
            if chunk.embedding is not None:
                new_embeddings.append(chunk.embedding)
            else:
                # Generate random embedding as placeholder
                new_embeddings.append(np.random.randn(self.embedding_dim))
                
        if new_embeddings:
            new_emb_array = np.array(new_embeddings)
            if self.embeddings is None:
                self.embeddings = new_emb_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb_array])
                
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k chunks by embedding similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
            
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = emb_norms @ query_norm
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(similarities[idx]),
                retrieval_method="dense"
            ))
            
        return results


class SparseRetriever:
    """Sparse retrieval using BM25."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.vocab: set = set()
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
        
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index."""
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            
            # Term frequency
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
                self.vocab.add(token)
                
            # Document frequency
            for token in set(tokens):
                self.doc_freqs[token] += 1
                
            self.chunks.append(chunk)
            self.term_freqs.append(dict(tf))
            self.doc_lengths.append(len(tokens))
            
        # Update average document length
        if self.doc_lengths:
            self.avg_doc_length = np.mean(self.doc_lengths)
            
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k chunks using BM25."""
        if not self.chunks:
            return []
            
        query_tokens = self._tokenize(query)
        n_docs = len(self.chunks)
        
        scores = []
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            doc_len = self.doc_lengths[i]
            tf = self.term_freqs[i]
            
            for token in query_tokens:
                if token not in tf:
                    continue
                    
                # BM25 formula
                f = tf[token]
                df = self.doc_freqs.get(token, 0)
                idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
                
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                
                score += idf * numerator / denominator
                
            scores.append(score)
            
        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    retrieval_method="sparse"
                ))
                
        return results


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(self, 
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4,
                 rrf_k: int = 60):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to both retrievers."""
        self.dense_retriever.add_chunks(chunks)
        self.sparse_retriever.add_chunks(chunks)
        
    def retrieve(self, query: str, 
                 query_embedding: np.ndarray,
                 k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve using hybrid approach.
        
        Args:
            query: Query text (for sparse)
            query_embedding: Query embedding (for dense)
            k: Number of results
            
        Returns:
            Fused retrieval results
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.retrieve(query_embedding, k=k*2)
        sparse_results = self.sparse_retriever.retrieve(query, k=k*2)
        
        # Reciprocal Rank Fusion
        chunk_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Chunk] = {}
        
        for rank, result in enumerate(dense_results):
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            chunk_scores[result.chunk.id] += self.dense_weight * rrf_score
            chunk_map[result.chunk.id] = result.chunk
            
        for rank, result in enumerate(sparse_results):
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            chunk_scores[result.chunk.id] += self.sparse_weight * rrf_score
            chunk_map[result.chunk.id] = result.chunk
            
        # Sort by fused score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in sorted_chunks[:k]:
            results.append(RetrievalResult(
                chunk=chunk_map[chunk_id],
                score=score,
                retrieval_method="hybrid"
            ))
            
        return results


# ============================================================
# MODEL ROUTER
# ============================================================

class ModelTier(Enum):
    """Model tiers based on capability and cost."""
    FAST = "fast"           # Fastest, lowest cost
    BALANCED = "balanced"   # Balance of speed and quality
    REASONING = "reasoning" # Best quality, highest cost


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    tier: ModelTier
    context_window: int
    cost_per_1k_tokens: float
    avg_latency_ms: float
    capabilities: List[str]


class ModelRouter:
    """
    Intelligent model routing for cost optimization.
    
    Routes queries to the most appropriate model based on:
    - Task complexity
    - Context length requirements
    - Cost constraints
    - Latency requirements
    
    Inspired by Notion AI's model routing architecture.
    """
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.default_model: Optional[str] = None
        self._setup_default_models()
        
    def _setup_default_models(self):
        """Setup default model configurations."""
        self.register_model(ModelConfig(
            name="gpt-4-turbo",
            tier=ModelTier.REASONING,
            context_window=128000,
            cost_per_1k_tokens=0.03,
            avg_latency_ms=2000,
            capabilities=["reasoning", "analysis", "code", "creative"]
        ))
        
        self.register_model(ModelConfig(
            name="gpt-3.5-turbo",
            tier=ModelTier.BALANCED,
            context_window=16384,
            cost_per_1k_tokens=0.002,
            avg_latency_ms=500,
            capabilities=["general", "translation", "summarization"]
        ))
        
        self.register_model(ModelConfig(
            name="claude-instant",
            tier=ModelTier.FAST,
            context_window=100000,
            cost_per_1k_tokens=0.0008,
            avg_latency_ms=200,
            capabilities=["general", "extraction", "formatting"]
        ))
        
        self.default_model = "gpt-3.5-turbo"
        
    def register_model(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self.models[config.name] = config
        
    def route(self, 
              query: str,
              context_length: int,
              task_type: str = "general",
              max_latency_ms: Optional[float] = None,
              max_cost_per_1k: Optional[float] = None) -> str:
        """
        Route to the optimal model.
        
        Args:
            query: User query
            context_length: Total context length (query + context)
            task_type: Type of task (reasoning, summarization, etc.)
            max_latency_ms: Maximum acceptable latency
            max_cost_per_1k: Maximum acceptable cost
            
        Returns:
            Model name to use
        """
        # Filter by constraints
        candidates = []
        
        for name, config in self.models.items():
            # Check context window
            if config.context_window < context_length:
                continue
                
            # Check latency constraint
            if max_latency_ms and config.avg_latency_ms > max_latency_ms:
                continue
                
            # Check cost constraint
            if max_cost_per_1k and config.cost_per_1k_tokens > max_cost_per_1k:
                continue
                
            # Check capability
            if task_type not in config.capabilities and "general" not in config.capabilities:
                continue
                
            candidates.append(config)
            
        if not candidates:
            logger.warning(f"No suitable model found, using default: {self.default_model}")
            return self.default_model
            
        # Score candidates
        scored = []
        for config in candidates:
            score = self._score_model(config, task_type, context_length)
            scored.append((config.name, score))
            
        # Return best scoring model
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def _score_model(self, config: ModelConfig, 
                     task_type: str, context_length: int) -> float:
        """Score a model for the given task."""
        score = 0.0
        
        # Task-specific scoring
        if task_type == "reasoning" and config.tier == ModelTier.REASONING:
            score += 10.0
        elif task_type in ["extraction", "formatting"] and config.tier == ModelTier.FAST:
            score += 8.0
        elif config.tier == ModelTier.BALANCED:
            score += 5.0
            
        # Cost efficiency
        cost_score = 1.0 / (config.cost_per_1k_tokens + 0.001)
        score += cost_score * 0.1
        
        # Latency bonus
        latency_score = 1.0 / (config.avg_latency_ms + 100)
        score += latency_score * 1000
        
        # Context fit (prefer models with context window close to needed)
        context_ratio = context_length / config.context_window
        if 0.1 < context_ratio < 0.8:
            score += 2.0
            
        return score


# ============================================================
# LLM-AS-JUDGE EVALUATION
# ============================================================

@dataclass
class EvaluationCriteria:
    """Criteria for LLM-as-Judge evaluation."""
    name: str
    description: str
    weight: float = 1.0
    scale: int = 5  # 1-5 scale


class LLMJudge:
    """
    LLM-as-Judge for automated quality evaluation.
    
    Uses an LLM to evaluate generated responses against
    defined criteria, providing scalable quality assessment.
    """
    
    DEFAULT_CRITERIA = [
        EvaluationCriteria(
            name="relevance",
            description="How relevant is the answer to the question?",
            weight=1.0
        ),
        EvaluationCriteria(
            name="accuracy",
            description="Is the answer factually accurate based on the context?",
            weight=1.5
        ),
        EvaluationCriteria(
            name="completeness",
            description="Does the answer fully address the question?",
            weight=1.0
        ),
        EvaluationCriteria(
            name="coherence",
            description="Is the answer well-structured and coherent?",
            weight=0.5
        )
    ]
    
    def __init__(self, criteria: Optional[List[EvaluationCriteria]] = None):
        self.criteria = criteria or self.DEFAULT_CRITERIA
        
    def evaluate(self, 
                 query: str,
                 answer: str,
                 context: str,
                 generate_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate an answer using LLM-as-Judge.
        
        Args:
            query: Original question
            answer: Generated answer
            context: Retrieved context used
            generate_fn: LLM generation function (simulated if not provided)
            
        Returns:
            Evaluation results with scores and feedback
        """
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(query, answer, context)
        
        # Get evaluation (simulated if no generate_fn)
        if generate_fn:
            evaluation_text = generate_fn(prompt)
            scores = self._parse_evaluation(evaluation_text)
        else:
            # Simulated evaluation
            scores = self._simulate_evaluation(query, answer, context)
            
        # Calculate weighted score
        total_weight = sum(c.weight for c in self.criteria)
        weighted_score = sum(
            scores.get(c.name, 3) * c.weight 
            for c in self.criteria
        ) / total_weight
        
        return {
            "scores": scores,
            "weighted_score": weighted_score,
            "max_score": 5.0,
            "criteria": [c.name for c in self.criteria],
            "pass": weighted_score >= 3.5
        }
    
    def _build_evaluation_prompt(self, query: str, 
                                  answer: str, context: str) -> str:
        """Build the evaluation prompt."""
        criteria_text = "\n".join([
            f"- {c.name}: {c.description} (1-{c.scale} scale)"
            for c in self.criteria
        ])
        
        return f"""Evaluate the following answer based on these criteria:

{criteria_text}

Question: {query}

Context provided:
{context[:2000]}

Answer to evaluate:
{answer}

For each criterion, provide a score (1-5) and brief justification.
Format:
criterion_name: score - justification
"""

    def _parse_evaluation(self, text: str) -> Dict[str, float]:
        """Parse evaluation text to extract scores."""
        scores = {}
        
        for criterion in self.criteria:
            pattern = rf'{criterion.name}:\s*(\d+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[criterion.name] = float(match.group(1))
            else:
                scores[criterion.name] = 3.0  # Default mid-score
                
        return scores
    
    def _simulate_evaluation(self, query: str, 
                             answer: str, context: str) -> Dict[str, float]:
        """Simulate evaluation for testing."""
        scores = {}
        
        # Simple heuristic-based scoring
        for criterion in self.criteria:
            # Base score
            score = 3.0
            
            # Relevance: check keyword overlap
            if criterion.name == "relevance":
                query_words = set(query.lower().split())
                answer_words = set(answer.lower().split())
                overlap = len(query_words & answer_words)
                score = min(5.0, 2.0 + overlap * 0.5)
                
            # Completeness: check answer length
            elif criterion.name == "completeness":
                if len(answer) > 200:
                    score = 4.0
                elif len(answer) > 100:
                    score = 3.5
                else:
                    score = 2.5
                    
            # Coherence: check for common issues
            elif criterion.name == "coherence":
                score = 4.0 if answer.endswith(('.', '!', '?')) else 3.0
                
            # Accuracy: check context usage
            elif criterion.name == "accuracy":
                context_words = set(context.lower().split())
                answer_words = set(answer.lower().split())
                overlap = len(context_words & answer_words)
                score = min(5.0, 2.0 + overlap * 0.1)
                
            scores[criterion.name] = score
            
        return scores


# ============================================================
# ENTERPRISE RAG SYSTEM
# ============================================================

class EnterpriseRAG:
    """
    Enterprise-grade RAG system combining all components.
    
    Features:
    - Semantic document chunking
    - Hybrid retrieval (dense + sparse)
    - Intelligent model routing
    - Automated quality evaluation
    - Source citation
    
    Example:
        >>> rag = EnterpriseRAG()
        >>> rag.add_documents([Document(id="doc1", content="...")])
        >>> result = rag.query("What is X?", query_embedding=np.random.randn(384))
        >>> print(result.answer)
        >>> print(result.sources)
    """
    
    def __init__(self,
                 chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
                 embedding_dim: int = 384,
                 enable_evaluation: bool = True):
        self.chunker = SemanticChunker(strategy=chunking_strategy)
        self.retriever = HybridRetriever()
        self.router = ModelRouter()
        self.judge = LLMJudge() if enable_evaluation else None
        self.embedding_dim = embedding_dim
        
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        
        # Generation function (placeholder - replace with actual LLM)
        self.generate_fn: Optional[Callable] = None
        
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generate_fn = generate_fn
        
    def add_documents(self, documents: List[Document],
                      embeddings: Optional[Dict[str, np.ndarray]] = None) -> int:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings (chunk_id -> embedding)
            
        Returns:
            Number of chunks created
        """
        all_chunks = []
        
        for doc in documents:
            # Store document
            self.documents[doc.id] = doc
            
            # Chunk document
            chunks = self.chunker.chunk_document(doc)
            
            for chunk in chunks:
                # Assign embedding
                if embeddings and chunk.id in embeddings:
                    chunk.embedding = embeddings[chunk.id]
                elif doc.embedding is not None:
                    # Use document embedding as fallback
                    chunk.embedding = doc.embedding
                else:
                    # Generate random embedding as placeholder
                    chunk.embedding = np.random.randn(self.embedding_dim)
                    
                self.chunks[chunk.id] = chunk
                all_chunks.append(chunk)
                
        # Add to retriever
        self.retriever.add_chunks(all_chunks)
        
        logger.info(f"Added {len(documents)} documents, {len(all_chunks)} chunks")
        return len(all_chunks)
    
    def query(self, 
              query: str,
              query_embedding: np.ndarray,
              k: int = 5,
              task_type: str = "general",
              max_latency_ms: Optional[float] = None) -> GenerationResult:
        """
        Query the RAG system.
        
        Args:
            query: User question
            query_embedding: Query embedding vector
            k: Number of chunks to retrieve
            task_type: Type of task for model routing
            max_latency_ms: Latency constraint
            
        Returns:
            GenerationResult with answer and sources
        """
        import time
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve(query, query_embedding, k=k)
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        for result in retrieval_results:
            context_parts.append(result.chunk.content)
            sources.append(result.chunk)
            
        context = "\n\n---\n\n".join(context_parts)
        
        # Route to appropriate model
        context_length = len(query) + len(context)
        model = self.router.route(
            query=query,
            context_length=context_length,
            task_type=task_type,
            max_latency_ms=max_latency_ms
        )
        
        # Generate answer
        prompt = self._build_prompt(query, context)
        
        if self.generate_fn:
            answer = self.generate_fn(prompt)
        else:
            # Placeholder answer
            answer = self._generate_placeholder(query, context)
            
        latency_ms = (time.time() - start_time) * 1000
        
        # Build result
        result = GenerationResult(
            answer=answer,
            sources=sources,
            model_used=model,
            confidence=self._estimate_confidence(retrieval_results),
            latency_ms=latency_ms,
            token_count=len(prompt.split()) + len(answer.split())
        )
        
        return result
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the generation prompt."""
        return f"""Answer the question based on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Always cite your sources.

Context:
{context}

Question: {query}

Answer:"""

    def _generate_placeholder(self, query: str, context: str) -> str:
        """Generate a placeholder answer for testing."""
        # Extract key sentences from context
        sentences = context.split('.')[:3]
        summary = '. '.join(s.strip() for s in sentences if s.strip())
        
        return f"Based on the provided context: {summary}."
    
    def _estimate_confidence(self, results: List[RetrievalResult]) -> float:
        """Estimate confidence based on retrieval scores."""
        if not results:
            return 0.0
            
        # Average of top retrieval scores
        scores = [r.score for r in results[:3]]
        avg_score = np.mean(scores)
        
        # Normalize to 0-1 range
        return float(min(1.0, avg_score))
    
    def evaluate(self, query: str, answer: str, 
                 context: str) -> Optional[Dict[str, Any]]:
        """Evaluate answer quality using LLM-as-Judge."""
        if not self.judge:
            return None
        return self.judge.evaluate(query, answer, context, self.generate_fn)


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_usage():
    """Demonstrate Enterprise RAG usage."""
    
    # Create RAG system
    rag = EnterpriseRAG(
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        embedding_dim=384
    )
    
    # Sample documents
    documents = [
        Document(
            id="doc1",
            content="""# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a technique that combines 
retrieval systems with language models.

## How RAG Works

1. First, relevant documents are retrieved from a knowledge base
2. These documents are used as context for the language model
3. The model generates a response grounded in the retrieved context

## Benefits

RAG reduces hallucination by grounding responses in actual data.
It also allows models to access up-to-date information.
""",
            metadata={"source": "tutorial", "topic": "rag"}
        ),
        Document(
            id="doc2",
            content="""# Vector Databases

Vector databases store and search embeddings efficiently.

## Common Vector Databases

- Pinecone: Managed vector database
- Weaviate: Open-source with hybrid search
- Milvus: Scalable open-source option
- Turbopuffer: Object storage-native design

## Indexing Algorithms

HNSW (Hierarchical Navigable Small World) is the most common
indexing algorithm used for approximate nearest neighbor search.
""",
            metadata={"source": "tutorial", "topic": "vector_db"}
        )
    ]
    
    # Add documents
    num_chunks = rag.add_documents(documents)
    print(f"Created {num_chunks} chunks")
    
    # Query
    query = "How does RAG reduce hallucination?"
    query_embedding = np.random.randn(384)  # In practice, use actual embedding
    
    result = rag.query(query, query_embedding, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Model used: {result.model_used}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"\nAnswer: {result.answer}")
    print(f"\nSources: {len(result.sources)}")
    for source in result.sources:
        print(f"  - Chunk {source.id}: {source.content[:50]}...")
        
    return rag


if __name__ == "__main__":
    example_usage()
