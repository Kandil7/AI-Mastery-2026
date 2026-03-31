"""
Temporal-Aware RAG (Retrieval-Augmented Generation) Module

This module implements a temporal-aware RAG system that considers time-based
information in both retrieval and generation processes. It can handle time-sensitive
queries and retrieve documents based on temporal relevance, recency, and
historical context.

Key Features:
- Time-aware document indexing with timestamps
- Temporal similarity matching
- Recency bias adjustment
- Historical context retrieval
- Time-series aware generation
- Temporal query understanding

Architecture:
- Temporal Document Indexer: Maintains time-based document organization
- Temporal Retriever: Retrieves documents considering temporal factors
- Temporal Scorer: Scores documents based on time relevance
- Temporal Generator: Generates responses considering temporal context
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import hashlib
import datetime
from dateutil import parser
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class TemporalScope(Enum):
    """Temporal scopes for queries and documents."""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    HISTORICAL = "historical"
    RECENT = "recent"
    ALL_TIME = "all_time"


@dataclass
class TemporalDocument:
    """A document with temporal information."""
    id: str
    content: str
    timestamp: datetime.datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    validity_period: Optional[Tuple[datetime.datetime, datetime.datetime]] = None  # Valid from, valid to
    temporal_tags: List[str] = field(default_factory=list)  # e.g., "breaking_news", "historical_record"
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from content and timestamp hash
            content_hash = hashlib.md5(
                (self.content + str(self.timestamp)).encode()
            ).hexdigest()[:16]
            self.id = content_hash
    
    @property
    def age_days(self) -> float:
        """Calculate age of document in days."""
        return (datetime.datetime.now() - self.timestamp).total_seconds() / (24 * 3600)
    
    @property
    def is_current(self) -> bool:
        """Check if document is currently valid."""
        if self.validity_period:
            start, end = self.validity_period
            now = datetime.datetime.now()
            return start <= now <= end
        return True  # Assume valid if no validity period specified


@dataclass
class TemporalQuery:
    """A query with temporal context."""
    text: str
    reference_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    temporal_scope: TemporalScope = TemporalScope.ALL_TIME
    recency_bias: float = 0.5  # 0.0 = no recency bias, 1.0 = high recency bias
    time_window_days: Optional[int] = None  # Only consider documents within this time window
    temporal_keywords: List[str] = field(default_factory=list)  # e.g., "recent", "historical", "past year"
    
    def __post_init__(self):
        # Extract temporal keywords from text
        self.temporal_keywords = self._extract_temporal_keywords()
    
    def _extract_temporal_keywords(self) -> List[str]:
        """Extract temporal keywords from query text."""
        text_lower = self.text.lower()
        temporal_patterns = [
            r'current|now|today|recent|latest|new',
            r'past|previous|old|earlier|before|ago',
            r'future|next|upcoming|later|tomorrow',
            r'historical|history|ancient|old',
            r'last week|last month|last year|recently',
            r'in (\d+) (days|weeks|months|years)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)'
        ]
        
        keywords = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        return [str(k) for k in keywords if k]


@dataclass
class TemporalRetrievalResult:
    """Result from temporal retrieval."""
    document: TemporalDocument
    score: float
    temporal_score: float  # Score component based on temporal relevance
    relevance_score: float  # Score component based on content relevance
    age_factor: float  # Factor representing document age impact
    rank: int = 0


@dataclass
class TemporalGenerationResult:
    """Result from temporal generation."""
    answer: str
    sources: List[TemporalDocument]
    temporal_context: str  # Summary of temporal context used
    confidence: float
    latency_ms: float
    token_count: int
    temporal_accuracy: float  # How well temporal aspects were addressed


# ============================================================
# TEMPORAL SCORERS
# ============================================================

class BaseTemporalScorer(ABC):
    """Base class for temporal scorers."""
    
    @abstractmethod
    def score(self, query: TemporalQuery, document: TemporalDocument) -> float:
        """Score document based on temporal relevance to query."""
        pass


class RecencyScorer(BaseTemporalScorer):
    """Scores documents based on recency."""
    
    def __init__(self, decay_factor: float = 0.1):
        """
        Initialize recency scorer.
        
        Args:
            decay_factor: How quickly older documents are penalized (higher = faster decay)
        """
        self.decay_factor = decay_factor
    
    def score(self, query: TemporalQuery, document: TemporalDocument) -> float:
        """Score based on document recency."""
        # Calculate age in days
        age_days = document.age_days
        
        # Apply exponential decay
        recency_score = np.exp(-self.decay_factor * age_days)
        
        # Adjust based on query's recency bias
        adjusted_score = recency_score * query.recency_bias
        
        return min(1.0, max(0.0, adjusted_score))


class TemporalWindowScorer(BaseTemporalScorer):
    """Scores documents based on temporal window constraints."""
    
    def score(self, query: TemporalQuery, document: TemporalDocument) -> float:
        """Score based on temporal window."""
        if query.time_window_days is None:
            return 1.0  # No window constraint
        
        # Calculate time difference in days
        time_diff = abs((query.reference_time - document.timestamp).total_seconds()) / (24 * 3600)
        
        # Score decreases as we move away from the window
        if time_diff <= query.time_window_days:
            # Within window - full score
            return 1.0
        else:
            # Outside window - decreasing score
            excess_days = time_diff - query.time_window_days
            penalty = min(1.0, excess_days / (query.time_window_days * 2))
            return max(0.0, 1.0 - penalty)


class TemporalScopeScorer(BaseTemporalScorer):
    """Scores documents based on temporal scope alignment."""
    
    def score(self, query: TemporalQuery, document: TemporalDocument) -> float:
        """Score based on temporal scope alignment."""
        doc_time = document.timestamp
        query_time = query.reference_time
        
        if query.temporal_scope == TemporalScope.PAST:
            return 1.0 if doc_time < query_time else 0.1
        elif query.temporal_scope == TemporalScope.FUTURE:
            return 1.0 if doc_time > query_time else 0.1
        elif query.temporal_scope == TemporalScope.PRESENT:
            # Consider documents from past week to next week
            week_ago = query_time - datetime.timedelta(days=7)
            next_week = query_time + datetime.timedelta(days=7)
            return 1.0 if week_ago <= doc_time <= next_week else 0.3
        elif query.temporal_scope == TemporalScope.HISTORICAL:
            # Prefer older documents
            age_months = document.age_days / 30
            return min(1.0, age_months / 12)  # Up to 1 year gets full score
        elif query.temporal_scope == TemporalScope.RECENT:
            # Prefer newer documents
            age_days = document.age_days
            return max(0.0, 1.0 - (age_days / 30))  # Full score for documents less than 30 days old
        else:  # ALL_TIME
            return 1.0


class TemporalKeywordScorer(BaseTemporalScorer):
    """Scores documents based on temporal keyword alignment."""
    
    def score(self, query: TemporalQuery, document: TemporalDocument) -> float:
        """Score based on temporal keyword alignment."""
        if not query.temporal_keywords:
            return 1.0
        
        # Check if document content mentions years or dates that align with query
        doc_text = document.content.lower()
        query_text = query.text.lower()
        
        # Look for year patterns in document
        doc_years = re.findall(r'\b(19|20)\d{2}\b', doc_text)
        query_years = re.findall(r'\b(19|20)\d{2}\b', query_text)
        
        if query_years and doc_years:
            # If both have years, score based on proximity
            query_year = int(query_years[0]) if query_years else None
            doc_year = int(doc_years[0]) if doc_years else None
            
            if query_year and doc_year:
                year_diff = abs(query_year - doc_year)
                return max(0.0, 1.0 - (year_diff / 10))  # Penalize for year differences
        
        # Check for temporal phrases in document
        temporal_indicators = [
            'recently', 'currently', 'now', 'today', 'yesterday', 'last week', 
            'last month', 'last year', 'previously', 'historically', 'ancient',
            'modern', 'contemporary', 'outdated', 'current'
        ]
        
        doc_has_temporal = any(indicator in doc_text for indicator in temporal_indicators)
        return 1.0 if doc_has_temporal else 0.5


# ============================================================
# TEMPORAL RETRIEVER
# ============================================================

class TemporalRetriever:
    """
    Temporal-aware retriever that considers time-based factors.
    
    This retriever combines traditional semantic similarity with temporal
    relevance scoring to retrieve time-appropriate documents.
    """
    
    def __init__(self, 
                 embedding_dim: int = 384,
                 recency_decay: float = 0.1,
                 temporal_weight: float = 0.3):
        """
        Initialize temporal retriever.
        
        Args:
            embedding_dim: Dimension of document embeddings
            recency_decay: How quickly older documents are penalized
            temporal_weight: Weight given to temporal factors vs content relevance
        """
        self.embedding_dim = embedding_dim
        self.temporal_weight = temporal_weight
        self.documents: List[TemporalDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Initialize temporal scorers
        self.recency_scorer = RecencyScorer(decay_factor=recency_decay)
        self.window_scorer = TemporalWindowScorer()
        self.scope_scorer = TemporalScopeScorer()
        self.keyword_scorer = TemporalKeywordScorer()
    
    def add_documents(self, documents: List[TemporalDocument]) -> None:
        """Add temporal documents to the retriever."""
        self.documents.extend(documents)
        
        # Collect embeddings
        new_embeddings = []
        for doc in documents:
            if doc.embedding is not None:
                new_embeddings.append(doc.embedding)
            else:
                # Generate random embedding as placeholder
                new_embeddings.append(np.random.randn(self.embedding_dim))
        
        if new_embeddings:
            new_emb_array = np.array(new_embeddings)
            if self.embeddings is None:
                self.embeddings = new_emb_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb_array])
    
    def _compute_semantic_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute semantic similarity between query and documents."""
        if self.embeddings is None or len(self.documents) == 0:
            return np.array([])
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(emb_norms, query_norm)
        return similarities
    
    def _compute_temporal_score(self, query: TemporalQuery, document: TemporalDocument) -> float:
        """Compute composite temporal score for document."""
        # Combine scores from different temporal scorers
        recency_score = self.recency_scorer.score(query, document)
        window_score = self.window_scorer.score(query, document)
        scope_score = self.scope_scorer.score(query, document)
        keyword_score = self.keyword_scorer.score(query, document)
        
        # Average the scores
        temporal_score = (recency_score + window_score + scope_score + keyword_score) / 4.0
        
        return temporal_score
    
    def retrieve(self, 
                 query: TemporalQuery, 
                 query_embedding: np.ndarray, 
                 k: int = 5) -> List[TemporalRetrievalResult]:
        """Retrieve documents considering both semantic and temporal relevance."""
        if len(self.documents) == 0:
            return []
        
        # Compute semantic similarities
        semantic_similarities = self._compute_semantic_similarity(query_embedding)
        
        # Compute temporal scores for each document
        temporal_scores = []
        for doc in self.documents:
            temp_score = self._compute_temporal_score(query, doc)
            temporal_scores.append(temp_score)
        
        temporal_scores = np.array(temporal_scores)
        
        # Combine semantic and temporal scores
        combined_scores = (
            (1 - self.temporal_weight) * semantic_similarities + 
            self.temporal_weight * temporal_scores
        )
        
        # Get top-k indices
        top_k_indices = np.argsort(combined_scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            if idx < len(self.documents):
                doc = self.documents[idx]
                age_factor = 1.0 / (1.0 + doc.age_days / 30)  # Normalize by age in months
                
                result = TemporalRetrievalResult(
                    document=doc,
                    score=float(combined_scores[idx]),
                    temporal_score=float(temporal_scores[idx]),
                    relevance_score=float(semantic_similarities[idx]),
                    age_factor=age_factor,
                    rank=rank
                )
                results.append(result)
        
        return results


# ============================================================
# TEMPORAL GENERATOR
# ============================================================

class TemporalGenerator:
    """
    Generator that considers temporal context in responses.
    
    This component generates responses that are temporally aware,
    mentioning when information was relevant and providing context
    about the timing of events.
    """
    
    def __init__(self):
        self.generate_fn: Optional[Callable] = None
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generate_fn = generate_fn
    
    def generate(self, 
                 query: TemporalQuery, 
                 context: str, 
                 retrieved_docs: List[TemporalDocument]) -> TemporalGenerationResult:
        """Generate temporally-aware response."""
        import time
        start_time = time.time()
        
        # Build temporal context
        temporal_context = self._build_temporal_context(query, retrieved_docs)
        
        # Build the generation prompt
        prompt = self._build_prompt(query, context, temporal_context)
        
        # Generate answer
        if self.generate_fn:
            answer = self.generate_fn(prompt)
        else:
            # Placeholder answer
            answer = self._generate_placeholder(query, context, retrieved_docs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimate temporal accuracy based on how well temporal aspects were addressed
        temporal_accuracy = self._estimate_temporal_accuracy(answer, query, retrieved_docs)
        
        result = TemporalGenerationResult(
            answer=answer,
            sources=retrieved_docs,
            temporal_context=temporal_context,
            confidence=0.7,  # Placeholder confidence
            latency_ms=latency_ms,
            token_count=len(prompt.split()) + len(answer.split()),
            temporal_accuracy=temporal_accuracy
        )
        
        return result
    
    def _build_temporal_context(self, 
                               query: TemporalQuery, 
                               docs: List[TemporalDocument]) -> str:
        """Build temporal context summary."""
        if not docs:
            return "No temporal context available."
        
        # Find the time range of retrieved documents
        timestamps = [doc.timestamp for doc in docs]
        earliest = min(timestamps)
        latest = max(timestamps)
        
        # Count documents by time period
        now = datetime.datetime.now()
        recent_docs = sum(1 for doc in docs if (now - doc.timestamp).days <= 30)
        older_docs = len(docs) - recent_docs
        
        context_parts = [
            f"Retrieved {len(docs)} documents spanning from {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}.",
            f"Of these, {recent_docs} are from the last 30 days and {older_docs} are older."
        ]
        
        # Add temporal scope information
        if query.temporal_scope != TemporalScope.ALL_TIME:
            context_parts.append(f"The query focused on the {query.temporal_scope.value} timeframe.")
        
        # Add time window information if applicable
        if query.time_window_days:
            context_parts.append(f"Documents were filtered to those within {query.time_window_days} days of the reference time.")
        
        return " ".join(context_parts)
    
    def _build_prompt(self, 
                     query: TemporalQuery, 
                     context: str, 
                     temporal_context: str) -> str:
        """Build the generation prompt with temporal awareness."""
        return f"""Answer the question based on the provided context, paying special attention to temporal aspects.
The query was made on {query.reference_time.strftime('%Y-%m-%d %H:%M:%S')}.
Temporal context: {temporal_context}

If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Always cite your sources and mention the timeframes when information was relevant.

Context:
{context}

Question: {query.text}

Answer:"""
    
    def _generate_placeholder(self, 
                            query: TemporalQuery, 
                            context: str, 
                            docs: List[TemporalDocument]) -> str:
        """Generate a placeholder answer for testing."""
        if docs:
            time_range = f"from {min(d.timestamp for d in docs).strftime('%Y-%m-%d')} to {max(d.timestamp for d in docs).strftime('%Y-%m-%d')}"
            return f"Based on the provided context {time_range}, the system found {len(docs)} relevant documents. The query was made on {query.reference_time.strftime('%Y-%m-%d')}."
        else:
            return "No relevant temporal information found for your query."
    
    def _estimate_temporal_accuracy(self, 
                                 answer: str, 
                                 query: TemporalQuery, 
                                 docs: List[TemporalDocument]) -> float:
        """Estimate how well temporal aspects were addressed in the answer."""
        answer_lower = answer.lower()
        
        # Check for temporal indicators in the answer
        temporal_indicators = [
            'date', 'time', 'year', 'month', 'day', 'recent', 'current', 
            'past', 'future', 'ago', 'since', 'until', 'during', 'when'
        ]
        
        has_temporal_indicators = any(indicator in answer_lower for indicator in temporal_indicators)
        
        # Check if specific dates from documents appear in answer
        doc_dates_mentioned = 0
        for doc in docs:
            date_str = doc.timestamp.strftime('%Y')
            if date_str in answer_lower:
                doc_dates_mentioned += 1
        
        # Calculate accuracy score
        temporal_score = 0.5  # Base score
        
        if has_temporal_indicators:
            temporal_score += 0.3
        
        if doc_dates_mentioned > 0:
            temporal_score += 0.2 * min(1.0, doc_dates_mentioned / len(docs))
        
        return min(1.0, temporal_score)


# ============================================================
# TEMPORAL RAG SYSTEM
# ============================================================

class TemporalAwareRAG:
    """
    Temporal-Aware RAG system that handles time-sensitive queries.
    
    This system can process queries that have temporal context and
    retrieve documents considering time-based relevance factors.
    """
    
    def __init__(self, 
                 embedding_dim: int = 384,
                 recency_decay: float = 0.1,
                 temporal_weight: float = 0.3):
        """
        Initialize temporal-aware RAG system.
        
        Args:
            embedding_dim: Dimension of document embeddings
            recency_decay: How quickly older documents are penalized
            temporal_weight: Weight given to temporal factors vs content relevance
        """
        self.retriever = TemporalRetriever(
            embedding_dim=embedding_dim,
            recency_decay=recency_decay,
            temporal_weight=temporal_weight
        )
        self.generator = TemporalGenerator()
        
        logger.info("Initialized Temporal-Aware RAG system")
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function for both retriever and generator."""
        self.generator.set_generator(generate_fn)
    
    def add_documents(self, documents: List[TemporalDocument]) -> int:
        """Add temporal documents to the RAG system."""
        # Ensure documents have embeddings
        for doc in documents:
            if doc.embedding is None:
                # Generate a simple embedding based on content
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                embedding = np.frombuffer(bytes.fromhex(content_hash[:32]), dtype=np.float32)
                # Pad or truncate to desired dimension
                if len(embedding) < self.retriever.embedding_dim:
                    embedding = np.pad(embedding, (0, self.retriever.embedding_dim - len(embedding)), 'constant')
                elif len(embedding) > self.retriever.embedding_dim:
                    embedding = embedding[:self.retriever.embedding_dim]
                doc.embedding = embedding
        
        self.retriever.add_documents(documents)
        logger.info(f"Added {len(documents)} temporal documents")
        return len(documents)
    
    def query(self, 
              query: TemporalQuery, 
              query_embedding: np.ndarray, 
              k: int = 5) -> TemporalGenerationResult:
        """
        Query the temporal-aware RAG system.
        
        Args:
            query: Temporal query object
            query_embedding: Embedding vector for the query
            k: Number of results to retrieve
            
        Returns:
            TemporalGenerationResult with answer and temporal context
        """
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(query, query_embedding, k=k)
        
        if not retrieval_results:
            # No results found, return a default response
            return TemporalGenerationResult(
                answer="No relevant temporal information found for your query.",
                sources=[],
                temporal_context="No documents retrieved.",
                confidence=0.0,
                latency_ms=0.0,
                token_count=10,
                temporal_accuracy=0.0
            )
        
        # Build context from retrieved documents
        context_parts = []
        retrieved_docs = []
        for result in retrieval_results:
            context_parts.append(f"[{result.document.timestamp.strftime('%Y-%m-%d')}] {result.document.content}")
            retrieved_docs.append(result.document)
        
        context = "\n\n".join(context_parts)
        
        # Generate response with temporal awareness
        result = self.generator.generate(query, context, retrieved_docs)
        
        return result


# ============================================================
# EXAMPLE USAGE AND TESTING
# ============================================================

def example_usage():
    """Demonstrate Temporal-Aware RAG usage."""
    
    # Create RAG system
    rag = TemporalAwareRAG(temporal_weight=0.4)
    
    # Sample temporal documents
    now = datetime.datetime.now()
    documents = [
        TemporalDocument(
            id="doc1",
            content="The company reported record profits in Q4 2023, with revenues reaching $1.2 billion.",
            timestamp=now - datetime.timedelta(days=30),  # 1 month ago
            metadata={"source": "financial_report", "quarter": "Q4_2023"},
            temporal_tags=["recent", "financial"]
        ),
        TemporalDocument(
            id="doc2",
            content="Market analysis from early 2023 showed steady growth across all sectors.",
            timestamp=now - datetime.timedelta(days=300),  # ~10 months ago
            metadata={"source": "market_analysis", "period": "early_2023"},
            temporal_tags=["historical", "analysis"]
        ),
        TemporalDocument(
            id="doc3",
            content="The new product launch scheduled for next quarter is expected to drive significant growth.",
            timestamp=now + datetime.timedelta(days=90),  # 3 months in future
            metadata={"source": "planning_doc", "event": "product_launch"},
            temporal_tags=["future", "planned"]
        ),
        TemporalDocument(
            id="doc4",
            content="Historical data from 2020 shows the impact of global events on market stability.",
            timestamp=now - datetime.timedelta(days=2200),  # ~6 years ago
            metadata={"source": "historical_data", "year": "2020"},
            temporal_tags=["historical", "past_event"]
        )
    ]
    
    # Add documents
    num_docs = rag.add_documents(documents)
    print(f"Added {num_docs} temporal documents")
    
    # Create a temporal query
    temporal_query = TemporalQuery(
        text="What were the recent financial results?",
        reference_time=now,
        temporal_scope=TemporalScope.RECENT,
        recency_bias=0.8,
        time_window_days=60  # Only consider last 60 days
    )
    
    # Create a simple query embedding (in practice, this would come from an embedding model)
    query_text_hash = hashlib.md5(temporal_query.text.encode()).hexdigest()
    query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
    if len(query_embedding) < 384:
        query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
    elif len(query_embedding) > 384:
        query_embedding = query_embedding[:384]
    
    # Query the system
    result = rag.query(temporal_query, query_embedding, k=3)
    
    print(f"\nQuery: {temporal_query.text}")
    print(f"Reference time: {temporal_query.reference_time}")
    print(f"Temporal scope: {temporal_query.temporal_scope.value}")
    print(f"Temporal context: {result.temporal_context}")
    print(f"Temporal accuracy: {result.temporal_accuracy:.2f}")
    print(f"Answer: {result.answer}")
    print(f"Sources: {len(result.sources)}")
    
    # Show another example with historical focus
    print("\n" + "="*60)
    print("Historical Query Example:")
    
    hist_query = TemporalQuery(
        text="What happened in 2020?",
        reference_time=now,
        temporal_scope=TemporalScope.HISTORICAL,
        recency_bias=0.1  # Low recency bias for historical focus
    )
    
    hist_result = rag.query(hist_query, query_embedding, k=2)
    print(f"Query: {hist_query.text}")
    print(f"Temporal context: {hist_result.temporal_context}")
    print(f"Answer: {hist_result.answer}")
    
    return rag


if __name__ == "__main__":
    example_usage()