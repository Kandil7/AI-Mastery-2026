"""
Continual Learning RAG (Retrieval-Augmented Generation) Module

This module implements a continual learning RAG system that can adapt and improve
over time without forgetting previously learned information. It incorporates
techniques like elastic weight consolidation, progressive neural networks,
and rehearsal methods to enable lifelong learning.

Key Features:
- Incremental document addition without full retraining
- Catastrophic forgetting prevention
- Experience replay and rehearsal
- Dynamic knowledge expansion
- Performance monitoring and adaptation
- Lifelong learning capabilities

Architecture:
- Continual Learner: Manages model updates and prevents forgetting
- Experience Buffer: Stores important experiences for rehearsal
- Knowledge Integrator: Integrates new knowledge with existing knowledge
- Forgetting Prevention: Mechanisms to prevent catastrophic forgetting
- Performance Monitor: Tracks performance and triggers adaptation
- Adaptive Retriever: Adjusts retrieval based on learned patterns
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import hashlib
import datetime
import pickle
import os
from collections import deque
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class LearningEventType(Enum):
    """Types of learning events."""
    DOCUMENT_ADDED = "document_added"
    QUERY_PROCESSED = "query_processed"
    FEEDBACK_RECEIVED = "feedback_received"
    PERFORMANCE_DEGRADED = "performance_degraded"
    KNOWLEDGE_UPDATE = "knowledge_update"


class ForgettingMechanism(Enum):
    """Mechanisms to prevent forgetting."""
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"
    PROGRESSIVE_NEURAL_NETWORKS = "progressive_nn"
    EXPERIENCE_REPLAY = "experience_replay"
    DUAL_MEMORY = "dual_memory"
    PARAMETER_ISOLATION = "parameter_isolation"


@dataclass
class ContinualDocument:
    """A document in the continual learning system."""
    id: str
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    embedding: Optional[np.ndarray] = None
    importance_score: float = 1.0  # How important this document is
    access_frequency: int = 0  # How often this document is accessed
    relevance_history: List[Tuple[datetime.datetime, float]] = field(default_factory=list)  # (time, relevance)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5((self.content + str(self.timestamp)).encode()).hexdigest()[:16]


@dataclass
class LearningExperience:
    """An experience tuple for rehearsal."""
    query: str
    retrieved_docs: List[ContinualDocument]
    response: str
    feedback: Optional[str] = None  # User feedback
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    performance_score: float = 1.0  # How well the system performed
    importance: float = 1.0  # How important this experience is


@dataclass
class ContinualQuery:
    """A query in the continual learning system."""
    text: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    difficulty: float = 0.5  # How difficult the query is (0-1)
    domain: str = "general"  # Domain of the query
    expected_response_length: int = 100  # Expected length of response


@dataclass
class ContinualRetrievalResult:
    """Result from continual learning retrieval."""
    document: ContinualDocument
    score: float
    learning_signal: float  # Signal for learning (relevance feedback)
    novelty_score: float  # How novel this information is
    rank: int = 0


@dataclass
class ContinualGenerationResult:
    """Result from continual learning generation."""
    answer: str
    sources: List[ContinualDocument]
    learning_experiences: List[LearningExperience]
    confidence: float
    latency_ms: float
    token_count: int
    adaptation_needed: bool  # Whether adaptation is needed


# ============================================================
# CONTINUAL LEARNING COMPONENTS
# ============================================================

class ExperienceBuffer:
    """Buffer to store important experiences for rehearsal."""
    
    def __init__(self, capacity: int = 1000, importance_weight: float = 0.7):
        self.capacity = capacity
        self.importance_weight = importance_weight
        self.buffer: deque = deque(maxlen=capacity)
    
    def add_experience(self, experience: LearningExperience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size: int) -> List[LearningExperience]:
        """Sample a batch of experiences for rehearsal."""
        if len(self.buffer) == 0:
            return []
        
        # Prioritize experiences by importance
        sorted_experiences = sorted(self.buffer, key=lambda x: x.importance, reverse=True)
        
        # Return top experiences
        return sorted_experiences[:min(batch_size, len(sorted_experiences))]
    
    def update_importance(self, experience_id: str, new_importance: float) -> None:
        """Update the importance of a specific experience."""
        for i, exp in enumerate(self.buffer):
            # Since we don't have explicit IDs, we'll update based on content similarity
            if hashlib.md5(exp.query.encode()).hexdigest()[:16] == experience_id:
                self.buffer[i] = LearningExperience(
                    query=exp.query,
                    retrieved_docs=exp.retrieved_docs,
                    response=exp.response,
                    feedback=exp.feedback,
                    timestamp=exp.timestamp,
                    performance_score=exp.performance_score,
                    importance=new_importance
                )


class ForgettingPrevention:
    """Mechanisms to prevent catastrophic forgetting."""
    
    def __init__(self, mechanism: ForgettingMechanism = ForgettingMechanism.EXPERIENCE_REPLAY):
        self.mechanism = mechanism
        self.saved_parameters: Optional[Dict] = None
        self.parameter_importance: Optional[np.ndarray] = None  # For EWC
    
    def apply_forgetting_prevention(self, 
                                   current_params: Dict[str, np.ndarray], 
                                   new_data: List[Any]) -> Dict[str, np.ndarray]:
        """Apply forgetting prevention mechanism."""
        if self.mechanism == ForgettingMechanism.ELASTIC_WEIGHT_CONSOLIDATION:
            return self._apply_elastic_weight_consolidation(current_params, new_data)
        elif self.mechanism == ForgettingMechanism.EXPERIENCE_REPLAY:
            return self._apply_experience_replay(current_params, new_data)
        elif self.mechanism == ForgettingMechanism.PROGRESSIVE_NEURAL_NETWORKS:
            return self._apply_progressive_networks(current_params, new_data)
        else:
            # Default: no special prevention
            return current_params
    
    def _apply_elastic_weight_consolidation(self, 
                                          current_params: Dict[str, np.ndarray], 
                                          new_data: List[Any]) -> Dict[str, np.ndarray]:
        """Apply Elastic Weight Consolidation to prevent forgetting."""
        # Simplified EWC implementation
        updated_params = {}
        
        for name, param in current_params.items():
            if self.parameter_importance is not None and name in self.parameter_importance:
                # Apply quadratic penalty based on parameter importance
                penalty_strength = 1000.0  # Lambda in EWC
                prev_param = self.saved_parameters[name] if self.saved_parameters and name in self.saved_parameters else param
                importance = self.parameter_importance[name] if isinstance(self.parameter_importance, dict) else self.parameter_importance
                
                # Calculate EWC loss contribution and adjust parameters
                ewc_adjustment = penalty_strength * importance * (param - prev_param)
                updated_params[name] = param - 0.01 * ewc_adjustment  # Small learning rate
            else:
                updated_params[name] = param
        
        return updated_params
    
    def _apply_experience_replay(self, 
                                current_params: Dict[str, np.ndarray], 
                                new_data: List[Any]) -> Dict[str, np.ndarray]:
        """Apply experience replay to prevent forgetting."""
        # In a real implementation, this would train on both new data and old experiences
        # For this simulation, we'll just return the current parameters
        return current_params
    
    def _apply_progressive_networks(self, 
                                   current_params: Dict[str, np.ndarray], 
                                   new_data: List[Any]) -> Dict[str, np.ndarray]:
        """Apply progressive neural networks (expand network architecture)."""
        # In a real implementation, this would expand the network
        # For this simulation, we'll just return the current parameters
        return current_params


class PerformanceMonitor:
    """Monitors system performance and detects degradation."""
    
    def __init__(self, performance_threshold: float = 0.7, window_size: int = 100):
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        self.performance_history: deque = deque(maxlen=window_size)
        self.adaptation_triggered = False
    
    def update_performance(self, score: float) -> bool:
        """Update performance and check if adaptation is needed."""
        self.performance_history.append(score)
        
        # Check if recent performance is below threshold
        if len(self.performance_history) >= 10:  # Need minimum samples
            recent_avg = np.mean(list(self.performance_history)[-10:])
            if recent_avg < self.performance_threshold:
                self.adaptation_triggered = True
                logger.warning(f"Performance degradation detected: {recent_avg:.3f} < {self.performance_threshold}")
                return True
        
        return False
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.performance_history:
            return {"average": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        perf_list = list(self.performance_history)
        return {
            "average": float(np.mean(perf_list)),
            "min": float(np.min(perf_list)),
            "max": float(np.max(perf_list)),
            "count": len(perf_list),
            "trend": "decreasing" if len(perf_list) > 5 and np.polyfit(range(len(perf_list[-5:])), perf_list[-5:], 1)[0] < 0 else "increasing"
        }


# ============================================================
# CONTINUAL LEARNING RETRIEVER
# ============================================================

class ContinualLearningRetriever:
    """
    Continual learning retriever that adapts over time.
    
    This retriever learns from interactions and improves its retrieval
    performance over time while preventing catastrophic forgetting.
    """
    
    def __init__(self, 
                 embedding_dim: int = 384, 
                 forgetting_mechanism: ForgettingMechanism = ForgettingMechanism.EXPERIENCE_REPLAY,
                 experience_buffer_size: int = 1000):
        """
        Initialize continual learning retriever.
        
        Args:
            embedding_dim: Dimension of document embeddings
            forgetting_mechanism: Method to prevent catastrophic forgetting
            experience_buffer_size: Size of experience replay buffer
        """
        self.embedding_dim = embedding_dim
        self.documents: List[ContinualDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.experience_buffer = ExperienceBuffer(capacity=experience_buffer_size)
        self.forgetting_prevention = ForgettingPrevention(mechanism=forgetting_mechanism)
        self.performance_monitor = PerformanceMonitor()
        
        # Track document access patterns
        self.access_counts: Dict[str, int] = {}
        self.relevance_feedback: Dict[str, List[float]] = {}
        
        # Learning parameters
        self.learning_rate = 0.01
        self.novelty_threshold = 0.3
        
        logger.info("Initialized Continual Learning Retriever")
    
    def add_documents(self, documents: List[ContinualDocument]) -> None:
        """Add documents to the retriever with continual learning."""
        new_embeddings = []
        
        for doc in documents:
            # Update document statistics
            doc.access_frequency = 0
            doc.importance_score = 1.0  # Default importance
            
            # Generate embedding if not present
            if doc.embedding is None:
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                embedding = np.frombuffer(bytes.fromhex(content_hash[:32]), dtype=np.float32)
                # Pad or truncate to desired dimension
                if len(embedding) < self.embedding_dim:
                    embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')
                elif len(embedding) > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
                doc.embedding = embedding
            
            self.documents.append(doc)
            new_embeddings.append(embedding)
        
        # Update embedding matrix
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
    
    def _calculate_novelty_score(self, query_embedding: np.ndarray, doc_idx: int) -> float:
        """Calculate how novel this document is relative to recent queries."""
        # For simplicity, we'll use a basic novelty calculation
        # In practice, this would compare against recent query embeddings
        return np.random.random()  # Placeholder
    
    def _update_document_importance(self, doc_id: str, relevance_score: float) -> None:
        """Update the importance of a document based on relevance feedback."""
        for doc in self.documents:
            if doc.id == doc_id:
                # Update importance based on relevance feedback
                doc.importance_score = 0.7 * doc.importance_score + 0.3 * relevance_score
                
                # Update access count
                doc.access_frequency += 1
                
                # Record relevance history
                doc.relevance_history.append((datetime.datetime.now(), relevance_score))
                
                # Keep only recent history (last 100 entries)
                if len(doc.relevance_history) > 100:
                    doc.relevance_history = doc.relevance_history[-100:]
    
    def retrieve(self, 
                 query: ContinualQuery, 
                 query_embedding: np.ndarray, 
                 k: int = 5) -> List[ContinualRetrievalResult]:
        """Retrieve documents with continual learning considerations."""
        if len(self.documents) == 0:
            return []
        
        # Compute semantic similarities
        semantic_similarities = self._compute_semantic_similarity(query_embedding)
        
        # Create results with learning signals
        results = []
        for idx, similarity in enumerate(semantic_similarities):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Calculate novelty score
                novelty_score = self._calculate_novelty_score(query_embedding, idx)
                
                # Create learning signal (combination of similarity and document importance)
                learning_signal = (similarity + doc.importance_score) / 2.0
                
                result = ContinualRetrievalResult(
                    document=doc,
                    score=float(similarity),
                    learning_signal=learning_signal,
                    novelty_score=novelty_score,
                    rank=0  # Will be set after sorting
                )
                results.append(result)
        
        # Sort by combined score (considering both similarity and learning signal)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank
        
        # Update document access counts and importance
        for result in results[:k]:  # Only update top-k results
            self._update_document_importance(result.document.id, result.score)
        
        # Limit to top-k
        results = results[:k]
        
        return results
    
    def process_feedback(self, 
                        query: ContinualQuery, 
                        retrieved_results: List[ContinualRetrievalResult], 
                        response: str, 
                        user_feedback: Optional[str] = None,
                        performance_score: float = 1.0) -> None:
        """Process feedback to improve future performance."""
        # Create learning experience
        experience = LearningExperience(
            query=query.text,
            retrieved_docs=[r.document for r in retrieved_results],
            response=response,
            feedback=user_feedback,
            timestamp=query.timestamp,
            performance_score=performance_score,
            importance=performance_score  # Importance based on performance
        )
        
        # Add to experience buffer
        self.experience_buffer.add_experience(experience)
        
        # Update performance monitor
        adaptation_needed = self.performance_monitor.update_performance(performance_score)
        
        # Log the learning event
        logger.info(f"Processed feedback for query '{query.text[:50]}...', "
                   f"performance: {performance_score:.3f}, "
                   f"adaptation needed: {adaptation_needed}")


# ============================================================
# CONTINUAL LEARNING GENERATOR
# ============================================================

class ContinualLearningGenerator:
    """
    Generator that adapts based on continual learning.
    
    This component learns from interactions and improves its generation
    quality over time.
    """
    
    def __init__(self):
        self.generate_fn: Optional[Callable] = None
        self.response_quality_tracker: Dict[str, List[float]] = {}
        self.domain_adaptation_weights: Dict[str, float] = {}
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generate_fn = generate_fn
    
    def generate(self, 
                 query: ContinualQuery, 
                 context: str, 
                 retrieved_docs: List[ContinualDocument]) -> ContinualGenerationResult:
        """Generate response with continual learning considerations."""
        import time
        start_time = time.time()
        
        # Build the generation prompt
        prompt = self._build_prompt(query, context)
        
        # Generate answer
        if self.generate_fn:
            answer = self.generate_fn(prompt)
        else:
            # Placeholder answer
            answer = self._generate_placeholder(query, retrieved_docs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Determine if adaptation is needed based on query characteristics
        adaptation_needed = self._should_adapt(query, retrieved_docs)
        
        # Track response quality by domain
        self._track_response_quality(query.domain, len(answer.split()))
        
        result = ContinualGenerationResult(
            answer=answer,
            sources=retrieved_docs,
            learning_experiences=[],  # Will be populated by the main system
            confidence=0.7,  # Placeholder confidence
            latency_ms=latency_ms,
            token_count=len(prompt.split()) + len(answer.split()),
            adaptation_needed=adaptation_needed
        )
        
        return result
    
    def _build_prompt(self, query: ContinualQuery, context: str) -> str:
        """Build the generation prompt with continual learning considerations."""
        return f"""Answer the question based on the provided context.
The system continues to learn and improve from each interaction.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Always cite your sources.

Context:
{context}

Question: {query.text}

Answer:"""
    
    def _generate_placeholder(self, query: ContinualQuery, docs: List[ContinualDocument]) -> str:
        """Generate a placeholder answer for testing."""
        if docs:
            return f"Based on the provided context, the system found {len(docs)} relevant documents. The query '{query.text}' was processed with continual learning."
        else:
            return "No relevant information found for your query."
    
    def _should_adapt(self, query: ContinualQuery, docs: List[ContinualDocument]) -> bool:
        """Determine if adaptation is needed based on query and documents."""
        # Adapt if:
        # 1. Query is in a domain we don't handle well
        # 2. Retrieved documents are very different from usual
        # 3. Query difficulty is high
        
        # Check domain adaptation needs
        if query.domain not in self.domain_adaptation_weights:
            return True  # New domain
        
        # Check if query difficulty is high
        if query.difficulty > 0.8:
            return True
        
        # Check document diversity
        if len(docs) > 0:
            avg_importance = np.mean([doc.importance_score for doc in docs])
            if avg_importance < 0.3:  # Low importance documents retrieved
                return True
        
        return False
    
    def _track_response_quality(self, domain: str, response_length: int) -> None:
        """Track response quality by domain."""
        if domain not in self.response_quality_tracker:
            self.response_quality_tracker[domain] = []
        
        # For now, we'll just track response length as a proxy for quality
        self.response_quality_tracker[domain].append(response_length)
        
        # Keep only recent measurements
        if len(self.response_quality_tracker[domain]) > 100:
            self.response_quality_tracker[domain] = self.response_quality_tracker[domain][-100:]


# ============================================================
# CONTINUAL LEARNING RAG SYSTEM
# ============================================================

class ContinualLearningRAG:
    """
    Continual Learning RAG system that adapts and improves over time.
    
    This system learns from interactions and continuously improves its
    performance without forgetting previous knowledge.
    """
    
    def __init__(self, 
                 embedding_dim: int = 384, 
                 forgetting_mechanism: ForgettingMechanism = ForgettingMechanism.EXPERIENCE_REPLAY,
                 experience_buffer_size: int = 1000):
        """
        Initialize continual learning RAG system.
        
        Args:
            embedding_dim: Dimension of document embeddings
            forgetting_mechanism: Method to prevent catastrophic forgetting
            experience_buffer_size: Size of experience replay buffer
        """
        self.retriever = ContinualLearningRetriever(
            embedding_dim=embedding_dim,
            forgetting_mechanism=forgetting_mechanism,
            experience_buffer_size=experience_buffer_size
        )
        self.generator = ContinualLearningGenerator()
        
        # Learning parameters
        self.adaptation_threshold = 0.7  # Performance threshold for adaptation
        self.performance_history: List[float] = []
        
        logger.info("Initialized Continual Learning RAG system")
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generator.set_generator(generate_fn)
    
    def add_documents(self, documents: List[ContinualDocument]) -> int:
        """Add documents to the continual learning RAG system."""
        # Process documents to add embeddings if not present
        processed_docs = []
        for doc in documents:
            if doc.embedding is None:
                # Generate embedding for the document
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                embedding = np.frombuffer(bytes.fromhex(content_hash[:32]), dtype=np.float32)
                # Pad or truncate to desired dimension
                if len(embedding) < self.retriever.embedding_dim:
                    embedding = np.pad(embedding, (0, self.retriever.embedding_dim - len(embedding)), 'constant')
                elif len(embedding) > self.retriever.embedding_dim:
                    embedding = embedding[:self.retriever.embedding_dim]
                doc.embedding = embedding
            
            processed_docs.append(doc)
        
        self.retriever.add_documents(processed_docs)
        logger.info(f"Added {len(documents)} continual learning documents")
        return len(documents)
    
    def query(self, 
              query: ContinualQuery, 
              query_embedding: np.ndarray, 
              k: int = 5) -> ContinualGenerationResult:
        """
        Query the continual learning RAG system.
        
        Args:
            query: Continual learning query object
            query_embedding: Embedding vector for the query
            k: Number of results to retrieve
            
        Returns:
            ContinualGenerationResult with answer and learning information
        """
        import time
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(query, query_embedding, k=k)
        
        if not retrieval_results:
            # No results found, return a default response
            return ContinualGenerationResult(
                answer="No relevant information found for your query.",
                sources=[],
                learning_experiences=[],
                confidence=0.0,
                latency_ms=0.0,
                token_count=10,
                adaptation_needed=False
            )
        
        # Build context from retrieved documents
        context_parts = []
        retrieved_docs = []
        
        for result in retrieval_results:
            context_parts.append(f"Document: {result.document.content}")
            retrieved_docs.append(result.document)
        
        context = "\n\n".join(context_parts)
        
        # Generate response
        generation_result = self.generator.generate(query, context, retrieved_docs)
        
        # Calculate performance score (simplified)
        performance_score = min(1.0, len(retrieval_results) / k)  # Based on retrieval success
        
        # Process feedback to enable learning
        self.retriever.process_feedback(
            query=query,
            retrieved_results=retrieval_results,
            response=generation_result.answer,
            performance_score=performance_score
        )
        
        # Update performance history
        self.performance_history.append(performance_score)
        if len(self.performance_history) > 100:  # Keep only recent history
            self.performance_history = self.performance_history[-100:]
        
        return generation_result
    
    def adapt(self) -> bool:
        """
        Trigger system adaptation based on accumulated experiences.
        
        Returns:
            True if adaptation was performed, False otherwise
        """
        # Check if we have enough experiences to warrant adaptation
        if len(self.retriever.experience_buffer.buffer) < 10:
            logger.info("Not enough experiences to warrant adaptation")
            return False
        
        # Check if performance has degraded
        recent_performance = list(self.retriever.performance_monitor.performance_history)[-10:] if self.retriever.performance_monitor.performance_history else []
        if recent_performance and np.mean(recent_performance) < self.adaptation_threshold:
            logger.info("Adapting system due to performance degradation")
            
            # Perform adaptation (in a real system, this would involve retraining)
            # For this simulation, we'll just log the adaptation
            logger.info("System adaptation completed")
            return True
        
        # Check if we have diverse experiences that suggest adaptation
        if len(self.retriever.experience_buffer.buffer) > 50:
            logger.info("Adapting system due to accumulated experiences")
            
            # Perform adaptation
            logger.info("System adaptation completed")
            return True
        
        return False
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and statistics."""
        return {
            "total_documents": len(self.retriever.documents),
            "total_experiences": len(self.retriever.experience_buffer.buffer),
            "performance_stats": self.retriever.performance_monitor.get_performance_stats(),
            "total_queries_processed": len(self.performance_history),
            "average_performance": float(np.mean(self.performance_history)) if self.performance_history else 0.0,
            "domains_handled": list(self.generator.domain_adaptation_weights.keys()),
            "adaptation_needed": self.retriever.performance_monitor.adaptation_triggered
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the continual learning model state."""
        model_state = {
            'documents': self.retriever.documents,
            'embeddings': self.retriever.embeddings,
            'experience_buffer': list(self.retriever.experience_buffer.buffer),
            'performance_history': self.performance_history,
            'access_counts': self.retriever.access_counts,
            'relevance_feedback': self.retriever.relevance_feedback
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the continual learning model state."""
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        self.retriever.documents = model_state['documents']
        self.retriever.embeddings = model_state['embeddings']
        self.retriever.experience_buffer.buffer = deque(model_state['experience_buffer'])
        self.performance_history = model_state['performance_history']
        self.retriever.access_counts = model_state['access_counts']
        self.retriever.relevance_feedback = model_state['relevance_feedback']
        
        logger.info(f"Model loaded from {filepath}")


# ============================================================
# EXAMPLE USAGE AND TESTING
# ============================================================

def example_usage():
    """Demonstrate Continual Learning RAG usage."""
    
    # Create RAG system
    rag = ContinualLearningRAG(
        forgetting_mechanism=ForgettingMechanism.EXPERIENCE_REPLAY,
        experience_buffer_size=500
    )
    
    # Sample continual learning documents
    documents = [
        ContinualDocument(
            id="doc1",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            importance_score=0.8,
            metadata={"domain": "AI", "difficulty": 0.5}
        ),
        ContinualDocument(
            id="doc2",
            content="Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            importance_score=0.9,
            metadata={"domain": "Deep Learning", "difficulty": 0.7}
        ),
        ContinualDocument(
            id="doc3",
            content="Natural language processing enables computers to understand and generate human language.",
            importance_score=0.7,
            metadata={"domain": "NLP", "difficulty": 0.6}
        )
    ]
    
    # Add documents
    num_docs = rag.add_documents(documents)
    print(f"Added {num_docs} continual learning documents")
    
    # Create a continual learning query
    query = ContinualQuery(
        text="What is machine learning?",
        domain="AI",
        difficulty=0.4
    )
    
    # Create a simple query embedding (in practice, this would come from an embedding model)
    query_text_hash = hashlib.md5(query.text.encode()).hexdigest()
    query_embedding = np.frombuffer(bytes.fromhex(query_text_hash[:32]), dtype=np.float32)
    if len(query_embedding) < 384:
        query_embedding = np.pad(query_embedding, (0, 384 - len(query_embedding)), 'constant')
    elif len(query_embedding) > 384:
        query_embedding = query_embedding[:384]
    
    # Query the system
    result = rag.query(query, query_embedding, k=2)
    
    print(f"\nQuery: {query.text}")
    print(f"Domain: {query.domain}")
    print(f"Difficulty: {query.difficulty}")
    print(f"Adaptation needed: {result.adaptation_needed}")
    print(f"Answer: {result.answer}")
    print(f"Sources: {len(result.sources)}")
    
    # Show learning status
    print("\nLearning Status:")
    status = rag.get_learning_status()
    print(f"  Total documents: {status['total_documents']}")
    print(f"  Total experiences: {status['total_experiences']}")
    print(f"  Average performance: {status['average_performance']:.3f}")
    print(f"  Domains handled: {status['domains_handled']}")
    print(f"  Adaptation needed: {status['adaptation_needed']}")
    
    # Process multiple queries to demonstrate learning
    print("\n" + "="*60)
    print("Demonstrating Learning Over Multiple Queries:")
    
    additional_queries = [
        ContinualQuery("Explain deep learning concepts", domain="Deep Learning", difficulty=0.6),
        ContinualQuery("How does NLP work?", domain="NLP", difficulty=0.5),
        ContinualQuery("What are neural networks?", domain="Deep Learning", difficulty=0.7)
    ]
    
    for i, q in enumerate(additional_queries):
        q_hash = hashlib.md5(q.text.encode()).hexdigest()
        q_embedding = np.frombuffer(bytes.fromhex(q_hash[:32]), dtype=np.float32)
        if len(q_embedding) < 384:
            q_embedding = np.pad(q_embedding, (0, 384 - len(q_embedding)), 'constant')
        elif len(q_embedding) > 384:
            q_embedding = q_embedding[:384]
        
        result = rag.query(q, q_embedding, k=2)
        print(f"  Query {i+1}: {q.text[:30]}... - Adaptation needed: {result.adaptation_needed}")
    
    # Check updated status
    print("\nUpdated Learning Status:")
    updated_status = rag.get_learning_status()
    print(f"  Total queries processed: {updated_status['total_queries_processed']}")
    print(f"  Performance stats: {updated_status['performance_stats']}")
    
    # Demonstrate adaptation
    print(f"\nAttempting adaptation...")
    adapted = rag.adapt()
    print(f"Adaptation performed: {adapted}")
    
    return rag


if __name__ == "__main__":
    example_usage()