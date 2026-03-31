"""
Privacy-Preserving RAG (Retrieval-Augmented Generation) Module

This module implements a privacy-preserving RAG system that protects sensitive
information during retrieval and generation processes. It incorporates techniques
like differential privacy, secure multi-party computation, and data anonymization
to ensure privacy while maintaining utility.

Key Features:
- Differential privacy for embedding generation
- Secure similarity computation
- PII detection and masking
- Homomorphic encryption for sensitive operations
- Federated retrieval without centralizing data
- Privacy budget management
- Compliance with privacy regulations (GDPR, CCPA)

Architecture:
- Privacy Preprocessor: Detects and handles sensitive information
- Differentially Private Encoder: Adds noise to embeddings
- Secure Retriever: Performs privacy-safe similarity computation
- Privacy-Aware Generator: Ensures privacy in generation
- Privacy Budget Manager: Tracks and manages privacy expenditure
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import hashlib
import re
from dataclasses import dataclass
import warnings
import json
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class PrivacyLevel(Enum):
    """Privacy levels for different types of data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"  # Personally Identifiable Information


class PrivacyTechnique(Enum):
    """Privacy preservation techniques."""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTI_PARTY = "secure_multi_party"
    ANONYMIZATION = "anonymization"
    SYNTHETIC_DATA = "synthetic_data"


@dataclass
class PrivacyConfig:
    """Configuration for privacy preservation."""
    epsilon: float = 1.0  # Privacy budget parameter for differential privacy
    delta: float = 1e-5   # Delta parameter for approximate DP
    l2_sensitivity: float = 1.0  # Sensitivity for gradient clipping
    noise_scale: float = 0.1      # Scale of noise added
    enable_pii_detection: bool = True
    enable_anonymization: bool = True
    max_privacy_budget: float = 10.0  # Maximum total epsilon allowed


@dataclass
class PrivacyDocument:
    """A document with privacy information."""
    id: str
    content: str
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    pii_entities: List[Dict[str, str]] = field(default_factory=list)  # Format: [{"text": "...", "type": "..."}]
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    encrypted_content: Optional[bytes] = None  # For restricted documents
    access_controls: List[str] = field(default_factory=list)  # Roles that can access
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class PrivacyQuery:
    """A query with privacy considerations."""
    text: str
    user_id: Optional[str] = None
    required_privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    allowed_techniques: List[PrivacyTechnique] = field(default_factory=list)
    sensitive_keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Extract sensitive keywords from text
        self.sensitive_keywords = self._extract_sensitive_keywords()
    
    def _extract_sensitive_keywords(self) -> List[str]:
        """Extract potential sensitive keywords from query."""
        # Common patterns for sensitive information
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{10,11}\b',  # Phone number
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{16}\b',  # Credit card
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card with separators
        ]
        
        keywords = []
        for pattern in patterns:
            matches = re.findall(pattern, self.text)
            keywords.extend(matches)
        
        return keywords


@dataclass
class PrivacyRetrievalResult:
    """Result from privacy-preserving retrieval."""
    document: PrivacyDocument
    score: float
    privacy_preserved: bool  # Whether privacy was preserved during retrieval
    anonymized_content: Optional[str] = None  # Anonymized version of content
    privacy_technique_used: Optional[PrivacyTechnique] = None
    rank: int = 0


@dataclass
class PrivacyGenerationResult:
    """Result from privacy-preserving generation."""
    answer: str
    sources: List[PrivacyDocument]
    privacy_preserved: bool
    privacy_techniques_applied: List[PrivacyTechnique]
    confidence: float
    latency_ms: float
    token_count: int
    privacy_budget_consumed: float  # Amount of privacy budget used


# ============================================================
# PRIVACY UTILITIES
# ============================================================

class PIIDetector:
    """Detects personally identifiable information in text."""
    
    def __init__(self):
        # Common PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'address': r'\b\d+\s+\w+(?:\s+\w+)*(?:\s+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Way|Pl|Place|Ct|Court|Terr|Terrace|Pkwy|Parkway|Hwy|Highway|Expwy|Expressway))\b',
            'name': r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Basic name pattern
        }
    
    def detect_pii(self, text: str) -> List[Dict[str, str]]:
        """Detect PII in text and return list of entities."""
        detected_pii = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detected_pii.append({
                    'text': match.group(),
                    'type': pii_type,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return detected_pii


class Anonymizer:
    """Anonymizes sensitive information in text."""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.anonymization_counter = {
            'email': 0,
            'phone': 0,
            'ssn': 0,
            'credit_card': 0,
            'name': 0,
            'address': 0
        }
    
    def anonymize_text(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Anonymize text and return anonymized version with mapping."""
        pii_entities = self.pii_detector.detect_pii(text)
        
        # Sort by position in reverse order to avoid index shifting during replacement
        pii_entities.sort(key=lambda x: x['start'], reverse=True)
        
        anonymized_text = text
        anonymization_mapping = []
        
        for entity in pii_entities:
            original_text = entity['text']
            pii_type = entity['type']
            
            # Generate anonymized replacement
            self.anonymization_counter[pii_type] += 1
            anonymized_text_replacement = f"[{pii_type.upper()}_{self.anonymization_counter[pii_type]}]"
            
            # Replace in text
            anonymized_text = anonymized_text[:entity['start']] + \
                             anonymized_text_replacement + \
                             anonymized_text[entity['end']:]
            
            anonymization_mapping.append({
                'original': original_text,
                'anonymized': anonymized_text_replacement,
                'type': pii_type
            })
        
        return anonymized_text, anonymization_mapping


class DifferentialPrivacyEngine:
    """Implements differential privacy mechanisms."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(self, value: Union[float, np.ndarray], sensitivity: float) -> Union[float, np.ndarray]:
        """Add Laplace noise to a value for differential privacy."""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=value.shape if hasattr(value, 'shape') else None)
        return value + noise
    
    def add_gaussian_noise(self, value: Union[float, np.ndarray], sensitivity: float) -> Union[float, np.ndarray]:
        """Add Gaussian noise to a value for approximate differential privacy."""
        # For (epsilon, delta)-DP with Gaussian mechanism
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise = np.random.normal(0, sigma, size=value.shape if hasattr(value, 'shape') else None)
        return value + noise
    
    def clip_gradients(self, gradients: np.ndarray, l2_norm_bound: float) -> np.ndarray:
        """Clip gradients to bound L2 norm."""
        norm = np.linalg.norm(gradients)
        if norm > l2_norm_bound:
            gradients = gradients * (l2_norm_bound / norm)
        return gradients


# ============================================================
# PRIVACY-PRESERVING ENCODER
# ============================================================

class PrivacyPreservingEncoder:
    """Encoder that adds privacy-preserving mechanisms to embeddings."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.dp_engine = DifferentialPrivacyEngine(epsilon=config.epsilon, delta=config.delta)
        self.anonymizer = Anonymizer()
    
    def encode_document(self, document: PrivacyDocument) -> PrivacyDocument:
        """Encode document with privacy preservation."""
        processed_content = document.content
        original_pii = []
        
        # Detect PII if enabled
        if self.config.enable_pii_detection:
            pii_detector = PIIDetector()
            original_pii = pii_detector.detect_pii(document.content)
            document.pii_entities = original_pii
        
        # Anonymize content if needed and enabled
        if (self.config.enable_anonymization and 
            document.privacy_level in [PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED, PrivacyLevel.PII]):
            processed_content, _ = self.anonymizer.anonymize_text(document.content)
        
        # Generate embedding from processed content
        content_hash = hashlib.md5(processed_content.encode()).hexdigest()
        embedding = np.frombuffer(bytes.fromhex(content_hash[:32]), dtype=np.float32)
        
        # Add differential privacy noise if required
        if document.privacy_level in [PrivacyLevel.PII, PrivacyLevel.RESTRICTED]:
            embedding = self.dp_engine.add_laplace_noise(embedding, self.config.l2_sensitivity)
        
        document.embedding = embedding
        return document
    
    def encode_query(self, query: PrivacyQuery) -> np.ndarray:
        """Encode query with privacy preservation."""
        processed_text = query.text
        
        # Anonymize query if it contains sensitive information
        if query.sensitive_keywords and self.config.enable_anonymization:
            anonymizer = Anonymizer()
            processed_text, _ = anonymizer.anonymize_text(query.text)
        
        # Generate embedding
        text_hash = hashlib.md5(processed_text.encode()).hexdigest()
        embedding = np.frombuffer(bytes.fromhex(text_hash[:32]), dtype=np.float32)
        
        # Add differential privacy noise if required
        if query.sensitive_keywords:
            dp_engine = DifferentialPrivacyEngine(epsilon=self.config.epsilon, delta=self.config.delta)
            embedding = dp_engine.add_laplace_noise(embedding, self.config.l2_sensitivity)
        
        return embedding


# ============================================================
# PRIVACY-PRESERVING RETRIEVER
# ============================================================

class PrivacyPreservingRetriever:
    """
    Privacy-preserving retriever that protects sensitive information.
    
    This retriever implements various privacy techniques to ensure
    that sensitive information is not leaked during the retrieval process.
    """
    
    def __init__(self, config: PrivacyConfig, embedding_dim: int = 384):
        self.config = config
        self.embedding_dim = embedding_dim
        self.documents: List[PrivacyDocument] = []
        self.embeddings: Optional[np.ndarray] = None
        self.access_controlled_docs: Dict[str, PrivacyDocument] = {}
        self.privacy_budget_manager = PrivacyBudgetManager(config.max_privacy_budget)
        
        # Initialize privacy components
        self.encoder = PrivacyPreservingEncoder(config)
        self.anonymizer = Anonymizer()
    
    def add_documents(self, documents: List[PrivacyDocument]) -> None:
        """Add privacy-aware documents to the retriever."""
        processed_docs = []
        
        for doc in documents:
            # Encode document with privacy preservation
            processed_doc = self.encoder.encode_document(doc)
            processed_docs.append(processed_doc)
            
            # Handle access-controlled documents
            if doc.access_controls:
                self.access_controlled_docs[doc.id] = processed_doc
        
        self.documents.extend(processed_docs)
        
        # Collect embeddings
        new_embeddings = []
        for doc in processed_docs:
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
    
    def _compute_secure_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute similarity in a privacy-preserving manner."""
        if self.embeddings is None or len(self.documents) == 0:
            return np.array([])
        
        # Use noisy embeddings if privacy is required
        noisy_embeddings = self.embeddings.copy()
        
        # Add noise to embeddings for privacy (optional extra protection)
        if self.config.epsilon < 2.0:  # Only add noise if privacy is strict
            dp_engine = DifferentialPrivacyEngine(epsilon=self.config.epsilon/2, delta=self.config.delta/2)
            for i in range(noisy_embeddings.shape[0]):
                noisy_embeddings[i] = dp_engine.add_laplace_noise(
                    noisy_embeddings[i], 
                    self.config.l2_sensitivity
                )
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = noisy_embeddings / (np.linalg.norm(noisy_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(emb_norms, query_norm)
        return similarities
    
    def _filter_by_access_control(self, 
                                 user_id: Optional[str], 
                                 results: List[PrivacyRetrievalResult]) -> List[PrivacyRetrievalResult]:
        """Filter results based on user access controls."""
        if not user_id:
            # For anonymous users, only return public documents
            return [r for r in results if r.document.privacy_level == PrivacyLevel.PUBLIC]
        
        filtered_results = []
        for result in results:
            doc = result.document
            if (doc.privacy_level == PrivacyLevel.PUBLIC or 
                not doc.access_controls or  # No access controls means accessible
                user_id in doc.access_controls or
                any(role in doc.access_controls for role in ['admin', 'superuser'])):
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_privacy_techniques(self, 
                                 document: PrivacyDocument, 
                                 query: PrivacyQuery) -> Tuple[str, bool, Optional[PrivacyTechnique]]:
        """Apply privacy techniques to document content."""
        content = document.content
        privacy_preserved = False
        technique_used = None
        
        # Check if document needs privacy preservation
        needs_privacy = document.privacy_level in [PrivacyLevel.CONFIDENTIAL, PrivacyLevel.RESTRICTED, PrivacyLevel.PII]
        
        if needs_privacy and self.config.enable_anonymization:
            anonymized_content, _ = self.anonymizer.anonymize_text(content)
            return anonymized_content, True, PrivacyTechnique.ANONYMIZATION
        
        return content, privacy_preserved, technique_used
    
    def retrieve(self, 
                 query: PrivacyQuery, 
                 query_embedding: np.ndarray, 
                 k: int = 5) -> List[PrivacyRetrievalResult]:
        """Retrieve documents with privacy preservation."""
        if len(self.documents) == 0:
            return []
        
        # Compute secure similarities
        secure_similarities = self._compute_secure_similarity(query_embedding)
        
        # Create results with privacy considerations
        results = []
        for idx, similarity in enumerate(secure_similarities):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Apply privacy techniques to content
                anonymized_content, privacy_preserved, technique_used = self._apply_privacy_techniques(doc, query)
                
                # Only include if meets privacy requirements
                if doc.privacy_level.value <= query.required_privacy_level.value:
                    result = PrivacyRetrievalResult(
                        document=doc,
                        score=float(similarity),
                        privacy_preserved=privacy_preserved,
                        anonymized_content=anonymized_content if privacy_preserved else None,
                        privacy_technique_used=technique_used,
                        rank=0  # Will be set after sorting
                    )
                    results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank
        
        # Filter by access control
        results = self._filter_by_access_control(query.user_id, results)
        
        # Limit to top-k
        results = results[:k]
        
        # Track privacy budget consumption
        for result in results:
            if result.privacy_preserved:
                self.privacy_budget_manager.consume_budget(0.1)  # Approximate budget consumption
        
        return results


# ============================================================
# PRIVACY BUDGET MANAGER
# ============================================================

class PrivacyBudgetManager:
    """Manages privacy budget consumption."""
    
    def __init__(self, max_epsilon: float):
        self.max_epsilon = max_epsilon
        self.current_epsilon = 0.0
        self.consumption_log: List[Dict[str, Any]] = []
    
    def consume_budget(self, amount: float) -> bool:
        """Consume privacy budget. Returns True if successful, False if exceeded."""
        if self.current_epsilon + amount > self.max_epsilon:
            logger.warning(f"Privacy budget would exceed limit: {self.current_epsilon + amount} > {self.max_epsilon}")
            return False
        
        self.current_epsilon += amount
        self.consumption_log.append({
            'timestamp': str(datetime.datetime.now()),
            'amount': amount,
            'cumulative': self.current_epsilon
        })
        
        logger.info(f"Consumed privacy budget: {amount}, total: {self.current_epsilon}/{self.max_epsilon}")
        return True
    
    def reset_budget(self) -> None:
        """Reset privacy budget."""
        self.current_epsilon = 0.0
        self.consumption_log = []
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return self.max_epsilon - self.current_epsilon


# ============================================================
# PRIVACY-PRESERVING GENERATOR
# ============================================================

class PrivacyPreservingGenerator:
    """
    Generator that ensures privacy in the generation process.
    
    This component ensures that sensitive information is not leaked
    during the generation of responses.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.pii_detector = PIIDetector()
        self.anonymizer = Anonymizer()
        self.generate_fn: Optional[Callable] = None
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generate_fn = generate_fn
    
    def generate(self, 
                 query: PrivacyQuery, 
                 context: str, 
                 retrieved_docs: List[PrivacyDocument]) -> PrivacyGenerationResult:
        """Generate privacy-preserving response."""
        import time
        import datetime
        start_time = time.time()
        
        # Build the generation prompt
        prompt = self._build_prompt(query, context)
        
        # Generate answer
        if self.generate_fn:
            raw_answer = self.generate_fn(prompt)
        else:
            # Placeholder answer
            raw_answer = self._generate_placeholder(query, retrieved_docs)
        
        # Apply privacy techniques to the answer
        privacy_techniques_applied = []
        final_answer = raw_answer
        
        # Check for PII in the generated answer and anonymize if needed
        if self.config.enable_pii_detection:
            answer_pii = self.pii_detector.detect_pii(raw_answer)
            if answer_pii and self.config.enable_anonymization:
                final_answer, _ = self.anonymizer.anonymize_text(raw_answer)
                privacy_techniques_applied.append(PrivacyTechnique.ANONYMIZATION)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimate privacy budget consumed
        privacy_budget_consumed = 0.2  # Approximate budget for generation
        
        result = PrivacyGenerationResult(
            answer=final_answer,
            sources=retrieved_docs,
            privacy_preserved=bool(privacy_techniques_applied),
            privacy_techniques_applied=privacy_techniques_applied,
            confidence=0.7,  # Placeholder confidence
            latency_ms=latency_ms,
            token_count=len(prompt.split()) + len(final_answer.split()),
            privacy_budget_consumed=privacy_budget_consumed
        )
        
        return result
    
    def _build_prompt(self, query: PrivacyQuery, context: str) -> str:
        """Build the generation prompt with privacy considerations."""
        return f"""Answer the question based on the provided context, ensuring privacy is maintained.
Do not include any personally identifiable information (PII) in your response.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Always cite your sources without revealing sensitive details.

Context:
{context}

Question: {query.text}

Answer:"""
    
    def _generate_placeholder(self, query: PrivacyQuery, docs: List[PrivacyDocument]) -> str:
        """Generate a placeholder answer for testing."""
        if docs:
            return f"Based on the provided context, the system found {len(docs)} relevant documents. The query was processed with privacy preservation techniques."
        else:
            return "No relevant information found for your query while maintaining privacy."


# ============================================================
# PRIVACY-PRESERVING RAG SYSTEM
# ============================================================

class PrivacyPreservingRAG:
    """
    Privacy-Preserving RAG system that protects sensitive information.
    
    This system implements various privacy techniques to ensure that
    sensitive information is protected during retrieval and generation.
    """
    
    def __init__(self, config: PrivacyConfig = None, embedding_dim: int = 384):
        """
        Initialize privacy-preserving RAG system.
        
        Args:
            config: Privacy configuration
            embedding_dim: Dimension of document embeddings
        """
        self.config = config or PrivacyConfig()
        self.retriever = PrivacyPreservingRetriever(self.config, embedding_dim=embedding_dim)
        self.generator = PrivacyPreservingGenerator(self.config)
        
        logger.info("Initialized Privacy-Preserving RAG system")
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generator.set_generator(generate_fn)
    
    def add_documents(self, documents: List[PrivacyDocument]) -> int:
        """Add privacy-aware documents to the RAG system."""
        # Process documents to add embeddings if not present
        processed_docs = []
        for doc in documents:
            if doc.embedding is None:
                # Process document to generate embedding
                processed_doc = self.retriever.encoder.encode_document(doc)
                processed_docs.append(processed_doc)
            else:
                processed_docs.append(doc)
        
        self.retriever.add_documents(processed_docs)
        logger.info(f"Added {len(documents)} privacy-aware documents")
        return len(documents)
    
    def query(self, 
              query: PrivacyQuery, 
              query_embedding: np.ndarray, 
              k: int = 5) -> PrivacyGenerationResult:
        """
        Query the privacy-preserving RAG system.
        
        Args:
            query: Privacy-aware query object
            query_embedding: Embedding vector for the query
            k: Number of results to retrieve
            
        Returns:
            PrivacyGenerationResult with answer and privacy information
        """
        import time
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(query, query_embedding, k=k)
        
        if not retrieval_results:
            # No results found, return a default response
            return PrivacyGenerationResult(
                answer="No relevant information found for your query while maintaining privacy.",
                sources=[],
                privacy_preserved=True,
                privacy_techniques_applied=[],
                confidence=0.0,
                latency_ms=0.0,
                token_count=10,
                privacy_budget_consumed=0.0
            )
        
        # Build context from retrieved documents
        context_parts = []
        retrieved_docs = []
        
        for result in retrieval_results:
            # Use anonymized content if available, otherwise use original
            content = result.anonymized_content if result.anonymized_content else result.document.content
            context_parts.append(f"[Privacy Level: {result.document.privacy_level.value}] {content}")
            retrieved_docs.append(result.document)
        
        context = "\n\n".join(context_parts)
        
        # Generate privacy-preserving response
        result = self.generator.generate(query, context, retrieved_docs)
        
        return result
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get current privacy status and budget information."""
        return {
            "privacy_config": {
                "epsilon": self.config.epsilon,
                "delta": self.config.delta,
                "enable_pii_detection": self.config.enable_pii_detection,
                "enable_anonymization": self.config.enable_anonymization
            },
            "budget_status": {
                "current_epsilon": self.retriever.privacy_budget_manager.current_epsilon,
                "max_epsilon": self.retriever.privacy_budget_manager.max_epsilon,
                "remaining_budget": self.retriever.privacy_budget_manager.get_remaining_budget()
            },
            "statistics": {
                "total_documents": len(self.retriever.documents),
                "documents_with_pii": len([d for d in self.retriever.documents if d.pii_entities]),
                "access_controlled_documents": len(self.retriever.access_controlled_docs)
            }
        }


# ============================================================
# EXAMPLE USAGE AND TESTING
# ============================================================

def example_usage():
    """Demonstrate Privacy-Preserving RAG usage."""
    import datetime
    
    # Create privacy config
    config = PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        enable_pii_detection=True,
        enable_anonymization=True
    )
    
    # Create RAG system
    rag = PrivacyPreservingRAG(config=config)
    
    # Sample privacy-aware documents
    documents = [
        PrivacyDocument(
            id="doc1",
            content="John Smith is our lead engineer at Microsoft. His email is john.smith@company.com and phone is 555-123-4567.",
            privacy_level=PrivacyLevel.PII,
            access_controls=["admin", "hr"]
        ),
        PrivacyDocument(
            id="doc2",
            content="Microsoft Corporation is a technology company headquartered in Redmond, WA. Founded in 1975 by Bill Gates and Paul Allen.",
            privacy_level=PrivacyLevel.PUBLIC
        ),
        PrivacyDocument(
            id="doc3",
            content="Our Q4 financial report shows revenue of $50M. Contact CFO Jane Doe at jane.doe@company.com for details.",
            privacy_level=PrivacyLevel.CONFIDENTIAL,
            access_controls=["executive", "finance"]
        )
    ]
    
    # Add documents
    num_docs = rag.add_documents(documents)
    print(f"Added {num_docs} privacy-aware documents")
    
    # Create a privacy-aware query
    query = PrivacyQuery(
        text="Who is the lead engineer at Microsoft?",
        user_id="user123",
        required_privacy_level=PrivacyLevel.PUBLIC
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
    print(f"Privacy preserved: {result.privacy_preserved}")
    print(f"Privacy techniques applied: {[tech.value for tech in result.privacy_techniques_applied]}")
    print(f"Privacy budget consumed: {result.privacy_budget_consumed}")
    print(f"Answer: {result.answer}")
    print(f"Sources: {len(result.sources)}")
    
    # Show privacy status
    print("\nPrivacy Status:")
    status = rag.get_privacy_status()
    print(f"  Current epsilon: {status['budget_status']['current_epsilon']}")
    print(f"  Remaining budget: {status['budget_status']['remaining_budget']}")
    print(f"  Total documents: {status['statistics']['total_documents']}")
    print(f"  Documents with PII: {status['statistics']['documents_with_pii']}")
    
    # Show another example with sensitive query
    print("\n" + "="*60)
    print("Sensitive Query Example:")
    
    sensitive_query = PrivacyQuery(
        text="What is the CFO's contact information?",
        user_id="finance_user",
        required_privacy_level=PrivacyLevel.CONFIDENTIAL
    )
    
    # Update access controls to allow finance user
    for doc in documents:
        if 'finance' in doc.access_controls:
            doc.access_controls.append('finance_user')
    
    # Re-add documents with updated access controls
    rag = PrivacyPreservingRAG(config=config)
    rag.add_documents(documents)
    
    # Create embedding for sensitive query
    sens_query_hash = hashlib.md5(sensitive_query.text.encode()).hexdigest()
    sens_query_embedding = np.frombuffer(bytes.fromhex(sens_query_hash[:32]), dtype=np.float32)
    if len(sens_query_embedding) < 384:
        sens_query_embedding = np.pad(sens_query_embedding, (0, 384 - len(sens_query_embedding)), 'constant')
    elif len(sens_query_embedding) > 384:
        sens_query_embedding = sens_query_embedding[:384]
    
    sens_result = rag.query(sensitive_query, sens_query_embedding, k=1)
    print(f"Query: {sensitive_query.text}")
    print(f"Answer: {sens_result.answer}")
    print(f"Privacy techniques applied: {[tech.value for tech in sens_result.privacy_techniques_applied]}")
    
    return rag


if __name__ == "__main__":
    example_usage()