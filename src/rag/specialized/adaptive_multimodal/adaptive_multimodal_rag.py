"""
Adaptive Multi-Modal RAG (Retrieval-Augmented Generation) Module

This module implements an adaptive multi-modal RAG system that can handle
different types of media (text, images, audio, video) and dynamically adjust
its retrieval and generation strategies based on the input modality and context.

Key Features:
- Multi-modal input processing (text, images, audio, video)
- Adaptive retrieval based on input type
- Modality-specific embedding generation
- Cross-modal similarity matching
- Dynamic response generation based on modalities

Architecture:
- Modality Router: Determines input type and routes to appropriate processor
- Multi-Modal Encoder: Generates embeddings for different modalities
- Adaptive Retriever: Adjusts retrieval strategy based on modality
- Cross-Modal Fusion: Combines information from different modalities
- Modality-Aware Generator: Generates responses considering input modalities
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import hashlib
import base64
from PIL import Image
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND DATA CLASSES
# ============================================================

class ModalityType(Enum):
    """Types of modalities supported by the system."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MIXED = "mixed"


@dataclass
class MultiModalDocument:
    """A document that can contain multiple modalities."""
    id: str
    text_content: str = ""
    image_content: Optional[bytes] = None  # Image bytes
    audio_content: Optional[bytes] = None  # Audio bytes
    video_content: Optional[bytes] = None  # Video bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_embedding: Optional[np.ndarray] = None
    image_embedding: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None
    video_embedding: Optional[np.ndarray] = None
    modality_type: ModalityType = ModalityType.TEXT

    def __post_init__(self):
        if not self.id:
            # Generate ID from content hash
            content_hash = hashlib.md5(
                (self.text_content + 
                 str(self.image_content) + 
                 str(self.audio_content) + 
                 str(self.video_content)).encode()
            ).hexdigest()[:16]
            self.id = content_hash


@dataclass
class MultiModalQuery:
    """A query that can contain multiple modalities."""
    text_query: str = ""
    image_query: Optional[bytes] = None
    audio_query: Optional[bytes] = None
    video_query: Optional[bytes] = None
    preferred_modality: ModalityType = ModalityType.TEXT
    similarity_threshold: float = 0.3

    @property
    def modality_type(self) -> ModalityType:
        """Determine the primary modality of the query."""
        modalities_present = []
        if self.text_query:
            modalities_present.append(ModalityType.TEXT)
        if self.image_query:
            modalities_present.append(ModalityType.IMAGE)
        if self.audio_query:
            modalities_present.append(ModalityType.AUDIO)
        if self.video_query:
            modalities_present.append(ModalityType.VIDEO)
        
        if len(modalities_present) == 0:
            return ModalityType.TEXT
        elif len(modalities_present) == 1:
            return modalities_present[0]
        else:
            return ModalityType.MIXED


@dataclass
class MultiModalRetrievalResult:
    """Result from multi-modal retrieval."""
    document: MultiModalDocument
    score: float
    modality_used: ModalityType
    cross_modal_score: Optional[float] = None  # Score from cross-modal matching


@dataclass
class MultiModalGenerationResult:
    """Result from multi-modal generation."""
    answer: str
    sources: List[MultiModalDocument]
    modalities_used: List[ModalityType]
    confidence: float
    latency_ms: float
    token_count: int


# ============================================================
# MODALITY PROCESSORS
# ============================================================

class BaseModalityProcessor(ABC):
    """Base class for processing different modalities."""
    
    def __init__(self, modality_type: ModalityType):
        self.modality_type = modality_type
    
    @abstractmethod
    def encode(self, content: Any) -> np.ndarray:
        """Encode content into embedding space."""
        pass
    
    @abstractmethod
    def preprocess(self, content: Any) -> Any:
        """Preprocess content before encoding."""
        pass


class TextProcessor(BaseModalityProcessor):
    """Processor for text modality."""
    
    def __init__(self):
        super().__init__(ModalityType.TEXT)
        # In a real implementation, this would load a text encoder
        # For now, we'll simulate with a simple approach
        self.vocabulary = {}
        self.vocab_size = 1000
        self.embedding_dim = 384
        
    def preprocess(self, content: str) -> str:
        """Preprocess text content."""
        if not isinstance(content, str):
            content = str(content)
        # Simple preprocessing
        content = content.lower().strip()
        return content
    
    def encode(self, content: str) -> np.ndarray:
        """Encode text content into embedding space."""
        processed_content = self.preprocess(content)
        
        # Simulate text encoding using a simple approach
        # In a real implementation, this would use a pre-trained model like BERT
        text_hash = hashlib.md5(processed_content.encode()).hexdigest()
        embedding = np.frombuffer(bytes.fromhex(text_hash[:32]), dtype=np.float32)
        
        # Pad or truncate to desired dimension
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')
        elif len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
            
        return embedding


class ImageProcessor(BaseModalityProcessor):
    """Processor for image modality."""
    
    def __init__(self):
        super().__init__(ModalityType.IMAGE)
        self.embedding_dim = 512  # Typical for image embeddings
        
    def preprocess(self, content: bytes) -> Image.Image:
        """Preprocess image content."""
        if isinstance(content, bytes):
            image = Image.open(io.BytesIO(content))
            # Resize to standard size for processing
            image = image.resize((224, 224))
            return image
        else:
            raise ValueError("Image content must be bytes")
    
    def encode(self, content: bytes) -> np.ndarray:
        """Encode image content into embedding space."""
        try:
            image = self.preprocess(content)
            
            # Simulate image encoding
            # In a real implementation, this would use a CNN like ResNet
            image_array = np.array(image)
            # Simple hash-based embedding simulation
            image_hash = hashlib.md5(image_array.tobytes()).hexdigest()
            embedding = np.frombuffer(bytes.fromhex(image_hash[:64]), dtype=np.float32)
            
            # Pad or truncate to desired dimension
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
                
            return embedding
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            # Return a zero embedding as fallback
            return np.zeros(self.embedding_dim)


class AudioProcessor(BaseModalityProcessor):
    """Processor for audio modality."""
    
    def __init__(self):
        super().__init__(ModalityType.AUDIO)
        self.embedding_dim = 256  # Typical for audio embeddings
        
    def preprocess(self, content: bytes) -> bytes:
        """Preprocess audio content."""
        # In a real implementation, this would extract audio features
        # For now, just return the content
        return content
    
    def encode(self, content: bytes) -> np.ndarray:
        """Encode audio content into embedding space."""
        try:
            processed_content = self.preprocess(content)
            
            # Simulate audio encoding
            # In a real implementation, this would use audio-specific models
            audio_hash = hashlib.md5(processed_content).hexdigest()
            embedding = np.frombuffer(bytes.fromhex(audio_hash[:32]), dtype=np.float32)
            
            # Pad or truncate to desired dimension
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
                
            return embedding
        except Exception as e:
            logger.error(f"Error encoding audio: {e}")
            # Return a zero embedding as fallback
            return np.zeros(self.embedding_dim)


class VideoProcessor(BaseModalityProcessor):
    """Processor for video modality."""
    
    def __init__(self):
        super().__init__(ModalityType.VIDEO)
        self.embedding_dim = 512  # Typical for video embeddings
        
    def preprocess(self, content: bytes) -> bytes:
        """Preprocess video content."""
        # In a real implementation, this would extract video frames/features
        # For now, just return the content
        return content
    
    def encode(self, content: bytes) -> np.ndarray:
        """Encode video content into embedding space."""
        try:
            processed_content = self.preprocess(content)
            
            # Simulate video encoding
            # In a real implementation, this would use video-specific models
            video_hash = hashlib.md5(processed_content).hexdigest()
            embedding = np.frombuffer(bytes.fromhex(video_hash[:64]), dtype=np.float32)
            
            # Pad or truncate to desired dimension
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)), 'constant')
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
                
            return embedding
        except Exception as e:
            logger.error(f"Error encoding video: {e}")
            # Return a zero embedding as fallback
            return np.zeros(self.embedding_dim)


# ============================================================
# MULTI-MODAL ENCODER
# ============================================================

class MultiModalEncoder:
    """Encodes content across multiple modalities."""
    
    def __init__(self):
        self.processors = {
            ModalityType.TEXT: TextProcessor(),
            ModalityType.IMAGE: ImageProcessor(),
            ModalityType.AUDIO: AudioProcessor(),
            ModalityType.VIDEO: VideoProcessor()
        }
    
    def encode_document(self, document: MultiModalDocument) -> MultiModalDocument:
        """Encode all modalities in a document."""
        if document.text_content and document.text_embedding is None:
            document.text_embedding = self.processors[ModalityType.TEXT].encode(document.text_content)
        
        if document.image_content and document.image_embedding is None:
            document.image_embedding = self.processors[ModalityType.IMAGE].encode(document.image_content)
        
        if document.audio_content and document.audio_embedding is None:
            document.audio_embedding = self.processors[ModalityType.AUDIO].encode(document.audio_content)
        
        if document.video_content and document.video_embedding is None:
            document.video_embedding = self.processors[ModalityType.VIDEO].encode(document.video_content)
        
        return document
    
    def encode_query(self, query: MultiModalQuery) -> Dict[ModalityType, np.ndarray]:
        """Encode all modalities in a query."""
        embeddings = {}
        
        if query.text_query:
            embeddings[ModalityType.TEXT] = self.processors[ModalityType.TEXT].encode(query.text_query)
        
        if query.image_query:
            embeddings[ModalityType.IMAGE] = self.processors[ModalityType.IMAGE].encode(query.image_query)
        
        if query.audio_query:
            embeddings[ModalityType.AUDIO] = self.processors[ModalityType.AUDIO].encode(query.audio_query)
        
        if query.video_query:
            embeddings[ModalityType.VIDEO] = self.processors[ModalityType.VIDEO].encode(query.video_query)
        
        return embeddings


# ============================================================
# ADAPTIVE RETRIEVER
# ============================================================

class AdaptiveMultiModalRetriever:
    """
    Adaptive retriever that adjusts strategy based on query modality.
    
    Supports:
    - Same-modality retrieval (text query → text content)
    - Cross-modality retrieval (image query → text content)
    - Mixed-modality retrieval (combined query → mixed content)
    """
    
    def __init__(self, encoder: MultiModalEncoder):
        self.encoder = encoder
        self.documents: List[MultiModalDocument] = []
        self.text_embeddings: Optional[np.ndarray] = None
        self.image_embeddings: Optional[np.ndarray] = None
        self.audio_embeddings: Optional[np.ndarray] = None
        self.video_embeddings: Optional[np.ndarray] = None
        
        # Track which embedding matrices correspond to which documents
        self.doc_indices = {
            ModalityType.TEXT: [],
            ModalityType.IMAGE: [],
            ModalityType.AUDIO: [],
            ModalityType.VIDEO: []
        }
    
    def add_documents(self, documents: List[MultiModalDocument]) -> None:
        """Add documents to the retriever."""
        for doc in documents:
            # Encode the document if not already encoded
            doc = self.encoder.encode_document(doc)
            self.documents.append(doc)
            
            # Track document indices for each modality
            if doc.text_embedding is not None:
                self.doc_indices[ModalityType.TEXT].append(len(self.documents) - 1)
            if doc.image_embedding is not None:
                self.doc_indices[ModalityType.IMAGE].append(len(self.documents) - 1)
            if doc.audio_embedding is not None:
                self.doc_indices[ModalityType.AUDIO].append(len(self.documents) - 1)
            if doc.video_embedding is not None:
                self.doc_indices[ModalityType.VIDEO].append(len(self.documents) - 1)
        
        # Rebuild embedding matrices
        self._build_embedding_matrices()
    
    def _build_embedding_matrices(self) -> None:
        """Build matrices of embeddings for efficient similarity computation."""
        # Build text embedding matrix
        text_docs = [self.documents[i] for i in self.doc_indices[ModalityType.TEXT]]
        if text_docs:
            self.text_embeddings = np.array([doc.text_embedding for doc in text_docs])
        else:
            self.text_embeddings = None
            
        # Build image embedding matrix
        image_docs = [self.documents[i] for i in self.doc_indices[ModalityType.IMAGE]]
        if image_docs:
            self.image_embeddings = np.array([doc.image_embedding for doc in image_docs])
        else:
            self.image_embeddings = None
            
        # Build audio embedding matrix
        audio_docs = [self.documents[i] for i in self.doc_indices[ModalityType.AUDIO]]
        if audio_docs:
            self.audio_embeddings = np.array([doc.audio_embedding for doc in audio_docs])
        else:
            self.audio_embeddings = None
            
        # Build video embedding matrix
        video_docs = [self.documents[i] for i in self.doc_indices[ModalityType.VIDEO]]
        if video_docs:
            self.video_embeddings = np.array([doc.video_embedding for doc in video_docs])
        else:
            self.video_embeddings = None
    
    def _compute_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document embeddings."""
        # Normalize embeddings
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def retrieve(self, query: MultiModalQuery, k: int = 5) -> List[MultiModalRetrievalResult]:
        """Retrieve relevant documents based on multi-modal query."""
        # Encode the query
        query_embeddings = self.encoder.encode_query(query)
        
        all_results = []
        
        # Process each modality in the query
        for query_modality, query_embedding in query_embeddings.items():
            # Determine which document embeddings to compare against
            if query_modality == ModalityType.TEXT:
                doc_embeddings = self.text_embeddings
                doc_indices = self.doc_indices[ModalityType.TEXT]
            elif query_modality == ModalityType.IMAGE:
                doc_embeddings = self.image_embeddings
                doc_indices = self.doc_indices[ModalityType.IMAGE]
            elif query_modality == ModalityType.AUDIO:
                doc_embeddings = self.audio_embeddings
                doc_indices = self.doc_indices[ModalityType.AUDIO]
            elif query_modality == ModalityType.VIDEO:
                doc_embeddings = self.video_embeddings
                doc_indices = self.doc_indices[ModalityType.VIDEO]
            else:
                continue  # Skip unknown modalities
            
            if doc_embeddings is None or len(doc_embeddings) == 0:
                continue
            
            # Compute similarities
            similarities = self._compute_similarity(query_embedding, doc_embeddings)
            
            # Create results for this modality
            for i, (doc_idx, similarity) in enumerate(zip(doc_indices, similarities)):
                if similarity >= query.similarity_threshold:
                    all_results.append(MultiModalRetrievalResult(
                        document=self.documents[doc_idx],
                        score=float(similarity),
                        modality_used=query_modality,
                        cross_modal_score=None  # Will be computed later if needed
                    ))
        
        # For cross-modal retrieval, we also want to find documents that match
        # across different modalities (e.g., image query matching text content)
        all_results.extend(self._perform_cross_modal_retrieval(query, k))
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]
    
    def _perform_cross_modal_retrieval(self, query: MultiModalQuery, k: int) -> List[MultiModalRetrievalResult]:
        """Perform cross-modal retrieval (e.g., text query for image content)."""
        query_embeddings = self.encoder.encode_query(query)
        cross_modal_results = []
        
        # For each query modality, check against all document modalities
        for query_modality, query_embedding in query_embeddings.items():
            # Compare against all document modalities
            for doc_modality in [ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO, ModalityType.VIDEO]:
                if query_modality == doc_modality:
                    continue  # Skip same modality (already handled above)
                
                # Get document embeddings for this modality
                if doc_modality == ModalityType.TEXT:
                    doc_embeddings = self.text_embeddings
                    doc_indices = self.doc_indices[ModalityType.TEXT]
                elif doc_modality == ModalityType.IMAGE:
                    doc_embeddings = self.image_embeddings
                    doc_indices = self.doc_indices[ModalityType.IMAGE]
                elif doc_modality == ModalityType.AUDIO:
                    doc_embeddings = self.audio_embeddings
                    doc_indices = self.doc_indices[ModalityType.AUDIO]
                elif doc_modality == ModalityType.VIDEO:
                    doc_embeddings = self.video_embeddings
                    doc_indices = self.doc_indices[ModalityType.VIDEO]
                else:
                    continue
                
                if doc_embeddings is None or len(doc_embeddings) == 0:
                    continue
                
                # Compute cross-modal similarities
                similarities = self._compute_similarity(query_embedding, doc_embeddings)
                
                # Create cross-modal results
                for i, (doc_idx, similarity) in enumerate(zip(doc_indices, similarities)):
                    if similarity >= query.similarity_threshold:
                        cross_modal_results.append(MultiModalRetrievalResult(
                            document=self.documents[doc_idx],
                            score=float(similarity),
                            modality_used=doc_modality,
                            cross_modal_score=float(similarity)
                        ))
        
        return cross_modal_results


# ============================================================
# CROSS-MODAL FUSION
# ============================================================

class CrossModalFusion:
    """
    Fuses information from multiple modalities to create a unified context.
    
    This component combines information from different modalities to create
    a comprehensive context for generation.
    """
    
    def __init__(self):
        self.fusion_weights = {
            ModalityType.TEXT: 0.4,
            ModalityType.IMAGE: 0.2,
            ModalityType.AUDIO: 0.2,
            ModalityType.VIDEO: 0.2
        }
    
    def fuse_contexts(self, retrieval_results: List[MultiModalRetrievalResult]) -> str:
        """Fuse contexts from multiple modalities into a single context string."""
        context_parts = []
        
        # Group results by document
        doc_groups = {}
        for result in retrieval_results:
            doc_id = result.document.id
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # Process each document group
        for doc_id, results in doc_groups.items():
            doc_context_parts = [f"Document {doc_id}:"]
            
            # Add text content if available
            text_content_added = False
            for result in results:
                if (result.document.text_content and 
                    not text_content_added and 
                    result.modality_used in [ModalityType.TEXT, ModalityType.MIXED]):
                    doc_context_parts.append(f"Text: {result.document.text_content[:500]}...")
                    text_content_added = True
            
            # Add other modalities as references
            other_modalities = []
            for result in results:
                if result.modality_used == ModalityType.IMAGE:
                    other_modalities.append("image")
                elif result.modality_used == ModalityType.AUDIO:
                    other_modalities.append("audio")
                elif result.modality_used == ModalityType.VIDEO:
                    other_modalities.append("video")
            
            if other_modalities:
                doc_context_parts.append(f"Related to: {', '.join(set(other_modalities))}")
            
            context_parts.append(" ".join(doc_context_parts))
        
        return "\n\n".join(context_parts)


# ============================================================
# ADAPTIVE MULTI-MODAL RAG SYSTEM
# ============================================================

class AdaptiveMultiModalRAG:
    """
    Adaptive Multi-Modal RAG system that handles different input types.
    
    This system can accept queries in various modalities and adapt its
    retrieval and generation strategies accordingly.
    """
    
    def __init__(self):
        self.encoder = MultiModalEncoder()
        self.retriever = AdaptiveMultiModalRetriever(self.encoder)
        self.fusion = CrossModalFusion()
        
        # Generation function (placeholder - replace with actual LLM)
        self.generate_fn: Optional[Callable] = None
        
        logger.info("Initialized Adaptive Multi-Modal RAG system")
    
    def set_generator(self, generate_fn: Callable[[str], str]) -> None:
        """Set the LLM generation function."""
        self.generate_fn = generate_fn
    
    def add_documents(self, documents: List[MultiModalDocument]) -> int:
        """Add documents to the RAG system."""
        # Encode documents
        encoded_docs = []
        for doc in documents:
            encoded_doc = self.encoder.encode_document(doc)
            encoded_docs.append(encoded_doc)
        
        # Add to retriever
        self.retriever.add_documents(encoded_docs)
        
        logger.info(f"Added {len(documents)} multi-modal documents")
        return len(documents)
    
    def query(self, 
              query: MultiModalQuery, 
              k: int = 5) -> MultiModalGenerationResult:
        """
        Query the multi-modal RAG system.
        
        Args:
            query: Multi-modal query object
            k: Number of results to retrieve
            
        Returns:
            MultiModalGenerationResult with answer and sources
        """
        import time
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(query, k=k)
        
        if not retrieval_results:
            # No results found, return a default response
            latency_ms = (time.time() - start_time) * 1000
            return MultiModalGenerationResult(
                answer="No relevant information found for your query.",
                sources=[],
                modalities_used=[],
                confidence=0.0,
                latency_ms=latency_ms,
                token_count=10
            )
        
        # Fuse the contexts from different modalities
        context = self.fusion.fuse_contexts(retrieval_results)
        
        # Build the generation prompt
        prompt = self._build_prompt(query, context)
        
        # Generate answer
        if self.generate_fn:
            answer = self.generate_fn(prompt)
        else:
            # Placeholder answer
            answer = self._generate_placeholder_answer(query, retrieval_results)
        
        # Extract modalities used
        modalities_used = list(set(result.modality_used for result in retrieval_results))
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = MultiModalGenerationResult(
            answer=answer,
            sources=[result.document for result in retrieval_results],
            modalities_used=modalities_used,
            confidence=self._estimate_confidence(retrieval_results),
            latency_ms=latency_ms,
            token_count=len(prompt.split()) + len(answer.split())
        )
        
        return result
    
    def _build_prompt(self, query: MultiModalQuery, context: str) -> str:
        """Build the generation prompt."""
        query_description = []
        if query.text_query:
            query_description.append(f"text: '{query.text_query}'")
        if query.image_query:
            query_description.append("image")
        if query.audio_query:
            query_description.append("audio")
        if query.video_query:
            query_description.append("video")
        
        query_desc_str = ", ".join(query_description)
        
        return f"""Answer the question based on the provided multi-modal context.
The query contains the following modalities: {query_desc_str}.
If the answer cannot be found in the context, say "I don't have enough information to answer this question."
Always cite your sources.

Context:
{context}

Question: {query.text_query or 'Please respond based on the multi-modal input'}

Answer:"""
    
    def _generate_placeholder_answer(self, 
                                   query: MultiModalQuery, 
                                   results: List[MultiModalRetrievalResult]) -> str:
        """Generate a placeholder answer for testing."""
        if query.text_query:
            return f"Based on the multi-modal context, the system found {len(results)} relevant documents. The primary query was: '{query.text_query}'."
        else:
            return f"The system analyzed multi-modal inputs and found {len(results)} relevant documents based on the non-text modalities provided."
    
    def _estimate_confidence(self, results: List[MultiModalRetrievalResult]) -> float:
        """Estimate confidence based on retrieval scores."""
        if not results:
            return 0.0
        
        # Average of top retrieval scores
        scores = [r.score for r in results[:3]]
        avg_score = np.mean(scores)
        
        # Normalize to 0-1 range
        return float(min(1.0, avg_score))


# ============================================================
# EXAMPLE USAGE AND TESTING
# ============================================================

def example_usage():
    """Demonstrate Adaptive Multi-Modal RAG usage."""
    
    # Create RAG system
    rag = AdaptiveMultiModalRAG()
    
    # Sample multi-modal documents
    documents = [
        MultiModalDocument(
            id="doc1",
            text_content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            metadata={"source": "AI textbook", "topic": "ML basics"},
            modality_type=ModalityType.TEXT
        ),
        MultiModalDocument(
            id="doc2",
            text_content="Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            image_content=b"dummy_image_bytes_here",  # In practice, this would be actual image bytes
            metadata={"source": "DL course", "topic": "neural networks"},
            modality_type=ModalityType.MIXED
        ),
        MultiModalDocument(
            id="doc3",
            text_content="Natural language processing enables computers to understand and generate human language.",
            metadata={"source": "NLP guide", "topic": "language models"},
            modality_type=ModalityType.TEXT
        )
    ]
    
    # Add documents
    num_docs = rag.add_documents(documents)
    print(f"Added {num_docs} multi-modal documents")
    
    # Create a text query
    text_query = MultiModalQuery(
        text_query="What is machine learning?",
        preferred_modality=ModalityType.TEXT
    )
    
    # Query the system
    result = rag.query(text_query, k=3)
    
    print(f"\nQuery: {text_query.text_query}")
    print(f"Modalities used: {[m.value for m in result.modalities_used]}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Latency: {result.latency_ms:.1f}ms")
    print(f"Answer: {result.answer}")
    print(f"Sources: {len(result.sources)}")
    
    return rag


if __name__ == "__main__":
    example_usage()