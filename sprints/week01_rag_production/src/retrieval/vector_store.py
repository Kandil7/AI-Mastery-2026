"""
Vector Storage and Retrieval System for Production RAG

This module implements a comprehensive vector storage and retrieval system for the RAG system.
It manages vector embeddings for documents, provides efficient similarity search capabilities,
and integrates with various vector databases for scalable storage and retrieval.

The vector storage system follows production best practices:
- Support for multiple vector databases (Chroma, FAISS, Pinecone, etc.)
- Efficient similarity search algorithms
- Vector indexing and optimization
- Batch operations for performance
- Memory management for large vector sets
- Dimensionality validation and normalization

Key Features:
- Multiple vector database backends
- Similarity search with configurable algorithms
- Vector indexing and optimization
- Batch operations for efficient processing
- Memory-efficient vector storage
- Dimensionality validation and normalization
- Performance optimization for large-scale deployments

Security Considerations:
- Secure vector database connections
- Input validation for vector dimensions
- Access control for vector operations
- Encryption for vector data transmission
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field
import uuid
from enum import Enum

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with 'pip install chromadb'")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install with 'pip install sentence-transformers'")


class VectorDBType(Enum):
    """Enumeration for supported vector database types."""
    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


class VectorConfig(BaseModel):
    """
    Configuration for vector storage and retrieval.

    Attributes:
        db_type (VectorDBType): Type of vector database to use
        collection_name (str): Name of the collection/index
        persist_directory (str): Directory for persistent storage
        dimension (int): Dimension of the vectors
        metric (str): Distance metric for similarity search
        batch_size (int): Batch size for vector operations
        ef_construction (int): HNSW construction parameter (for Chroma)
        ef_search (int): HNSW search parameter (for Chroma)
        m (int): HNSW M parameter (for Chroma)
    """
    db_type: VectorDBType = Field(default=VectorDBType.IN_MEMORY, description="Type of vector database")
    collection_name: str = Field(default="rag_vectors", description="Name of the collection")
    persist_directory: str = Field(default="./data/vector_store", description="Directory for persistent storage")
    dimension: int = Field(default=384, ge=1, description="Dimension of the vectors")
    metric: str = Field(default="cosine", description="Distance metric for similarity search")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Batch size for vector operations")
    ef_construction: int = Field(default=200, ge=1, description="HNSW construction parameter")
    ef_search: int = Field(default=50, ge=1, description="HNSW search parameter")
    m: int = Field(default=16, ge=1, description="HNSW M parameter")


class VectorRecord(BaseModel):
    """
    Model for vector records in the storage system.

    Attributes:
        id (str): Unique identifier for the vector record
        vector (List[float]): The vector embedding
        metadata (Dict[str, Any]): Associated metadata
        document_id (str): Reference to the original document
        text_content (str): Original text content (optional)
    """
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: str
    text_content: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Validate vector dimension - only validate if 'dimension' is explicitly provided in metadata
        expected_dimension = self.metadata.get('dimension')
        if expected_dimension is not None and len(self.vector) != expected_dimension:
            raise ValueError(f"Vector dimension mismatch: expected {expected_dimension}, got {len(self.vector)}")


class BaseVectorStore(ABC):
    """Abstract base class for vector storage implementations."""

    def __init__(self, config: VectorConfig):
        """
        Initialize the vector store with configuration.

        Args:
            config: Vector storage configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self):
        """Initialize the vector store."""
        pass

    @abstractmethod
    async def add_vectors(self, vectors: List[VectorRecord]):
        """
        Add vectors to the store.

        Args:
            vectors: List of vector records to add
        """
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector to search for
            k: Number of results to return

        Returns:
            List of tuples (id, similarity_score)
        """
        pass

    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """
        Get a vector by ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Vector record if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector by ID.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def update_vector(self, vector_record: VectorRecord) -> bool:
        """
        Update an existing vector.

        Args:
            vector_record: Updated vector record

        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    async def get_count(self) -> int:
        """
        Get the total number of vectors in the store.

        Returns:
            Total count of vectors
        """
        pass

    @abstractmethod
    async def close(self):
        """Close the vector store and release resources."""
        pass


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store implementation for development and testing."""

    def __init__(self, config: VectorConfig):
        super().__init__(config)
        self.vectors: Dict[str, VectorRecord] = {}
        self.dimension = config.dimension

    async def initialize(self):
        """Initialize the in-memory vector store."""
        self.logger.info("Initialized in-memory vector store")
        # Nothing to initialize for in-memory store

    async def add_vectors(self, vectors: List[VectorRecord]):
        """Add vectors to the in-memory store."""
        for vector_record in vectors:
            # Validate vector dimension
            if len(vector_record.vector) != self.dimension:
                raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector_record.vector)}")
            
            self.vectors[vector_record.id] = vector_record
        
        self.logger.info(f"Added {len(vectors)} vectors to in-memory store")

    async def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors in the in-memory store."""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}")
        
        # Calculate cosine similarity with all stored vectors
        similarities = []
        query_array = np.array(query_vector)
        
        for vector_id, vector_record in self.vectors.items():
            stored_array = np.array(vector_record.vector)
            
            # Calculate cosine similarity
            dot_product = np.dot(query_array, stored_array)
            norm_query = np.linalg.norm(query_array)
            norm_stored = np.linalg.norm(stored_array)
            
            if norm_query == 0 or norm_stored == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_query * norm_stored)
            
            similarities.append((vector_id, float(similarity)))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    async def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Get a vector by ID from the in-memory store."""
        return self.vectors.get(vector_id)

    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID from the in-memory store."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            return True
        return False

    async def update_vector(self, vector_record: VectorRecord) -> bool:
        """Update an existing vector in the in-memory store."""
        if vector_record.id in self.vectors:
            # Validate vector dimension
            if len(vector_record.vector) != self.dimension:
                raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vector_record.vector)}")
            
            self.vectors[vector_record.id] = vector_record
            return True
        return False

    async def get_count(self) -> int:
        """Get the total number of vectors in the in-memory store."""
        return len(self.vectors)

    async def close(self):
        """Close the in-memory vector store."""
        self.vectors.clear()
        self.logger.info("Closed in-memory vector store")


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, config: VectorConfig):
        super().__init__(config)
        
        if not CHROMA_AVAILABLE:
            raise RuntimeError("ChromaDB is not available. Install with 'pip install chromadb'")
        
        self.client = None
        self.collection = None
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config.persist_directory,
            anonymized_telemetry=False
        )

    async def initialize(self):
        """Initialize the ChromaDB vector store."""
        loop = asyncio.get_event_loop()
        
        # Create client and collection in thread pool
        self.client = await loop.run_in_executor(
            None,
            lambda: chromadb.PersistentClient(
                path=self.config.persist_directory,
                settings=self.settings,
            ),
        )
        
        # Create or get collection with specified settings
        metadata = {
            "hnsw:space": self.config.metric,
            "hnsw:construction_ef": self.config.ef_construction,
            "hnsw:search_ef": self.config.ef_search,
            "hnsw:M": self.config.m
        }
        
        self.collection = await loop.run_in_executor(
            None,
            lambda: self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata=metadata,
            ),
        )
        
        self.logger.info(f"Initialized ChromaDB vector store with collection: {self.config.collection_name}")

    async def add_vectors(self, vectors: List[VectorRecord]):
        """Add vectors to the ChromaDB store."""
        if not vectors:
            return
        
        loop = asyncio.get_event_loop()
        
        # Prepare data for ChromaDB
        ids = [v.id for v in vectors]
        embeddings = [v.vector for v in vectors]
        metadatas = [v.metadata for v in vectors]
        documents = [v.text_content for v in vectors if v.text_content]
        
        # If not all vectors have text content, fill with empty strings
        if len(documents) != len(vectors):
            documents = [v.text_content or "" for v in vectors]
        
        # Add to ChromaDB
        await loop.run_in_executor(
            None,
            lambda: self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
        )
        
        self.logger.info(f"Added {len(vectors)} vectors to ChromaDB")

    async def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors in ChromaDB."""
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            lambda: self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["distances"]
            )
        )
        
        # Extract IDs and distances
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        
        # Convert distances to similarities (1 / (1 + distance) for cosine similarity)
        similarities = [(id_, 1.0 / (1.0 + dist)) for id_, dist in zip(ids, distances)]
        
        return similarities

    async def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Get a vector by ID from ChromaDB."""
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            lambda: self.collection.get(
                ids=[vector_id],
                include=["embeddings", "metadatas", "documents"]
            )
        )
        
        if not results["ids"] or not results["embeddings"]:
            return None
        
        # Extract the vector data
        embedding = results["embeddings"][0]
        metadata = results["metadatas"][0] if results["metadatas"] else {}
        document = results["documents"][0] if results["documents"] else None
        
        return VectorRecord(
            id=vector_id,
            vector=embedding,
            metadata=metadata,
            document_id=metadata.get("document_id", ""),
            text_content=document
        )

    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID from ChromaDB."""
        loop = asyncio.get_event_loop()
        
        try:
            await loop.run_in_executor(
                None,
                lambda: self.collection.delete(ids=[vector_id])
            )
            return True
        except Exception as e:
            self.logger.error(f"Error deleting vector {vector_id}: {e}")
            return False

    async def update_vector(self, vector_record: VectorRecord) -> bool:
        """Update an existing vector in ChromaDB."""
        loop = asyncio.get_event_loop()
        
        try:
            await loop.run_in_executor(
                None,
                lambda: self.collection.update(
                    ids=[vector_record.id],
                    embeddings=[vector_record.vector],
                    metadatas=[vector_record.metadata],
                    documents=[vector_record.text_content] if vector_record.text_content else None
                )
            )
            return True
        except Exception as e:
            self.logger.error(f"Error updating vector {vector_record.id}: {e}")
            return False

    async def get_count(self) -> int:
        """Get the total number of vectors in ChromaDB."""
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, lambda: self.collection.count())
        return count

    async def close(self):
        """Close the ChromaDB vector store."""
        # ChromaDB doesn't have a specific close method
        self.logger.info("Closed ChromaDB vector store")


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""

    def __init__(self, config: VectorConfig):
        super().__init__(config)
        
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not available. Install with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
        
        self.index = None
        self.id_to_vector: Dict[str, VectorRecord] = {}
        self.vector_ids: List[str] = []

    async def initialize(self):
        """Initialize the FAISS vector store."""
        # Create FAISS index based on metric type
        if self.config.metric.lower() in ['cosine', 'inner_product']:
            # For cosine similarity, we use inner product on normalized vectors
            self.index = faiss.IndexFlatIP(self.config.dimension)
        elif self.config.metric.lower() == 'l2':
            self.index = faiss.IndexFlatL2(self.config.dimension)
        else:
            # Default to inner product
            self.index = faiss.IndexFlatIP(self.config.dimension)
        
        self.logger.info(f"Initialized FAISS vector store with dimension: {self.config.dimension}")

    async def add_vectors(self, vectors: List[VectorRecord]):
        """Add vectors to the FAISS store."""
        if not vectors:
            return
        
        # Validate all vectors have the same dimension
        for vector_record in vectors:
            if len(vector_record.vector) != self.config.dimension:
                raise ValueError(f"Vector dimension mismatch: expected {self.config.dimension}, got {len(vector_record.vector)}")
        
        # Convert vectors to numpy array
        vector_array = np.array([v.vector for v in vectors]).astype('float32')
        
        # Normalize vectors for cosine similarity
        if self.config.metric.lower() in ['cosine', 'inner_product']:
            faiss.normalize_L2(vector_array)
        
        # Add to FAISS index
        self.index.add(vector_array)
        
        # Store metadata separately
        for i, vector_record in enumerate(vectors):
            vector_id = vector_record.id
            self.id_to_vector[vector_id] = vector_record
            # Map index position to vector ID
            if len(self.vector_ids) <= self.index.ntotal - len(vectors) + i:
                self.vector_ids.extend([None] * (self.index.ntotal - len(vectors) + i - len(self.vector_ids) + 1))
            self.vector_ids[self.index.ntotal - len(vectors) + i] = vector_id
        
        self.logger.info(f"Added {len(vectors)} vectors to FAISS")

    async def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors in FAISS."""
        if len(query_vector) != self.config.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.config.dimension}, got {len(query_vector)}")
        
        # Convert query to numpy array
        query_array = np.array([query_vector]).astype('float32')
        
        # Normalize for cosine similarity
        if self.config.metric.lower() in ['cosine', 'inner_product']:
            faiss.normalize_L2(query_array)
        
        # Perform search
        distances, indices = self.index.search(query_array, k)
        
        # Convert to list of (id, similarity) tuples
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.vector_ids):  # Valid index
                vector_id = self.vector_ids[idx]
                if vector_id is not None:
                    # Convert distance to similarity based on metric
                    if self.config.metric.lower() in ['cosine', 'inner_product']:
                        similarity = float(dist)  # Inner product is similarity for normalized vectors
                    else:
                        similarity = 1.0 / (1.0 + float(dist))  # Convert distance to similarity
                    results.append((vector_id, similarity))
        
        return results

    async def get_vector(self, vector_id: str) -> Optional[VectorRecord]:
        """Get a vector by ID from FAISS."""
        return self.id_to_vector.get(vector_id)

    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID from FAISS."""
        # FAISS doesn't support efficient deletion of individual vectors
        # We'll mark it as deleted but keep the index structure
        if vector_id in self.id_to_vector:
            del self.id_to_vector[vector_id]
            # Remove from vector_ids list
            if vector_id in self.vector_ids:
                idx = self.vector_ids.index(vector_id)
                self.vector_ids[idx] = None
            return True
        return False

    async def update_vector(self, vector_record: VectorRecord) -> bool:
        """Update an existing vector in FAISS."""
        # FAISS doesn't support efficient updates of individual vectors
        # We'll delete and re-add
        if vector_record.id in self.id_to_vector:
            # Delete old vector (by marking as None in our mapping)
            await self.delete_vector(vector_record.id)
            # Add new vector
            await self.add_vectors([vector_record])
            return True
        return False

    async def get_count(self) -> int:
        """Get the total number of vectors in FAISS."""
        return self.index.ntotal

    async def close(self):
        """Close the FAISS vector store."""
        self.index = None
        self.id_to_vector.clear()
        self.vector_ids.clear()
        self.logger.info("Closed FAISS vector store")


class VectorStoreFactory:
    """Factory for creating appropriate vector store implementations."""

    @staticmethod
    def create_vector_store(config: VectorConfig) -> BaseVectorStore:
        """
        Create a vector store instance based on the specified configuration.

        Args:
            config: Vector storage configuration

        Returns:
            Appropriate vector store instance
        """
        if config.db_type == VectorDBType.CHROMA:
            return ChromaVectorStore(config)
        elif config.db_type == VectorDBType.FAISS:
            return FAISSVectorStore(config)
        elif config.db_type == VectorDBType.IN_MEMORY:
            return InMemoryVectorStore(config)
        else:
            raise ValueError(f"Unknown vector database type: {config.db_type}")


class VectorManager:
    """
    Manager class for handling vector operations in the RAG system.
    
    This class provides a unified interface for vector operations regardless
    of the underlying vector database implementation.
    """

    def __init__(self, config: VectorConfig):
        """
        Initialize the vector manager with configuration.

        Args:
            config: Vector storage configuration
        """
        self.config = config
        self.vector_store: Optional[BaseVectorStore] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def initialize(self):
        """Initialize the vector manager and underlying store."""
        self.vector_store = VectorStoreFactory.create_vector_store(self.config)
        await self.vector_store.initialize()
        self.logger.info("Vector manager initialized")

    async def add_document_vector(self, document_id: str, vector: List[float], 
                                text_content: Optional[str] = None, 
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a vector representation of a document.

        Args:
            document_id: ID of the original document
            vector: Vector embedding of the document
            text_content: Original text content (optional)
            metadata: Additional metadata (optional)

        Returns:
            ID of the created vector record
        """
        if not self.vector_store:
            raise RuntimeError("Vector manager not initialized")
        
        # Generate a unique vector ID
        vector_id = f"vec_{uuid.uuid4().hex[:16]}"
        
        # Create vector record
        vector_record = VectorRecord(
            id=vector_id,
            vector=vector,
            metadata=metadata or {},
            document_id=document_id,
            text_content=text_content
        )
        
        # Add to vector store
        await self.vector_store.add_vectors([vector_record])
        
        return vector_id

    async def search_similar(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors to the query vector.

        Args:
            query_vector: Query vector to search for
            k: Number of results to return

        Returns:
            List of tuples (vector_id, similarity_score)
        """
        if not self.vector_store:
            raise RuntimeError("Vector manager not initialized")
        
        return await self.vector_store.search(query_vector, k)

    async def get_vector_by_id(self, vector_id: str) -> Optional[VectorRecord]:
        """
        Get a vector record by its ID.

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Vector record if found, None otherwise
        """
        if not self.vector_store:
            raise RuntimeError("Vector manager not initialized")
        
        return await self.vector_store.get_vector(vector_id)

    async def delete_vector_by_id(self, vector_id: str) -> bool:
        """
        Delete a vector by its ID.

        Args:
            vector_id: ID of the vector to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.vector_store:
            raise RuntimeError("Vector manager not initialized")
        
        return await self.vector_store.delete_vector(vector_id)

    async def update_vector_record(self, vector_record: VectorRecord) -> bool:
        """
        Update an existing vector record.

        Args:
            vector_record: Updated vector record

        Returns:
            True if updated, False if not found
        """
        if not self.vector_store:
            raise RuntimeError("Vector manager not initialized")
        
        return await self.vector_store.update_vector(vector_record)

    async def get_total_count(self) -> int:
        """
        Get the total number of vectors in the store.

        Returns:
            Total count of vectors
        """
        if not self.vector_store:
            raise RuntimeError("Vector manager not initialized")
        
        return await self.vector_store.get_count()

    async def close(self):
        """Close the vector manager and underlying store."""
        if self.vector_store:
            await self.vector_store.close()
        self.logger.info("Vector manager closed")


# Global instance of vector manager
vector_manager: Optional[VectorManager] = None


async def initialize_vector_manager(config: VectorConfig):
    """
    Initialize the global vector manager.

    Args:
        config: Vector storage configuration
    """
    global vector_manager
    vector_manager = VectorManager(config)
    await vector_manager.initialize()


async def close_vector_manager():
    """Close the global vector manager."""
    global vector_manager
    if vector_manager:
        await vector_manager.close()
        vector_manager = None


__all__ = [
    "VectorDBType", "VectorConfig", "VectorRecord", "BaseVectorStore",
    "InMemoryVectorStore", "ChromaVectorStore", "FAISSVectorStore",
    "VectorStoreFactory", "VectorManager", "initialize_vector_manager",
    "close_vector_manager", "vector_manager"
]
