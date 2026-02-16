"""
Legal Vector Indexing System
============================
Specialized vector index for legal document retrieval.
Handles embeddings, metadata filtering, and semantic search.

Uses FAISS-like logic or wrapper around a standard vector DB.
For this case study, we implement a persistent in-memory index using Numpy/Pickle
to demonstrate the mechanics without heavy external dependencies.
"""

import numpy as np
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: Dict[str, Any]
    chunk_text: str

class LegalVectorIndex:
    """
    Vector Index for Legal Documents.
    
    Features:
    - Dense vector storage
    - Cosine similarity search
    - Metadata filtering (e.g., by jurisdiction, year)
    - Persistence (save/load)
    """
    
    def __init__(self, index_path: str = "legal_index.pkl"):
        self.index_path = Path(index_path)
        self.vectors: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.chunks: List[str] = []
        self.dim: int = 0
        
    def add_documents(self, 
                     embeddings: np.ndarray, 
                     doc_ids: List[str], 
                     chunks: List[str], 
                     metadatas: List[Dict[str, Any]]):
        """
        Add documents to the index.
        
        Args:
            embeddings: (N, D) numpy array of vectors
            doc_ids: List of unique document identifiers
            chunks: List of text content
            metadatas: List of metadata dictionaries
        """
        if len(embeddings) == 0:
            return
            
        n_samples, dim = embeddings.shape
        
        # Initialize if empty
        if self.vectors is None:
            self.vectors = embeddings
            self.dim = dim
        else:
            if dim != self.dim:
                raise ValueError(f"Embedding dimension mismatch: {dim} != {self.dim}")
            self.vectors = np.vstack([self.vectors, embeddings])
            
        self.ids.extend(doc_ids)
        self.chunks.extend(chunks)
        self.metadata.extend(metadatas)
        
        logger.info(f"Added {n_samples} documents to index. Total: {len(self.ids)}")
        
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5, 
               filter_criteria: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for nearest neighbors with optional metadata filtering.
        
        Args:
            query_vector: (D,) numpy array
            k: Number of results to return
            filter_criteria: Dict of metadata fields to match exact values (e.g. {"court": "Supreme Court"})
        """
        if self.vectors is None:
            logger.warning("Index is empty.")
            return []
            
        # 1. Compute cosine similarity: (A . B) / (|A| |B|)
        # Assuming query_vector is normalized, and self.vectors are normalized optionally.
        # Let's simple perform dot product and manual norm for safety.
        
        q_norm = np.linalg.norm(query_vector)
        v_norms = np.linalg.norm(self.vectors, axis=1)
        
        # Avoid division by zero
        v_norms[v_norms == 0] = 1e-10
        
        scores = (self.vectors @ query_vector) / (v_norms * q_norm)
        
        # 2. Apply Filtering (if any)
        # We need to filter *before* top-k ideally, but for simple numpy imp, we can mask.
        valid_indices = np.arange(len(scores))
        
        if filter_criteria:
            mask = np.ones(len(scores), dtype=bool)
            for key, value in filter_criteria.items():
                # Check each metadata entry
                # This is slow O(N), but acceptable for case study demonstration
                field_mask = np.array([m.get(key) == value for m in self.metadata])
                mask = mask & field_mask
            
            # If nothing matches filter
            if not np.any(mask):
                return []
                
            scores[~mask] = -np.inf
            
        # 3. Get Top-K
        # invalid entires are -inf
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] == -np.inf:
                continue
                
            results.append(SearchResult(
                doc_id=self.ids[idx],
                score=float(scores[idx]),
                metadata=self.metadata[idx],
                chunk_text=self.chunks[idx]
            ))
            
        return results
    
    def save(self):
        """Save index to disk."""
        data = {
            "vectors": self.vectors,
            "ids": self.ids,
            "metadata": self.metadata,
            "chunks": self.chunks,
            "dim": self.dim
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Index saved to {self.index_path}")
        
    def load(self):
        """Load index from disk."""
        if not self.index_path.exists():
            logger.warning(f"Index file {self.index_path} not found.")
            return
            
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            
        self.vectors = data["vectors"]
        self.ids = data["ids"]
        self.metadata = data["metadata"]
        self.chunks = data["chunks"]
        self.dim = data["dim"]
        logger.info(f"Index loaded from {self.index_path}. Size: {len(self.ids)}")

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Utility to normalize vectors to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
