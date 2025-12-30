"""
RAG (Retrieval-Augmented Generation) Module

This module implements RAG systems with various retrieval strategies,
including dense retrieval, sparse retrieval, and hybrid approaches.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enumeration of retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    BM25 = "bm25"
    HNSW = "hnsw"


@dataclass
class Document:
    """Represents a document in the RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""
    document: Document
    score: float
    rank: int


class DenseRetriever:
    """
    Dense retrieval using sentence transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the retriever."""
        self.documents.extend(documents)
        
        # Encode all documents
        contents = [doc.content for doc in documents]
        new_embeddings = self.encoder.encode(contents, convert_to_numpy=True)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Rebuild index
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index for fast retrieval."""
        if self.embeddings is not None:
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings.astype('float32'))
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=i + 1
                ))
        
        return results


class SparseRetriever:
    """
    Sparse retrieval using TF-IDF.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.documents: List[Document] = []
        self.tfidf_matrix = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the retriever."""
        self.documents.extend(documents)
        
        # Extract content
        contents = [doc.content for doc in documents]
        
        # Fit or transform the TF-IDF matrix
        if self.tfidf_matrix is None:
            self.tfidf_matrix = self.vectorizer.fit_transform(contents)
        else:
            new_tfidf = self.vectorizer.transform(contents)
            from scipy.sparse import vstack
            self.tfidf_matrix = vstack([self.tfidf_matrix, new_tfidf])
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        if self.tfidf_matrix is None or len(self.documents) == 0:
            return []
        
        # Transform query
        query_tfidf = self.vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.documents):
                results.append(RetrievalResult(
                    document=self.documents[idx],
                    score=float(similarities[idx]),
                    rank=i + 1
                ))
        
        return results


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.
    """
    
    def __init__(self, dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both retrievers."""
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve using both methods and combine results."""
        dense_results = self.dense_retriever.retrieve(query, k * 2)  # Get more results for combination
        sparse_results = self.sparse_retriever.retrieve(query, k * 2)
        
        # Create a mapping from document ID to combined score
        doc_scores: Dict[str, float] = {}
        
        # Add dense scores (normalize to 0-1 range)
        if dense_results:
            max_dense_score = max(r.score for r in dense_results) if dense_results else 1
            for result in dense_results:
                doc_scores[result.document.id] = result.score / max_dense_score * self.dense_weight
        
        # Add sparse scores (normalize to 0-1 range)
        if sparse_results:
            max_sparse_score = max(r.score for r in sparse_results) if sparse_results else 1
            for result in sparse_results:
                if result.document.id in doc_scores:
                    doc_scores[result.document.id] += result.score / max_sparse_score * self.sparse_weight
                else:
                    doc_scores[result.document.id] = result.score / max_sparse_score * self.sparse_weight
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for i, (doc_id, score) in enumerate(sorted_docs):
            # Find the document in our collection
            doc = next((d for d in self.dense_retriever.documents if d.id == doc_id), None)
            if doc:
                results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    rank=i + 1
                ))
        
        return results


class RAGModel:
    """
    RAG model combining retrieval and generation.
    """
    
    def __init__(
        self, 
        retriever_strategy: RetrievalStrategy,
        generator_model: str = "gpt2", 
        **retriever_kwargs
    ):
        self.retriever_strategy = retriever_strategy
        
        # Initialize retriever based on strategy
        if retriever_strategy == RetrievalStrategy.DENSE:
            self.retriever = DenseRetriever(**retriever_kwargs)
        elif retriever_strategy == RetrievalStrategy.SPARSE:
            self.retriever = SparseRetriever()
        elif retriever_strategy == RetrievalStrategy.HYBRID:
            self.retriever = HybridRetriever(**retriever_kwargs)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {retriever_strategy}")
        
        # Initialize generator
        self.generator_model = generator_model
        self.generator = pipeline(
            "text-generation", 
            model=generator_model,
            tokenizer=generator_model,
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.documents: List[Document] = []
    
    def add_documents(self, documents: List[Union[Document, Dict[str, Any]]]):
        """Add documents to the RAG system."""
        processed_docs = []
        
        for doc in documents:
            if isinstance(doc, dict):
                # Convert dict to Document
                processed_docs.append(Document(
                    id=doc.get('id', str(len(self.documents))),
                    content=doc['content'],
                    metadata=doc.get('metadata', {})
                ))
            else:
                processed_docs.append(doc)
        
        self.documents.extend(processed_docs)
        self.retriever.add_documents(processed_docs)
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        return self.retriever.retrieve(query, k)
    
    def generate(self, query: str, k: int = 3, max_length: int = 200) -> str:
        """
        Generate a response using retrieved documents.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            max_length: Maximum length of generated response
            
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, k)
        
        if not retrieved_docs:
            # If no documents found, generate based on query alone
            prompt = f"Question: {query}\n\nI don't have specific information to answer this question."
            return self.generator(prompt, max_length=max_length)[0]['generated_text']
        
        # Construct context from retrieved documents
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(f"Document {doc.rank}: {doc.document.content}")
        
        context = "\n\n".join(context_parts)
        
        # Construct the full prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        generated = self.generator(
            prompt, 
            max_length=max_length,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        # Extract just the answer part
        full_text = generated[0]['generated_text']
        answer_start = full_text.find("Answer:") + len("Answer:")
        answer = full_text[answer_start:].strip()
        
        return answer
    
    def query(self, query: str, k: int = 3, max_length: int = 200) -> Dict[str, Any]:
        """
        Complete RAG query with both retrieval and generation.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            max_length: Maximum length of generated response
            
        Returns:
            Dictionary with results
        """
        # Retrieve documents
        retrieved_docs = self.retrieve(query, k)
        
        # Generate response
        response = self.generate(query, k, max_length)
        
        return {
            "query": query,
            "retrieved_documents": [
                {
                    "id": doc.document.id,
                    "content": doc.document.content,
                    "score": doc.score,
                    "rank": doc.rank,
                    "metadata": doc.document.metadata
                }
                for doc in retrieved_docs
            ],
            "response": response,
            "retriever_strategy": self.retriever_strategy.value
        }


class AdvancedRAGModel(RAGModel):
    """
    Advanced RAG model with additional features like query rewriting and reranking.
    """
    
    def __init__(
        self, 
        retriever_strategy: RetrievalStrategy,
        generator_model: str = "gpt2",
        use_query_rewriting: bool = True,
        use_reranking: bool = True,
        **retriever_kwargs
    ):
        super().__init__(retriever_strategy, generator_model, **retriever_kwargs)
        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking
        
        # Initialize query rewriting model if needed
        if use_query_rewriting:
            self.query_rewriter = pipeline(
                "text2text-generation",
                model="facebook/bart-large-cnn",  # Example model for query rewriting
                device=0 if torch.cuda.is_available() else -1
            )
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite the query to improve retrieval."""
        if not self.use_query_rewriting:
            return query
        
        # Simple query rewriting using a text generation model
        # In practice, you might use a specialized query rewriting model
        prompt = f"Rewrite the following question to be more specific for document search: {query}"
        
        rewritten = self.query_rewriter(
            prompt,
            max_length=100,
            pad_token_id=self.query_rewriter.tokenizer.eos_token_id
        )
        
        # Extract the rewritten query
        rewritten_text = rewritten[0]['generated_text']
        return rewritten_text
    
    def rerank_documents(self, query: str, documents: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank documents based on query relevance."""
        if not self.use_reranking or not documents:
            return documents
        
        # Simple reranking based on lexical overlap
        query_words = set(re.findall(r'\w+', query.lower()))
        
        for result in documents:
            doc_words = set(re.findall(r'\w+', result.document.content.lower()))
            overlap = len(query_words.intersection(doc_words))
            result.score += overlap * 0.01  # Small boost for lexical overlap
        
        # Re-sort by updated scores
        documents.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(documents):
            result.rank = i + 1
        
        return documents
    
    def query(self, query: str, k: int = 3, max_length: int = 200) -> Dict[str, Any]:
        """
        Advanced RAG query with query rewriting and reranking.
        """
        # Rewrite query if enabled
        if self.use_query_rewriting:
            rewritten_query = self.rewrite_query(query)
            logger.info(f"Original query: {query}")
            logger.info(f"Rewritten query: {rewritten_query}")
        else:
            rewritten_query = query
        
        # Retrieve documents
        retrieved_docs = self.retrieve(rewritten_query, k * 2)  # Retrieve more for reranking
        
        # Rerank documents if enabled
        if self.use_reranking:
            retrieved_docs = self.rerank_documents(rewritten_query, retrieved_docs)
        
        # Take top k after reranking
        retrieved_docs = retrieved_docs[:k]
        
        # Generate response
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(f"Document {doc.rank}: {doc.document.content}")
        
        context = "\n\n".join(context_parts)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        generated = self.generator(
            prompt, 
            max_length=max_length,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        full_text = generated[0]['generated_text']
        answer_start = full_text.find("Answer:") + len("Answer:")
        answer = full_text[answer_start:].strip()
        
        return {
            "query": query,
            "rewritten_query": rewritten_query if self.use_query_rewriting else query,
            "retrieved_documents": [
                {
                    "id": doc.document.id,
                    "content": doc.document.content,
                    "score": doc.score,
                    "rank": doc.rank,
                    "metadata": doc.document.metadata
                }
                for doc in retrieved_docs
            ],
            "response": answer,
            "retriever_strategy": self.retriever_strategy.value
        }


class GraphRAGModel:
    """
    Graph-based RAG model that uses knowledge graphs for retrieval.
    """
    
    def __init__(self, generator_model: str = "gpt2"):
        self.generator_model = generator_model
        self.generator = pipeline(
            "text-generation", 
            model=generator_model,
            tokenizer=generator_model,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Simple in-memory graph representation
        self.entities: Dict[str, Any] = {}  # entity_id -> entity_info
        self.relations: List[Dict[str, Any]] = []  # list of relations
    
    def add_entity(self, entity_id: str, entity_info: Dict[str, Any]):
        """Add an entity to the knowledge graph."""
        self.entities[entity_id] = {
            "id": entity_id,
            "info": entity_info,
            "relations": []
        }
    
    def add_relation(self, subject: str, predicate: str, obj: str, metadata: Dict[str, Any] = None):
        """Add a relation to the knowledge graph."""
        if metadata is None:
            metadata = {}
        
        relation = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "metadata": metadata
        }
        
        self.relations.append(relation)
        
        # Add to entity relations
        if subject in self.entities:
            self.entities[subject]["relations"].append(relation)
        if obj in self.entities:
            self.entities[obj]["relations"].append(relation)
    
    def retrieve_graph_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant graph context for a query."""
        # Simple approach: find entities that match query terms
        query_lower = query.lower()
        matched_entities = []
        
        for entity_id, entity_info in self.entities.items():
            # Check if query terms match entity info
            entity_text = f"{entity_id} {entity_info['info']}".lower()
            if any(term in entity_text for term in query_lower.split()):
                matched_entities.append(entity_id)
        
        # Build context from matched entities and their relations
        context_parts = []
        
        for entity_id in matched_entities[:k]:
            entity = self.entities[entity_id]
            context_parts.append(f"Entity: {entity_id}")
            context_parts.append(f"Info: {entity['info']}")
            
            # Add related entities
            for relation in entity['relations']:
                context_parts.append(f"Relation: {relation['subject']} {relation['predicate']} {relation['object']}")
        
        return "\n".join(context_parts)
    
    def query(self, query: str, k: int = 3, max_length: int = 200) -> Dict[str, Any]:
        """Query the Graph RAG model."""
        # Retrieve graph context
        graph_context = self.retrieve_graph_context(query, k)
        
        # Construct prompt
        prompt = f"Knowledge Graph Context:\n{graph_context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Generate response
        generated = self.generator(
            prompt, 
            max_length=max_length,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        full_text = generated[0]['generated_text']
        answer_start = full_text.find("Answer:") + len("Answer:")
        answer = full_text[answer_start:].strip()
        
        return {
            "query": query,
            "graph_context": graph_context,
            "response": answer,
            "model_type": "GraphRAG"
        }


def create_rag_model(
    strategy: RetrievalStrategy, 
    model_name: str = "gpt2", 
    **kwargs
) -> Union[RAGModel, AdvancedRAGModel, GraphRAGModel]:
    """
    Factory function to create RAG models.
    
    Args:
        strategy: Retrieval strategy
        model_name: Name of the generator model
        **kwargs: Additional arguments
        
    Returns:
        RAG model instance
    """
    if strategy == RetrievalStrategy.HYBRID:
        return AdvancedRAGModel(
            retriever_strategy=strategy,
            generator_model=model_name,
            **kwargs
        )
    elif strategy == RetrievalStrategy.DENSE:
        return RAGModel(
            retriever_strategy=strategy,
            generator_model=model_name,
            **kwargs
        )
    elif strategy == RetrievalStrategy.SPARSE:
        return RAGModel(
            retriever_strategy=strategy,
            generator_model=model_name,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


# Example usage
if __name__ == "__main__":
    # Create sample documents
    sample_docs = [
        Document(
            id="doc1",
            content="Machine learning is a method of data analysis that automates analytical model building.",
            metadata={"source": "wikipedia", "topic": "ML"}
        ),
        Document(
            id="doc2", 
            content="Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            metadata={"source": "wikipedia", "topic": "Deep Learning"}
        ),
        Document(
            id="doc3",
            content="Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
            metadata={"source": "wikipedia", "topic": "NLP"}
        )
    ]
    
    # Create and test a dense RAG model
    rag_model = create_rag_model(RetrievalStrategy.DENSE)
    rag_model.add_documents(sample_docs)
    
    # Query the model
    result = rag_model.query("What is machine learning?")
    print("RAG Response:", result["response"])
    print("Retrieved Documents:", len(result["retrieved_documents"]))
    
    # Test advanced RAG with query rewriting and reranking
    advanced_rag = AdvancedRAGModel(
        retriever_strategy=RetrievalStrategy.HYBRID,
        use_query_rewriting=True,
        use_reranking=True
    )
    advanced_rag.add_documents(sample_docs)
    
    advanced_result = advanced_rag.query("Explain deep learning concepts")
    print("\nAdvanced RAG Response:", advanced_result["response"])
    print("Retrieved Documents:", len(advanced_result["retrieved_documents"]))