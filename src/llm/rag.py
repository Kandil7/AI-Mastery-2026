"""
RAG (Retrieval-Augmented Generation) Pipeline
==============================================
Components for building production RAG systems.

Pipeline:
    Query → Embed → Retrieve → Rerank → Context → LLM → Response

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import re
import hashlib


@dataclass
class Document:
    """Document with content and metadata."""
    content: str
    id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class RetrievalResult:
    """Result from retrieval."""
    document: Document
    score: float
    rank: int


# ============================================================
# CHUNKING STRATEGIES
# ============================================================

class TextChunker:
    """
    Text chunking for RAG pipelines.
    
    Strategies:
    - Fixed size: Split by token/character count
    - Semantic: Split at sentence/paragraph boundaries
    - Recursive: Try larger delimiters first, then smaller
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50,
                 strategy: str = 'fixed'):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Document]:
        """Split text into chunks."""
        if self.strategy == 'fixed':
            return self._fixed_chunk(text, metadata or {})
        elif self.strategy == 'semantic':
            return self._semantic_chunk(text, metadata or {})
        elif self.strategy == 'recursive':
            return self._recursive_chunk(text, metadata or {})
        else:
            return self._fixed_chunk(text, metadata or {})
    
    def _fixed_chunk(self, text: str, metadata: Dict) -> List[Document]:
        """Fixed-size chunking with overlap."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(Document(
                content=chunk_text,
                metadata={**metadata, 'chunk_idx': len(chunks)}
            ))
        
        return chunks
    
    def _semantic_chunk(self, text: str, metadata: Dict) -> List[Document]:
        """Chunk at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            
            if current_size + words > self.chunk_size and current_chunk:
                chunks.append(Document(
                    content=' '.join(current_chunk),
                    metadata={**metadata, 'chunk_idx': len(chunks)}
                ))
                # Overlap: keep some sentences
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else []
                current_chunk = overlap_sentences
                current_size = sum(len(s.split()) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_size += words
        
        if current_chunk:
            chunks.append(Document(
                content=' '.join(current_chunk),
                metadata={**metadata, 'chunk_idx': len(chunks)}
            ))
        
        return chunks
    
    def _recursive_chunk(self, text: str, metadata: Dict) -> List[Document]:
        """Recursive splitting with multiple delimiters."""
        delimiters = ['\n\n', '\n', '. ', ' ']
        return self._recursive_split(text, delimiters, metadata)
    
    def _recursive_split(self, text: str, delimiters: List[str], 
                         metadata: Dict) -> List[Document]:
        """Helper for recursive chunking."""
        if len(text.split()) <= self.chunk_size:
            return [Document(content=text, metadata=metadata)]
        
        if not delimiters:
            return self._fixed_chunk(text, metadata)
        
        delimiter = delimiters[0]
        parts = text.split(delimiter)
        
        chunks = []
        current = []
        current_size = 0
        
        for part in parts:
            part_size = len(part.split())
            
            if current_size + part_size > self.chunk_size:
                if current:
                    combined = delimiter.join(current)
                    if len(combined.split()) > self.chunk_size:
                        chunks.extend(self._recursive_split(combined, delimiters[1:], metadata))
                    else:
                        chunks.append(Document(content=combined, 
                                              metadata={**metadata, 'chunk_idx': len(chunks)}))
                current = [part]
                current_size = part_size
            else:
                current.append(part)
                current_size += part_size
        
        if current:
            combined = delimiter.join(current)
            chunks.append(Document(content=combined, 
                                  metadata={**metadata, 'chunk_idx': len(chunks)}))
        
        return chunks


# ============================================================
# EMBEDDING
# ============================================================

class EmbeddingModel:
    """
    Embedding model interface.
    
    In production, use:
    - OpenAI: text-embedding-3-small/large
    - Cohere: embed-english-v3.0
    - Local: sentence-transformers
    """
    
    def __init__(self, model_fn: Optional[Callable] = None, dim: int = 384):
        self.model_fn = model_fn
        self.dim = dim
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed list of texts."""
        if self.model_fn:
            return self.model_fn(texts)
        else:
            # Dummy embeddings for demo
            return np.random.randn(len(texts), self.dim)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed([query])[0]
    
    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Embed documents and attach embeddings."""
        texts = [doc.content for doc in documents]
        embeddings = self.embed(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        return documents


# ============================================================
# RETRIEVAL
# ============================================================

class Retriever:
    """
    Document retriever with multiple strategies.
    
    Strategies:
    - Dense: Semantic search with embeddings
    - Sparse: BM25/TF-IDF keyword matching
    - Hybrid: Combine dense and sparse
    """
    
    def __init__(self, embedding_model: EmbeddingModel, 
                 strategy: str = 'dense'):
        self.embedding_model = embedding_model
        self.strategy = strategy
        self.documents: List[Document] = []
        self.index: Optional[np.ndarray] = None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the retriever."""
        documents = self.embedding_model.embed_documents(documents)
        self.documents.extend(documents)
        
        # Build index
        embeddings = [doc.embedding for doc in self.documents]
        self.index = np.vstack(embeddings) if embeddings else None
    
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k documents."""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model.embed_query(query)
        
        # Cosine similarity
        similarities = self.index @ query_embedding
        similarities /= (np.linalg.norm(self.index, axis=1) * np.linalg.norm(query_embedding))
        
        # Top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(similarities[idx]),
                rank=rank
            ))
        
        return results


# ============================================================
# RERANKING
# ============================================================

class Reranker:
    """
    Cross-encoder reranker.
    
    Unlike bi-encoders (separate embedding), cross-encoders
    jointly encode query-document pairs for better relevance.
    
    Use after initial retrieval to refine top-k results.
    """
    
    def __init__(self, rerank_fn: Optional[Callable] = None):
        self.rerank_fn = rerank_fn
    
    def rerank(self, query: str, results: List[RetrievalResult],
               top_k: int = 3) -> List[RetrievalResult]:
        """Rerank results."""
        if self.rerank_fn:
            pairs = [(query, r.document.content) for r in results]
            scores = self.rerank_fn(pairs)
            
            for result, score in zip(results, scores):
                result.score = score
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        for i, result in enumerate(results[:top_k]):
            result.rank = i
        
        return results[:top_k]


# ============================================================
# CONTEXT ASSEMBLY
# ============================================================

class ContextAssembler:
    """
    Assemble retrieved documents into LLM context.
    
    Strategies:
    - Simple: Concatenate all documents
    - Stuffing: Fit as many as possible in context window
    - Map-Reduce: Process each doc separately, then combine
    """
    
    def __init__(self, max_tokens: int = 2048, strategy: str = 'stuffing'):
        self.max_tokens = max_tokens
        self.strategy = strategy
    
    def assemble(self, query: str, results: List[RetrievalResult]) -> str:
        """Assemble context from retrieval results."""
        if self.strategy == 'stuffing':
            return self._stuffing(query, results)
        else:
            return self._simple(query, results)
    
    def _simple(self, query: str, results: List[RetrievalResult]) -> str:
        """Simple concatenation."""
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Document {i+1}]\n{result.document.content}")
        
        return '\n\n'.join(context_parts)
    
    def _stuffing(self, query: str, results: List[RetrievalResult]) -> str:
        """Fit documents within token limit."""
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(results):
            doc_tokens = len(result.document.content.split())
            
            if current_tokens + doc_tokens > self.max_tokens:
                break
            
            context_parts.append(f"[Document {i+1}]\n{result.document.content}")
            current_tokens += doc_tokens
        
        return '\n\n'.join(context_parts)


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    Complete RAG pipeline.
    
    Example:
        >>> rag = RAGPipeline()
        >>> rag.add_documents(documents)
        >>> response = rag.query("What is RAG?")
    """
    
    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 retriever: Optional[Retriever] = None,
                 reranker: Optional[Reranker] = None,
                 context_assembler: Optional[ContextAssembler] = None,
                 llm_fn: Optional[Callable] = None):
        
        self.embedding_model = embedding_model or EmbeddingModel()
        self.retriever = retriever or Retriever(self.embedding_model)
        self.reranker = reranker
        self.context_assembler = context_assembler or ContextAssembler()
        self.llm_fn = llm_fn
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the pipeline."""
        self.retriever.add_documents(documents)
    
    def query(self, query: str, k: int = 5, 
              return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Returns:
            Dict with 'answer', 'sources', 'context'
        """
        # Retrieve
        results = self.retriever.retrieve(query, k=k)
        
        # Rerank (optional)
        if self.reranker:
            results = self.reranker.rerank(query, results)
        
        # Assemble context
        context = self.context_assembler.assemble(query, results)
        
        # Generate answer
        if self.llm_fn:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            answer = self.llm_fn(prompt)
        else:
            answer = f"[Demo] Based on {len(results)} documents about: {query}"
        
        response = {'answer': answer, 'context': context}
        
        if return_sources:
            response['sources'] = [
                {'id': r.document.id, 'content': r.document.content[:200], 'score': r.score}
                for r in results
            ]
        
        return response


__all__ = [
    'Document', 'RetrievalResult',
    'TextChunker', 'EmbeddingModel', 'Retriever', 'Reranker',
    'ContextAssembler', 'RAGPipeline'
]
