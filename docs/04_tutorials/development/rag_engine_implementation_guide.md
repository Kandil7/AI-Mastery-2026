# 🏗️ Implementation Guide: From Theory to Production Code

This guide connects the theoretical concepts in the AI Engineering Curriculum to their actual implementation in the RAG Engine Mini codebase. Follow along to see how mathematical concepts become production-ready code.

## 📚 Table of Contents

1. [Core Architecture](#core-architecture)
2. [Embedding Implementation](#embedding-implementation)
3. [Vector Store Implementation](#vector-store-implementation)
4. [Hybrid Search Implementation](#hybrid-search-implementation)
5. [Reranking Implementation](#reranking-implementation)
6. [Document Processing Pipeline](#document-processing-pipeline)
7. [API Layer Implementation](#api-layer-implementation)
8. [Testing & Evaluation](#testing--evaluation)

---

## Core Architecture

### Clean Architecture Implementation

The project follows Clean Architecture principles with four main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│                    Thin controllers + DTOs                   │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  Use Cases   │  │     Ports       │  │   Services    │  │
│  │  (Orchestr.) │  │  (Interfaces)   │  │ (Pure Logic)  │  │
│  └──────────────┘  └─────────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│              Entities + Domain Rules (No I/O)                │
├─────────────────────────────────────────────────────────────┤
│                     Adapters Layer                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │  OpenAI  │ │  Qdrant  │ │ Postgres │ │    Redis     │   │
│  │   LLM    │ │  Vector  │ │   Repo   │ │    Cache     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Code Locations:

- **Domain Layer**: `src/domain/` - Contains entities and business rules
- **Application Layer**: `src/application/` - Contains use cases, services, and ports
- **Adapters Layer**: `src/adapters/` - Contains concrete implementations
- **API Layer**: `src/api/` - Contains FastAPI routes and controllers

### Example: Document Entity (Domain Layer)

```python
# src/domain/entities/document.py
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

class Document(BaseModel):
    id: str
    filename: str
    content_hash: str
    user_id: str
    status: DocumentStatus
    chunks_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime
```

### Example: Document Repository Port (Application Layer)

```python
# src/application/ports/document_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.entities.document import Document

class DocumentRepository(ABC):
    @abstractmethod
    async def save(self, document: Document) -> Document:
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str, user_id: str) -> Optional[Document]:
        pass
    
    @abstractmethod
    async def get_by_hash(self, content_hash: str, user_id: str) -> Optional[Document]:
        pass
    
    @abstractmethod
    async def list_user_documents(self, user_id: str) -> List[Document]:
        pass
```

### Example: Document Repository Implementation (Adapters Layer)

```python
# src/adapters/repositories/document_repository_impl.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from src.application.ports.document_repository import DocumentRepository
from src.domain.entities.document import Document, DocumentStatus
from src.adapters.db.models import DocumentModel

class DocumentRepositoryImpl(DocumentRepository):
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def save(self, document: Document) -> Document:
        db_doc = DocumentModel(
            id=document.id,
            filename=document.filename,
            content_hash=document.content_hash,
            user_id=document.user_id,
            status=document.status.value,
            chunks_count=document.chunks_count,
            created_at=document.created_at,
            updated_at=document.updated_at
        )
        
        self.db_session.add(db_doc)
        await self.db_session.commit()
        await self.db_session.refresh(db_doc)
        
        return Document(
            id=db_doc.id,
            filename=db_doc.filename,
            content_hash=db_doc.content_hash,
            user_id=db_doc.user_id,
            status=DocumentStatus(db_doc.status),
            chunks_count=db_doc.chunks_count,
            created_at=db_doc.created_at,
            updated_at=db_doc.updated_at
        )
    
    async def get_by_id(self, document_id: str, user_id: str) -> Optional[Document]:
        stmt = select(DocumentModel).where(
            DocumentModel.id == document_id,
            DocumentModel.user_id == user_id
        )
        result = await self.db_session.execute(stmt)
        db_doc = result.scalar_one_or_none()
        
        if not db_doc:
            return None
            
        return Document(
            id=db_doc.id,
            filename=db_doc.filename,
            content_hash=db_doc.content_hash,
            user_id=db_doc.user_id,
            status=DocumentStatus(db_doc.status),
            chunks_count=db_doc.chunks_count,
            created_at=db_doc.created_at,
            updated_at=db_doc.updated_at
        )
```

---

## Embedding Implementation

### Mathematical Foundation

Embeddings convert text to dense vectors in high-dimensional space. The core concept is:

```
text → tokens → token_embeddings → pooled_representation
```

Cosine similarity measures semantic similarity:
```
sim(A,B) = (A·B)/(||A||₂ × ||B||₂)
```

### Code Implementation

#### Embedding Port (Application Layer)

```python
# src/application/ports/embeddings.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingService(ABC):
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> np.ndarray:
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
```

#### OpenAI Embedding Implementation (Adapters Layer)

```python
# src/adapters/embeddings/openai_embeddings.py
import asyncio
import numpy as np
from typing import List
from openai import AsyncOpenAI
from src.application.ports.embeddings import EmbeddingService

class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dimension = dimension
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        # Batch embeddings for efficiency
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        
        embeddings = []
        for data in response.data:
            embeddings.append(np.array(data.embedding[:self._dimension]))
        
        return embeddings
    
    async def embed_query(self, text: str) -> np.ndarray:
        # For single query, use same endpoint
        response = await self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        
        return np.array(response.data[0].embedding[:self._dimension])
    
    @property
    def dimension(self) -> int:
        return self._dimension
```

#### Embedding Service with Caching (Application Layer)

```python
# src/application/services/embedding_service.py
import hashlib
from typing import List
import numpy as np
from src.application.ports.embeddings import EmbeddingService
from src.application.ports.cache import CacheService

class CachedEmbeddingService(EmbeddingService):
    def __init__(self, embedding_service: EmbeddingService, cache_service: CacheService):
        self.embedding_service = embedding_service
        self.cache_service = cache_service
    
    async def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = []
        
        for text in texts:
            # Create cache key from text
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            cache_key = f"embedding:{text_hash}"
            
            # Try to get from cache first
            cached_embedding = await self.cache_service.get(cache_key)
            if cached_embedding:
                embeddings.append(cached_embedding)
            else:
                # Compute embedding and cache it
                embedding = await self.embedding_service.embed_query(text)
                await self.cache_service.set(cache_key, embedding, ttl=3600)  # 1 hour TTL
                embeddings.append(embedding)
        
        return embeddings
    
    async def embed_query(self, text: str) -> np.ndarray:
        # Same logic as embed_texts but for single text
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"embedding:{text_hash}"
        
        cached_embedding = await self.cache_service.get(cache_key)
        if cached_embedding:
            return cached_embedding
        
        embedding = await self.embedding_service.embed_query(text)
        await self.cache_service.set(cache_key, embedding, ttl=3600)
        return embedding
    
    @property
    def dimension(self) -> int:
        return self.embedding_service.dimension
```

---

## Vector Store Implementation

### Mathematical Foundation

Vector stores enable fast similarity search using approximate nearest neighbor algorithms. The core operation is:

```
find k vectors v in V that minimize distance(query, v)
```

Common distance metrics:
- Cosine: `1 - (A·B)/(||A||×||B||)`
- Euclidean: `||A-B||₂`
- Dot product: `-A·B` (for maximization)

### Code Implementation

#### Vector Store Port (Application Layer)

```python
# src/application/ports/vector_store.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class VectorStore(ABC):
    @abstractmethod
    async def add_vectors(
        self, 
        vectors: List[np.ndarray], 
        payloads: List[dict], 
        ids: List[str]
    ) -> None:
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        pass
    
    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> None:
        pass
```

#### Qdrant Vector Store Implementation (Adapters Layer)

```python
# src/adapters/vector_stores/qdrant_store.py
from typing import List, Tuple, Optional
import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.application.ports.vector_store import VectorStore

class QdrantVectorStore(VectorStore):
    def __init__(
        self, 
        host: str, 
        port: int, 
        collection_name: str, 
        vector_dimension: int,
        api_key: Optional[str] = None
    ):
        self.client = AsyncQdrantClient(host=host, port=port, api_key=api_key)
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
    
    async def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        collections = await self.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if self.collection_name not in collection_names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dimension,
                    distance=models.Distance.COSINE
                )
            )
    
    async def add_vectors(
        self, 
        vectors: List[np.ndarray], 
        payloads: List[dict], 
        ids: List[str]
    ) -> None:
        await self._ensure_collection_exists()
        
        points = [
            models.PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload=payload
            )
            for idx, vector, payload in zip(ids, vectors, payloads)
        ]
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    async def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        # Build filter conditions if provided
        qdrant_filters = None
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            
            if filter_conditions:
                qdrant_filters = models.Filter(must=filter_conditions)
        
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filters,
            with_payload=True
        )
        
        return [
            (str(result.id), result.score, result.payload)
            for result in results
        ]
    
    async def delete_vectors(self, ids: List[str]) -> None:
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[int(id_) for id_ in ids]
            )
        )
```

---

## Hybrid Search Implementation

### Mathematical Foundation

Hybrid search combines vector and keyword search results using Reciprocal Rank Fusion (RRF):

```
RRF_score(d) = Σ(1/(k + rank_method_i(d)))
```

Where k is typically 60, and ranks come from different search methods.

### Code Implementation

#### Hybrid Search Service (Application Layer)

```python
# src/application/services/hybrid_search_service.py
from typing import List, Tuple, Optional
import numpy as np
from src.application.ports.vector_store import VectorStore
from src.application.ports.keyword_store import KeywordStore
from src.application.ports.embeddings import EmbeddingService

class HybridSearchService:
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embedding_service: EmbeddingService
    ):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.embedding_service = embedding_service
    
    async def search(
        self,
        query: str,
        k_vector: int = 20,
        k_keyword: int = 20,
        rerank_top_n: int = 10,
        filters: Optional[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        # Perform vector search
        query_embedding = await self.embedding_service.embed_query(query)
        vector_results = await self.vector_store.search(
            query_vector=query_embedding,
            top_k=k_vector,
            filters=filters
        )
        
        # Perform keyword search
        keyword_results = await self.keyword_store.search(
            query_text=query,
            top_k=k_keyword,
            filters=filters
        )
        
        # Apply RRF fusion
        fused_results = self._rrf_fusion(vector_results, keyword_results)
        
        # Return top N results after fusion
        return fused_results[:rerank_top_n]
    
    def _rrf_fusion(
        self, 
        vector_results: List[Tuple[str, float, dict]], 
        keyword_results: List[Tuple[str, float, dict]],
        k: int = 60
    ) -> List[Tuple[str, float, dict]]:
        """
        Apply Reciprocal Rank Fusion to combine results from different search methods
        """
        # Create dictionaries to store RRF scores
        rrf_scores = {}
        
        # Process vector results
        for rank, (doc_id, score, payload) in enumerate(vector_results, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0, 'payload': payload}
            rrf_scores[doc_id]['score'] += 1.0 / (k + rank)
        
        # Process keyword results
        for rank, (doc_id, score, payload) in enumerate(keyword_results, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0, 'payload': payload}
            rrf_scores[doc_id]['score'] += 1.0 / (k + rank)
        
        # Convert to list and sort by RRF score
        fused_results = [
            (doc_id, info['score'], info['payload'])
            for doc_id, info in rrf_scores.items()
        ]
        
        # Sort by score in descending order
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
```

#### Keyword Store Port (Application Layer)

```python
# src/application/ports/keyword_store.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

class KeywordStore(ABC):
    @abstractmethod
    async def index_document(self, doc_id: str, text: str, metadata: dict) -> None:
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_text: str, 
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        pass
```

#### Postgres Keyword Store Implementation (Adapters Layer)

```python
# src/adapters/keyword_stores/postgres_keyword_store.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Tuple, Optional
from src.application.ports.keyword_store import KeywordStore

class PostgresKeywordStore(KeywordStore):
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    async def index_document(self, doc_id: str, text: str, metadata: dict) -> None:
        # Create tsvector for full-text search
        sql = """
        INSERT INTO keyword_index (doc_id, content, metadata, search_vector)
        VALUES (:doc_id, :content, :metadata, to_tsvector('english', :content))
        ON CONFLICT (doc_id) 
        DO UPDATE SET 
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            search_vector = to_tsvector('english', EXCLUDED.content);
        """
        
        await self.db_session.execute(
            text(sql),
            {
                "doc_id": doc_id,
                "content": text,
                "metadata": metadata
            }
        )
        await self.db_session.commit()
    
    async def search(
        self, 
        query_text: str, 
        top_k: int = 10,
        filters: Optional[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        # Build search query with ranking
        sql = """
        SELECT 
            doc_id,
            ts_rank(search_vector, plainto_tsquery('english', :query)) as rank,
            metadata
        FROM keyword_index
        WHERE search_vector @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :limit;
        """
        
        result = await self.db_session.execute(
            text(sql),
            {
                "query": query_text,
                "limit": top_k
            }
        )
        
        rows = result.fetchall()
        return [(row.doc_id, row.rank, row.metadata) for row in rows]
    
    async def delete_document(self, doc_id: str) -> None:
        sql = "DELETE FROM keyword_index WHERE doc_id = :doc_id;"
        await self.db_session.execute(text(sql), {"doc_id": doc_id})
        await self.db_session.commit()
```

---

## Reranking Implementation

### Mathematical Foundation

Cross-encoder reranking computes the relevance of query-document pairs:

```
P(relevant|q,d) = sigmoid(W_classifier × f([q;d]) + b_classifier)
```

Where f is the transformer encoder output for concatenated query and document.

### Code Implementation

#### Reranker Service (Application Layer)

```python
# src/application/services/reranker_service.py
from typing import List, Tuple
from sentence_transformers import CrossEncoder
import torch
from src.application.ports.reranker import Reranker

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)
        self.device = device
    
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents based on relevance to query using cross-encoder
        """
        # Create query-document pairs
        query_doc_pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(query_doc_pairs)
        
        # Convert to list of (document, score) tuples
        doc_score_pairs = [(doc, float(score)) for doc, score in zip(documents, scores)]
        
        # Sort by score in descending order and return top_k
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return doc_score_pairs[:top_k]
```

#### Reranker Port (Application Layer)

```python
# src/application/ports/reranker.py
from abc import ABC, abstractmethod
from typing import List, Tuple

class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        pass
```

---

## Document Processing Pipeline

### Code Implementation

#### Document Processing Service (Application Layer)

```python
# src/application/services/document_processing_service.py
import hashlib
from typing import List
import numpy as np
from src.domain.entities.document import Document, DocumentStatus
from src.application.ports.document_repository import DocumentRepository
from src.application.ports.text_extractor import TextExtractor
from src.application.services.chunking_service import ChunkingService
from src.application.ports.vector_store import VectorStore
from src.application.ports.keyword_store import KeywordStore
from src.application.ports.embeddings import EmbeddingService

class DocumentProcessingService:
    def __init__(
        self,
        document_repo: DocumentRepository,
        text_extractor: TextExtractor,
        chunking_service: ChunkingService,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embedding_service: EmbeddingService
    ):
        self.document_repo = document_repo
        self.text_extractor = text_extractor
        self.chunking_service = chunking_service
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.embedding_service = embedding_service
    
    async def process_document(self, document: Document, file_path: str) -> None:
        """
        Process a document: extract text → chunk → embed → store
        """
        try:
            # Update document status
            document.status = DocumentStatus.PROCESSING
            await self.document_repo.save(document)
            
            # Extract text from document
            text_content = await self.text_extractor.extract(file_path)
            
            # Chunk the text
            chunks = self.chunking_service.chunk_text(text_content)
            
            # Prepare for embedding and storage
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_metadatas = [
                {
                    "document_id": document.id,
                    "chunk_index": i,
                    "user_id": document.user_id,
                    "page_number": getattr(chunk, 'page_number', None)
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Generate embeddings
            embeddings = await self.embedding_service.embed_texts(chunk_texts)
            
            # Generate unique IDs for chunks
            chunk_ids = [f"{document.id}_chunk_{i}" for i in range(len(chunks))]
            
            # Store in vector store
            await self.vector_store.add_vectors(
                vectors=embeddings,
                payloads=chunk_metadatas,
                ids=chunk_ids
            )
            
            # Store in keyword store
            for chunk_id, chunk_text, chunk_metadata in zip(chunk_ids, chunk_texts, chunk_metadatas):
                await self.keyword_store.index_document(
                    doc_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_metadata
                )
            
            # Update document status
            document.status = DocumentStatus.INDEXED
            document.chunks_count = len(chunks)
            await self.document_repo.save(document)
            
        except Exception as e:
            # Update document status to failed
            document.status = DocumentStatus.FAILED
            await self.document_repo.save(document)
            raise e
```

#### Chunking Service (Application Layer)

```python
# src/application/services/chunking_service.py
from typing import List
from dataclasses import dataclass
import tiktoken

@dataclass
class TextChunk:
    text: str
    start_idx: int
    end_idx: int
    page_number: int = None

class ChunkingService:
    def __init__(self, chunk_size: int = 512, overlap: int = 50, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks based on token count
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        start_idx = 0
        while start_idx < len(tokens):
            # Determine the end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Decode the token slice back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk object
            chunk = TextChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx
            )
            chunks.append(chunk)
            
            # Move start index forward, accounting for overlap
            start_idx = end_idx - self.overlap
            
            # Prevent infinite loop if overlap is too large
            if start_idx >= end_idx:
                start_idx = end_idx
        
        return chunks
```

---

## API Layer Implementation

### FastAPI Routes

```python
# src/api/routes/documents.py
from fastapi import APIRouter, Depends, UploadFile, File
from typing import List
import uuid
from datetime import datetime
from src.api.dependencies import get_document_use_case
from src.application.use_cases.document_use_cases import DocumentUseCase
from src.api.schemas.document import DocumentResponse, UploadResponse

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_use_case: DocumentUseCase = Depends(get_document_use_case)
):
    """
    Upload a document for processing and indexing
    """
    # Generate document ID
    document_id = str(uuid.uuid4())
    
    # Save file temporarily
    file_path = f"./uploads/{document_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Process document asynchronously
    await document_use_case.upload_document(
        document_id=document_id,
        filename=file.filename,
        file_path=file_path,
        user_id="current_user_id"  # In practice, get from auth
    )
    
    return UploadResponse(
        document_id=document_id,
        status="queued"
    )

@router.get("/{document_id}/status", response_model=DocumentResponse)
async def get_document_status(
    document_id: str,
    document_use_case: DocumentUseCase = Depends(get_document_use_case)
):
    """
    Get the processing status of a document
    """
    document = await document_use_case.get_document(document_id, "current_user_id")
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status.value,
        chunks_count=document.chunks_count,
        created_at=document.created_at
    )
```

#### Query API Routes

```python
# src/api/routes/queries.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from src.api.dependencies import get_query_use_case
from src.application.use_cases.query_use_cases import QueryUseCase
from src.api.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/queries", tags=["Queries"])

@router.post("/ask-hybrid", response_model=QueryResponse)
async def ask_hybrid(
    request: QueryRequest,
    query_use_case: QueryUseCase = Depends(get_query_use_case)
):
    """
    Ask a question using hybrid search (vector + keyword)
    """
    try:
        result = await query_use_case.ask_hybrid(
            question=request.question,
            user_id="current_user_id",  # In practice, get from auth
            k_vector=request.k_vec or 20,
            k_keyword=request.k_kw or 20,
            rerank_top_n=request.rerank_top_n or 8,
            document_id=request.document_id
        )
        
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            query_time=result.query_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Testing & Evaluation

### Unit Tests

```python
# tests/unit/test_hybrid_search_service.py
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
from src.application.services.hybrid_search_service import HybridSearchService

@pytest.mark.asyncio
async def test_hybrid_search_rrf_fusion():
    # Create mock services
    mock_vector_store = Mock()
    mock_keyword_store = Mock()
    mock_embedding_service = Mock()
    
    # Set up mock return values
    mock_embedding_service.embed_query = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]))
    
    # Mock vector search results: (doc_id, score, payload)
    vector_results = [
        ("doc1", 0.9, {"text": "first result"}),
        ("doc2", 0.8, {"text": "second result"})
    ]
    mock_vector_store.search = AsyncMock(return_value=vector_results)
    
    # Mock keyword search results
    keyword_results = [
        ("doc2", 0.7, {"text": "second result"}),
        ("doc3", 0.6, {"text": "third result"})
    ]
    mock_keyword_store.search = AsyncMock(return_value=keyword_results)
    
    # Create service instance
    service = HybridSearchService(
        vector_store=mock_vector_store,
        keyword_store=mock_keyword_store,
        embedding_service=mock_embedding_service
    )
    
    # Call the search method
    results = await service.search("test query", k_vector=2, k_keyword=2)
    
    # Assertions
    assert len(results) == 3  # Should have 3 unique docs: doc1, doc2, doc3
    # doc2 should have highest score due to appearing in both results
    assert results[0][0] == "doc2"  # doc2 should be first
```

### Integration Tests

```python
# tests/integration/test_document_ingestion.py
import pytest
import tempfile
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.adapters.repositories.document_repository_impl import DocumentRepositoryImpl
from src.adapters.db.models import Base

@pytest.mark.asyncio
async def test_document_ingestion_pipeline():
    # Setup database
    engine = create_async_engine("sqlite+aiosqlite:///test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with async_session() as session:
        # Create repository
        repo = DocumentRepositoryImpl(session)
        
        # Create a test document
        from src.domain.entities.document import Document, DocumentStatus
        from datetime import datetime
        
        test_doc = Document(
            id="test_doc_123",
            filename="test.pdf",
            content_hash="abc123",
            user_id="test_user",
            status=DocumentStatus.UPLOADED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Save document
        saved_doc = await repo.save(test_doc)
        
        # Retrieve document
        retrieved_doc = await repo.get_by_id("test_doc_123", "test_user")
        
        # Assertions
        assert retrieved_doc.id == saved_doc.id
        assert retrieved_doc.filename == "test.pdf"
        assert retrieved_doc.status == DocumentStatus.UPLOADED
```

---

## 🎯 Key Takeaways

This implementation guide demonstrates how theoretical concepts become production-ready code:

1. **Abstraction**: Use ports and adapters to decouple business logic from infrastructure
2. **Efficiency**: Implement batching and caching for expensive operations like embeddings
3. **Reliability**: Include proper error handling and status tracking
4. **Scalability**: Design for asynchronous processing and horizontal scaling
5. **Testability**: Structure code to be easily testable with mocks and dependency injection

By following this pattern, you can implement any RAG component while maintaining clean architecture and production readiness.