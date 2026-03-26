"""
Complete RAG Pipeline - From Ingestion to Generation

Following RAG Pipeline Guide 2026 - Complete Implementation

This module ties together all components:
- Data Ingestion
- Text Processing & Chunking
- Embedding Generation
- Hybrid Retrieval
- LLM Generation
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Data paths
    datasets_path: str = "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets"
    output_path: str = "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data"

    # Embedding settings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_device: str = "cpu"
    embedding_cache_dir: Optional[str] = None

    # Vector database settings
    vector_db_type: str = "memory"  # 'qdrant', 'chroma', 'memory'
    vector_size: int = 768

    # Retrieval settings
    retrieval_top_k: int = 50
    rerank_top_k: int = 5

    # BM25 settings
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Hybrid weights
    hybrid_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.6,
            "bm25": 0.4,
        }
    )

    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000

    # Generation settings
    max_context_chunks: int = 5
    include_citations: bool = True

    # Processing
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class QueryResult:
    """Result from a complete RAG query."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    tokens_used: int
    model: str


class CompleteRAGPipeline:
    """
    Complete RAG pipeline combining all components.

    Architecture:
    1. Data Ingestion → Load and process documents
    2. Text Processing → Chunk documents
    3. Embedding Generation → Generate vector embeddings
    4. Index Storage → Store in vector DB + BM25
    5. Hybrid Retrieval → Semantic + Keyword search
    6. Reranking → Cross-encoder refinement
    7. Generation → LLM with context
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        # Components (initialized lazily)
        self._embedding_model = None
        self._vector_store = None
        self._bm25_index = None
        self._hybrid_retriever = None
        self._reranker = None
        self._llm_client = None
        self._generator = None
        self._metadata = None

        # State
        self._indexed = False
        self._documents = []
        self._chunks = []

    # ==================== Initialization ====================

    def _init_embedding_model(self):
        """Initialize embedding model."""

        if self._embedding_model is None:
            try:
                from ..processing.embedding_pipeline import EmbeddingGenerator

                self._embedding_model = EmbeddingGenerator(
                    model_name=self.config.embedding_model,
                    batch_size=32,
                    max_length=512,
                    normalize=True,
                    cache_dir=self.config.embedding_cache_dir,
                    device=self.config.embedding_device,
                )
                logger.info(
                    f"Initialized embedding model: {self.config.embedding_model}"
                )
            except ImportError as e:
                logger.warning(f"Could not import embedding model: {e}")
                self._embedding_model = None
            except Exception as e:
                logger.warning(f"Could not initialize embedding model: {e}")
                self._embedding_model = None

    def _init_vector_store(self):
        """Initialize vector database."""

        if self._vector_store is None:
            from ..retrieval.vector_store import VectorStore, VectorStoreConfig

            config = VectorStoreConfig(
                store_type=self.config.vector_db_type,
                collection_name="arabic_islamic_literature",
                vector_size=self.config.vector_size,
            )

            self._vector_store = VectorStore(config)
            logger.info(f"Initialized vector store: {self.config.vector_db_type}")

    def _init_bm25(self):
        """Initialize BM25 index."""

        if self._bm25_index is None:
            from ..retrieval.hybrid_retriever import BM25Index

            self._bm25_index = BM25Index(
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )
            logger.info("Initialized BM25 index")

    def _init_hybrid_retriever(self):
        """Initialize hybrid retriever."""

        if self._hybrid_retriever is None:
            from ..retrieval.hybrid_retriever import HybridRetriever

            self._hybrid_retriever = HybridRetriever(
                vector_store=self._vector_store,
                embedding_model=self._embedding_model,
                bm25=self._bm25_index,
                weights=self.config.hybrid_weights,
            )
            logger.info("Initialized hybrid retriever")

    def _init_reranker(self):
        """Initialize reranker."""

        if self._reranker is None:
            # Skip if sentence-transformers not available
            try:
                from ..retrieval.hybrid_retriever import Reranker

                self._reranker = Reranker(
                    model_name="BAAI/bge-reranker-base",
                    device=self.config.embedding_device,
                )
                logger.info("Initialized reranker")
            except Exception as e:
                logger.warning(f"Could not initialize reranker: {e}")
                self._reranker = None

    def _init_llm(self):
        """Initialize LLM client."""

        if self._llm_client is None:
            from ..generation.generator import LLMClient, LLMProvider

            provider_map = {
                "openai": LLMProvider.OPENAI,
                "anthropic": LLMProvider.ANTHROPIC,
                "ollama": LLMProvider.OLLAMA,
                "mock": LLMProvider.MOCK,
            }

            self._llm_client = LLMClient(
                provider=provider_map.get(self.config.llm_provider, LLMProvider.MOCK),
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            logger.info(
                f"Initialized LLM: {self.config.llm_provider}/{self.config.llm_model}"
            )

    def _init_generator(self):
        """Initialize RAG generator."""

        if self._generator is None:
            from ..generation.generator import RAGGenerator

            self._generator = RAGGenerator(
                llm_client=self._llm_client,
                include_citations=self.config.include_citations,
                max_context_chunks=self.config.max_context_chunks,
            )
            logger.info("Initialized RAG generator")

    def _ensure_initialized(self):
        """Ensure all components are initialized."""

        self._init_embedding_model()
        self._init_vector_store()
        self._init_bm25()
        self._init_hybrid_retriever()
        self._init_reranker()
        self._init_llm()
        self._init_generator()

    # ==================== Indexing ====================

    async def index_documents(
        self,
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
    ):
        """
        Index documents from the dataset.

        Args:
            limit: Maximum number of books to index (None for all)
            categories: Filter by categories (None for all)
            progress_callback: Optional callback for progress
        """

        logger.info("Starting document indexing...")
        start_time = time.time()

        # Load metadata
        await self._load_metadata()

        # Get list of files to process
        files_to_process = self._get_files_to_process(categories, limit)

        logger.info(f"Processing {len(files_to_process)} files...")

        # Process files
        all_chunks = []
        processed = 0

        for file_info in files_to_process:
            file_path = file_info["path"]
            book_id = file_info["book_id"]
            book_title = file_info["title"]
            author = file_info["author"]
            category = file_info["category"]

            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Chunk the content
                chunks = self._chunk_document(
                    content=content,
                    book_id=book_id,
                    book_title=book_title,
                    author=author,
                    category=category,
                )

                all_chunks.extend(chunks)
                processed += 1

                if progress_callback:
                    progress_callback(processed, len(files_to_process))

                # Log progress
                if processed % 50 == 0:
                    logger.info(
                        f"Processed {processed}/{len(files_to_process)} books, "
                        f"{len(all_chunks)} chunks"
                    )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        self._chunks = all_chunks
        logger.info(f"Created {len(all_chunks)} chunks from {processed} books")

        # Ensure components are initialized
        self._ensure_initialized()

        # Generate embeddings
        await self._generate_embeddings(all_chunks)

        # Build BM25 index
        self._build_bm25_index(all_chunks)

        # Save indexes
        self._save_indexes()

        self._indexed = True

        elapsed = time.time() - start_time
        logger.info(f"Indexing complete in {elapsed:.2f} seconds")

    async def _load_metadata(self):
        """Load book metadata."""

        from ..data.ingestion_pipeline import MetadataIngestionPipeline

        metadata_path = os.path.join(self.config.datasets_path, "metadata")
        pipeline = MetadataIngestionPipeline(metadata_path)

        stats = await pipeline.load_metadata()
        logger.info(f"Loaded metadata: {stats}")

        self._metadata = pipeline

    def _get_files_to_process(
        self,
        categories: Optional[List[str]],
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Get list of files to process."""

        files = []

        extracted_path = os.path.join(self.config.datasets_path, "extracted_books")

        if not os.path.exists(extracted_path):
            logger.warning(f"Extracted books path not found: {extracted_path}")
            return []

        # Get all txt files
        txt_files = list(Path(extracted_path).glob("*.txt"))

        for file_path in txt_files:
            # Extract book ID from filename
            filename = file_path.stem
            parts = filename.split("_", 1)

            if not parts or not parts[0].isdigit():
                continue

            book_id = int(parts[0])

            # Get metadata
            book_meta = self._metadata.books.get(book_id, {})

            # Filter by category
            if categories:
                cat_name = book_meta.get("cat_name", "")
                if cat_name not in categories:
                    continue

            # Get author
            authors = book_meta.get("authors", [])
            author = authors[0].get("name", "Unknown") if authors else "Unknown"

            files.append(
                {
                    "path": str(file_path),
                    "book_id": book_id,
                    "title": book_meta.get("title", filename),
                    "author": author,
                    "category": book_meta.get("cat_name", "Unknown"),
                }
            )

        # Apply limit
        if limit:
            files = files[:limit]

        return files

    def _chunk_document(
        self,
        content: str,
        book_id: int,
        book_title: str,
        author: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """Chunk a document into smaller pieces."""

        from ..processing.arabic_processor import ArabicChunker

        chunker = ArabicChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            preserve_sentences=True,
        )

        chunks = chunker.chunk_by_chars(
            text=content,
            book_id=book_id,
            book_title=book_title,
            author=author,
            category=category,
        )

        return chunks

    async def _generate_embeddings(self, chunks: List[Dict[str, Any]]):
        """Generate embeddings for all chunks."""

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        if not chunks:
            return

        # Ensure embedding model is initialized
        self._init_embedding_model()

        # Extract texts
        texts = [chunk["content"] for chunk in chunks]

        # Generate embeddings in batches
        embeddings = self._embedding_model.embed_texts(
            texts,
            show_progress=True,
        )

        # Add embeddings to chunks
        import numpy as np

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        logger.info("Embeddings generated")

        # Store in vector database
        logger.info("Storing vectors in database...")

        ids = [chunk["chunk_id"] for chunk in chunks]
        vectors = [np.array(chunk["embedding"]) for chunk in chunks]

        # Prepare payloads
        payloads = [
            {
                "content": chunk["content"],
                "book_id": chunk["book_id"],
                "book_title": chunk["book_title"],
                "author": chunk["author"],
                "category": chunk["category"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ]

        self._vector_store.add_vectors(ids, vectors, payloads)

        logger.info("Vectors stored in database")

    def _build_bm25_index(self, chunks: List[Dict[str, Any]]):
        """Build BM25 index."""

        logger.info("Building BM25 index...")

        # Prepare documents for BM25
        documents = [
            {
                "id": chunk["chunk_id"],
                "content": chunk["content"],
                "book_id": chunk["book_id"],
                "book_title": chunk["book_title"],
                "author": chunk["author"],
                "category": chunk["category"],
            }
            for chunk in chunks
        ]

        self._bm25_index.index(documents)

        logger.info("BM25 index built")

    def _save_indexes(self):
        """Save indexes to disk."""

        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save BM25 index
        bm25_path = output_dir / "bm25_index.pkl"
        self._bm25_index.save(str(bm25_path))

        # Save chunks with embeddings
        chunks_path = output_dir / "chunks.pkl"
        import pickle

        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

        # Save vector store
        if hasattr(self._vector_store, "save"):
            self._vector_store.save(str(output_dir / "vector_store"))

        logger.info(f"Indexes saved to {output_dir}")

    def load_indexes(self):
        """Load existing indexes from disk."""

        output_dir = Path(self.config.output_path)
        bm25_path = output_dir / "bm25_index.pkl"

        if bm25_path.exists():
            self._init_bm25()
            self._bm25_index.load(str(bm25_path))
            logger.info("Loaded BM25 index")

            # Load chunks
            chunks_path = output_dir / "chunks.pkl"
            if chunks_path.exists():
                import pickle

                with open(chunks_path, "rb") as f:
                    self._chunks = pickle.load(f)
                logger.info(f"Loaded {len(self._chunks)} chunks")

            # Load vector store
            vector_path = output_dir / "vector_store"
            if vector_path.exists() and hasattr(self._vector_store, "load"):
                self._vector_store.load(str(vector_path))
                logger.info("Loaded vector store")

            # Initialize other components
            self._ensure_initialized()
            self._indexed = True

    # ==================== Querying ====================

    async def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Query the RAG pipeline.

        Args:
            question: User question
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            QueryResult with answer and sources
        """

        start_time = time.time()

        # Ensure initialized
        self._ensure_initialized()

        if not self._indexed:
            # Try to load existing indexes
            self.load_indexes()

            if not self._indexed:
                raise RuntimeError("No index available. Please index documents first.")

        # Step 1: Hybrid retrieval
        retrieval_results = await self._hybrid_retriever.search(
            query=question,
            top_k=self.config.retrieval_top_k,
            filters=filters,
        )

        # Step 2: Reranking
        if self._reranker is not None:
            reranked_results = self._reranker.rerank(
                query=question,
                candidates=retrieval_results,
                top_k=top_k,
            )
        else:
            # Skip reranking, use retrieval results directly
            logger.warning("Reranker not available, using retrieval results directly")
            reranked_results = retrieval_results[:top_k]

        # Step 3: Convert to chunks format
        chunks = [
            {
                "content": result.content,
                "metadata": result.metadata,
            }
            for result in reranked_results
        ]

        # Step 4: Generate answer
        generation_result = await self._generator.generate(
            query=question,
            retrieved_chunks=chunks,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Build sources list
        sources = [
            {
                "book_title": r.metadata.get("book_title", "Unknown"),
                "author": r.metadata.get("author", "Unknown"),
                "category": r.metadata.get("category", "Unknown"),
                "score": r.score,
                "content_preview": r.content[:200] + "...",
            }
            for r in reranked_results
        ]

        return QueryResult(
            query=question,
            answer=generation_result.answer,
            sources=sources,
            latency_ms=latency_ms,
            tokens_used=generation_result.tokens_used,
            model=generation_result.model,
        )

    async def query_stream(
        self,
        question: str,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """Query with streaming response."""

        # Ensure initialized
        self._ensure_initialized()

        # Get retrieval results
        retrieval_results = await self._hybrid_retriever.search(
            query=question,
            top_k=self.config.retrieval_top_k,
        )

        # Rerank
        reranked_results = self._reranker.rerank(
            query=question,
            candidates=retrieval_results,
            top_k=top_k,
        )

        # Build context
        chunks = [
            {
                "content": result.content,
                "metadata": result.metadata,
            }
            for result in reranked_results
        ]

        # Generate with streaming
        from ..generation.generator import ArabicPrompts

        context = ArabicPrompts.build_context(chunks, self.config.max_context_chunks)
        user_prompt = ArabicPrompts.build_user_prompt(question, context)

        system_prompt = ArabicPrompts.SYSTEM_PROMPT_ENGLISH
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        async for token in self._llm_client.generate_stream(full_prompt):
            yield token

    # ==================== Utilities ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""

        stats = {
            "indexed": self._indexed,
            "total_chunks": len(self._chunks),
            "config": {
                "embedding_model": self.config.embedding_model,
                "llm_model": self.config.llm_model,
                "chunk_size": self.config.chunk_size,
            },
        }

        if self._bm25_index:
            stats["bm25_documents"] = self._bm25_index.doc_count
            stats["bm25_vocabulary"] = len(self._bm25_index.term_idf)

        if self._vector_store:
            stats["vector_count"] = self._vector_store.count()

        return stats

    def get_categories(self) -> List[str]:
        """Get list of available categories."""

        if self._metadata:
            return list(
                set(
                    book.get("cat_name", "Unknown")
                    for book in self._metadata.books.values()
                )
            )
        return []


# ==================== Factory Function ====================


def create_rag_pipeline(
    config: Optional[RAGConfig] = None,
    **kwargs,
) -> CompleteRAGPipeline:
    """
    Factory function to create a RAG pipeline.

    Args:
        config: Optional configuration
        **kwargs: Override config values

    Returns:
        CompleteRAGPipeline instance
    """

    if config is None:
        config = RAGConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return CompleteRAGPipeline(config)
