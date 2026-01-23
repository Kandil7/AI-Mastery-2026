"""
Production-Grade RAG Pipeline for Advanced Information Retrieval and Generation

This module implements a comprehensive RAG (Retrieval-Augmented Generation) pipeline
that combines hybrid retrieval with language model generation to produce accurate,
well-sourced responses to user queries. The pipeline is designed for production
environments with focus on reliability, performance, and traceability.

The architecture follows the 2026 RAG production standards with emphasis on:
- Hybrid retrieval combining dense and sparse search
- Traceable answers with explicit citations
- Configurable generation parameters
- Structured response format for UI and evaluation

Key Components:
- HybridRetriever: Combines semantic and keyword-based search
- Generator: Language model for response generation
- RAGConfig: Centralized configuration management
- Structured output: Includes sources and metadata for transparency

Example:
    >>> from src.pipeline import RAGPipeline, RAGConfig
    >>> from src.retrieval import Document
    >>>
    >>> # Initialize pipeline with custom configuration
    >>> config = RAGConfig(
    ...     top_k=5,
    ...     max_new_tokens=300,
    ...     generator_model="gpt2",
    ...     dense_model="all-MiniLM-L6-v2",
    ...     alpha=0.7,
    ...     fusion="rrf"
    ... )
    >>> pipeline = RAGPipeline(config)
    >>>
    >>> # Index documents
    >>> docs = [Document("1", "AI is transforming industries")]
    >>> pipeline.index(docs)
    >>>
    >>> # Query the pipeline
    >>> result = pipeline.query("What is AI doing?")
    >>> print(result["response"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from transformers import pipeline

from src.retrieval import Document, HybridRetriever, RetrievalResult


@dataclass
class RAGConfig:
    """
    Configuration class for RAG pipeline parameters.

    This class centralizes all configurable parameters for the RAG pipeline,
    allowing for easy experimentation and tuning of different components.
    The configuration follows production best practices with sensible defaults
    that can be overridden as needed.

    Attributes:
        top_k (int): Number of top documents to retrieve for each query
        max_new_tokens (int): Maximum number of tokens to generate in response
        generator_model (str): Name of the language model to use for generation
        dense_model (str): Name of the sentence transformer model for dense retrieval
        alpha (float): Weight for dense retrieval in hybrid fusion (sparse weight = 1 - alpha)
        fusion (str): Fusion strategy for hybrid retrieval ('rrf' or 'weighted')

    Example:
        >>> config = RAGConfig(
        ...     top_k=5,
        ...     max_new_tokens=300,
        ...     generator_model="gpt2-medium",
        ...     dense_model="all-mpnet-base-v2",
        ...     alpha=0.6,
        ...     fusion="rrf"
        ... )
    """
    top_k: int = 3
    max_new_tokens: int = 200
    generator_model: str = "gpt2"
    dense_model: str = "all-MiniLM-L6-v2"
    alpha: float = 0.5
    fusion: str = "rrf"


class RAGPipeline:
    """
    Main RAG pipeline orchestrating retrieval and generation processes.

    This class serves as the central orchestrator for the RAG system, managing
    the complete flow from query input to response generation with cited sources.
    It integrates the hybrid retrieval system with a language model generator
    to produce accurate, well-sourced answers.

    The pipeline is designed for production use with:
    - Configurable components and parameters
    - Structured output format for UI and evaluation
    - Error handling and fallback mechanisms
    - Performance optimization considerations

    Args:
        config (RAGConfig, optional): Configuration object with pipeline parameters

    Example:
        >>> from src.pipeline import RAGPipeline, RAGConfig
        >>> from src.retrieval import Document
        >>>
        >>> # Create pipeline with default configuration
        >>> pipeline = RAGPipeline()
        >>>
        >>> # Add documents to knowledge base
        >>> docs = [Document("doc1", "Artificial Intelligence is transformative")]
        >>> pipeline.index(docs)
        >>>
        >>> # Query the system
        >>> result = pipeline.query("What is AI?")
        >>> print(result["response"])
        >>> print(len(result["retrieved_documents"]))
    """

    def __init__(self, config: RAGConfig | None = None):
        """
        Initialize the RAG pipeline with specified configuration.

        Sets up the hybrid retriever and language model generator based on
        the provided configuration. Creates the necessary components for
        the complete RAG workflow.

        Args:
            config (RAGConfig, optional): Configuration object. Uses defaults if None.
        """
        self.config = config or RAGConfig()
        self.retriever = HybridRetriever(
            alpha=self.config.alpha,
            fusion=self.config.fusion,
            dense_model=self.config.dense_model,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.config.generator_model,
            tokenizer=self.config.generator_model,
        )

    def index(self, documents: List[Document]) -> None:
        """
        Index documents in the retrieval system.

        Adds the provided documents to both dense and sparse retrieval indexes
        making them available for future queries. This method is typically called
        during the ingestion phase of the RAG pipeline.

        Args:
            documents (List[Document]): List of documents to index

        Note:
            Documents are added incrementally to existing index. No deduplication
            is performed at this level.
        """
        self.retriever.index(documents)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for the given query.

        Performs hybrid retrieval using the configured retrieval system to find
        the most relevant documents for the provided query. Uses the pipeline's
        default top_k unless overridden.

        Args:
            query (str): The search query
            top_k (int, optional): Number of top results to return. Uses config default if None.

        Returns:
            List[RetrievalResult]: Ranked list of relevant documents with scores and ranks
        """
        return self.retriever.retrieve(query, top_k=top_k or self.config.top_k)

    def generate(self, query: str, contexts: List[RetrievalResult]) -> str:
        """
        Generate a response based on the query and retrieved contexts.

        Constructs a prompt using the query and retrieved contexts, then uses
        the language model to generate a coherent response. Handles the case
        where no contexts are available by generating a response indicating
        insufficient information.

        Args:
            query (str): Original user query
            contexts (List[RetrievalResult]): Retrieved documents to use as context

        Returns:
            str: Generated response to the query
        """
        if not contexts:
            prompt = f"Question: {query}\n\nI do not have enough information."
            generated = self.generator(prompt, max_new_tokens=self.config.max_new_tokens)
            return generated[0]["generated_text"].strip()

        context_text = "\n\n".join(
            f"Document {result.rank}: {result.document.content}" for result in contexts
        )
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        generated = self.generator(prompt, max_new_tokens=self.config.max_new_tokens)
        full_text = generated[0]["generated_text"]
        answer_start = full_text.find("Answer:") + len("Answer:")
        return full_text[answer_start:].strip()

    def query(self, query: str, top_k: int | None = None) -> Dict[str, object]:
        """
        Execute a complete RAG query from retrieval to generation.

        This is the main entry point for the RAG pipeline, performing the
        complete workflow:
        1. Retrieve relevant documents using the hybrid retriever
        2. Generate a response using the language model with retrieved context
        3. Return a structured response with the answer and source information

        Args:
            query (str): User query to process
            top_k (int, optional): Number of documents to retrieve. Uses config default if None.

        Returns:
            Dict[str, object]: Structured response containing:
                - query: Original query
                - response: Generated answer
                - retrieved_documents: List of source documents with metadata
        """
        contexts = self.retrieve(query, top_k=top_k)
        answer = self.generate(query, contexts)
        return {
            "query": query,
            "response": answer,
            "retrieved_documents": [
                {
                    "id": result.document.id,
                    "content": result.document.content,
                    "score": result.score,
                    "rank": result.rank,
                    "metadata": result.document.metadata,
                }
                for result in contexts
            ],
        }
