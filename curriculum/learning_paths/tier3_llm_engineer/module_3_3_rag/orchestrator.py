"""
RAG Orchestrator Module

Production-ready RAG orchestration with:
- LangChain integration
- LlamaIndex integration
- Custom orchestration
- Streaming support
- Citation tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    # Retrieval settings
    top_k: int = 5
    score_threshold: float = 0.0
    rerank: bool = False
    rerank_top_k: int = 10

    # Generation settings
    temperature: float = 0.1
    max_tokens: int = 2048
    stream: bool = False

    # Context settings
    context_window: int = 4096
    context_compression: bool = False
    max_context_items: int = 10

    # Citation settings
    include_citations: bool = True
    citation_format: str = "numbered"  # numbered, inline, footnote

    # Advanced settings
    query_rewrite: bool = False
    multi_query: bool = False
    hyde: bool = False
    parent_chunk_retrieval: bool = False

    # Metadata
    metadata_filter: Optional[Dict[str, Any]] = None
    collection_name: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "rerank": self.rerank,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "context_window": self.context_window,
            "include_citations": self.include_citations,
        }


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    citation_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "citation_id": self.citation_id,
        }


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    answer: str
    sources: List[RetrievedDocument] = field(default_factory=list)
    query: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    model: str = ""
    citations: Optional[Dict[int, str]] = None
    reasoning: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "query": self.query,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "model": self.model,
            "citations": self.citations,
            "confidence": self.confidence,
        }

    def format_with_citations(self) -> str:
        """Format answer with citations."""
        if not self.citations or not self.include_citations:
            return self.answer

        answer = self.answer
        for cid, source in self.citations.items():
            answer = answer.replace(f"[{cid}]", f" [{cid}]")

        if self.citations:
            answer += "\n\nSources:\n"
            for cid, source in self.citations.items():
                answer += f"[{cid}] {source}\n"

        return answer


@dataclass
class RAGStep:
    """A step in the RAG pipeline for tracing."""

    name: str
    input: Any
    output: Any
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGOrchestrator(ABC):
    """Abstract base class for RAG orchestrators."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.config = config or RAGConfig()
        self._trace: List[RAGStep] = []
        self._stats: Dict[str, Any] = {
            "total_queries": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
        }

    @abstractmethod
    async def query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> RAGResponse:
        """Execute RAG query."""
        pass

    @abstractmethod
    async def query_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response."""
        pass

    def _add_trace(self, step: RAGStep) -> None:
        """Add step to trace."""
        self._trace.append(step)

    def get_trace(self) -> List[RAGStep]:
        """Get execution trace."""
        return self._trace.copy()

    def clear_trace(self) -> None:
        """Clear execution trace."""
        self._trace.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return self._stats.copy()

    def _update_stats(self, latency_ms: float) -> None:
        """Update statistics."""
        self._stats["total_queries"] += 1
        self._stats["total_latency_ms"] += latency_ms
        self._stats["avg_latency_ms"] = (
            self._stats["total_latency_ms"] / self._stats["total_queries"]
        )


class LangChainOrchestrator(RAGOrchestrator):
    """
    RAG orchestrator using LangChain.

    Features:
    - LangChain retriever integration
    - Chain composition
    - Memory integration
    - Callback support
    """

    def __init__(
        self,
        llm: Any = None,
        retriever: Any = None,
        config: Optional[RAGConfig] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(config)

        self.llm = llm
        self.retriever = retriever
        self.verbose = verbose

        self._chain = None
        self._memory = None

        # Try to import LangChain
        self._langchain = None
        try:
            from langchain_core.documents import Document as LCDocument
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            self._langchain = {
                "Document": LCDocument,
                "ChatPromptTemplate": ChatPromptTemplate,
                "RunnablePassthrough": RunnablePassthrough,
            }
        except ImportError:
            logger.warning("LangChain not installed. Some features unavailable.")

    def _create_chain(self) -> None:
        """Create LangChain RAG chain."""
        if not self._langchain:
            return

        # RAG prompt template
        template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""

        prompt = self._langchain["ChatPromptTemplate"].from_template(template)

        # Create chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self._chain = (
            {"context": self.retriever | format_docs, "question": self._langchain["RunnablePassthrough"]}
            | prompt
            | self.llm
        )

    async def query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> RAGResponse:
        """Execute RAG query using LangChain."""
        start_time = time.time()

        if not self._chain:
            self._create_chain()

        # Retrieve documents
        retrieve_start = time.time()
        docs = await self._async_retrieve(query)
        retrieve_latency = (time.time() - retrieve_start) * 1000

        self._add_trace(RAGStep(
            name="retrieval",
            input=query,
            output=[d.content if hasattr(d, 'content') else d.page_content for d in docs],
            latency_ms=retrieve_latency,
        ))

        # Format context
        context = "\n\n".join(
            d.content if hasattr(d, 'content') else d.page_content
            for d in docs
        )

        # Generate answer
        generate_start = time.time()

        if self._chain:
            response = await self._chain.ainvoke({"question": query})
            answer = response.content if hasattr(response, 'content') else str(response)
        else:
            # Fallback without LangChain
            answer = await self._generate_answer(query, context, conversation_history)

        generate_latency = (time.time() - generate_start) * 1000

        self._add_trace(RAGStep(
            name="generation",
            input={"query": query, "context": context[:500]},
            output=answer[:500],
            latency_ms=generate_latency,
        ))

        total_latency = (time.time() - start_time) * 1000
        self._update_stats(total_latency)

        # Build response
        sources = [
            RetrievedDocument(
                content=d.content if hasattr(d, 'content') else d.page_content,
                score=getattr(d, 'metadata', {}).get('score', 1.0),
                metadata=getattr(d, 'metadata', {}),
                citation_id=i + 1,
            )
            for i, d in enumerate(docs)
        ]

        citations = {s.citation_id: s.metadata.get('source', 'Unknown') for s in sources} if self.config.include_citations else None

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            latency_ms=total_latency,
            citations=citations,
        )

    async def _async_retrieve(self, query: str) -> List[Any]:
        """Async document retrieval."""
        if hasattr(self.retriever, 'aget_relevant_documents'):
            return await self.retriever.aget_relevant_documents(query)
        elif hasattr(self.retriever, 'invoke'):
            return await self.retriever.ainvoke(query)
        else:
            # Sync fallback
            return self.retriever.get_relevant_documents(query)

    async def _generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Generate answer without LangChain."""
        messages = [{"role": "system", "content": f"Answer based on this context:\n{context}"}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": query})

        if self.llm and hasattr(self.llm, 'generate'):
            response = await self.llm.generate(messages)
            return response.content if hasattr(response, 'content') else str(response)

        return f"Context: {context}\n\nQuery: {query}"

    async def query_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response."""
        # Retrieve first
        docs = await self._async_retrieve(query)
        context = "\n\n".join(
            d.content if hasattr(d, 'content') else d.page_content
            for d in docs
        )

        # Stream generation
        messages = [
            {"role": "system", "content": f"Answer based on this context:\n{context}"},
            {"role": "user", "content": query},
        ]

        if self.llm and hasattr(self.llm, 'stream'):
            async for chunk in self.llm.stream(messages):
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)
        else:
            # Non-streaming fallback
            answer = await self._generate_answer(query, context, conversation_history)
            yield answer


class LlamaIndexOrchestrator(RAGOrchestrator):
    """
    RAG orchestrator using LlamaIndex.

    Features:
    - LlamaIndex query engine
    - Advanced retrieval strategies
    - Response synthesis
    """

    def __init__(
        self,
        index: Any = None,
        llm: Any = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        super().__init__(config)

        self.index = index
        self.llm = llm
        self._query_engine = None

        # Try to import LlamaIndex
        self._llama_index = None
        try:
            import llama_index
            self._llama_index = llama_index
        except ImportError:
            logger.warning("LlamaIndex not installed. Some features unavailable.")

    def _create_query_engine(self) -> None:
        """Create LlamaIndex query engine."""
        if not self._llama_index or not self.index:
            return

        # Configure retriever
        retriever = self.index.as_retriever(
            similarity_top_k=self.config.top_k,
            score_threshold=self.config.score_threshold,
        )

        # Create query engine
        self._query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=self.config.top_k,
            response_mode="compact",
        )

    async def query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> RAGResponse:
        """Execute RAG query using LlamaIndex."""
        start_time = time.time()

        if not self._query_engine:
            self._create_query_engine()

        # Query
        retrieve_start = time.time()

        if self._query_engine and hasattr(self._query_engine, 'aquery'):
            response = await self._query_engine.aquery(query)
        elif self.index and hasattr(self.index, 'aquery'):
            response = await self.index.aquery(query)
        else:
            response = await self._fallback_query(query)

        retrieve_latency = (time.time() - retrieve_start) * 1000

        # Extract sources
        sources = []
        if hasattr(response, 'source_nodes'):
            for i, node in enumerate(response.source_nodes):
                sources.append(RetrievedDocument(
                    content=node.node.text if hasattr(node.node, 'text') else str(node.node),
                    score=node.score if hasattr(node, 'score') else 1.0,
                    metadata=node.node.metadata if hasattr(node.node, 'metadata') else {},
                    citation_id=i + 1,
                ))

        total_latency = (time.time() - start_time) * 1000
        self._update_stats(total_latency)

        citations = {s.citation_id: s.metadata.get('file_name', 'Unknown') for s in sources} if self.config.include_citations else None

        return RAGResponse(
            answer=str(response),
            sources=sources,
            query=query,
            latency_ms=total_latency,
            citations=citations,
        )

    async def _fallback_query(self, query: str) -> Any:
        """Fallback query without LlamaIndex."""
        # Simple retrieval and generation
        return type('Response', (), {"response": f"Query: {query}", "source_nodes": []})()

    async def query_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response using LlamaIndex."""
        if self._query_engine and hasattr(self._query_engine, 'streaming_query'):
            response = self._query_engine.streaming_query(query)
            for chunk in response:
                yield str(chunk)
        else:
            # Fallback
            result = await self.query(query, conversation_history)
            yield result.answer


class CustomRAGOrchestrator(RAGOrchestrator):
    """
    Custom RAG orchestrator without framework dependencies.

    Features:
    - Full control over pipeline
    - Custom retrieval logic
    - Flexible generation
    """

    def __init__(
        self,
        llm_client: Any,
        vector_store: Any,
        embedding_generator: Any,
        config: Optional[RAGConfig] = None,
    ) -> None:
        super().__init__(config)

        self.llm_client = llm_client
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    async def query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> RAGResponse:
        """Execute RAG query with custom orchestration."""
        start_time = time.time()

        # Step 1: Generate query embedding
        embed_start = time.time()
        query_embedding = await self.embedding_generator.embed_text(query)
        embed_latency = (time.time() - embed_start) * 1000

        self._add_trace(RAGStep(
            name="embedding",
            input=query[:100],
            output=f"embedding[{len(query_embedding.embedding)}]",
            latency_ms=embed_latency,
        ))

        # Step 2: Retrieve documents
        retrieve_start = time.time()
        search_result = await self.vector_store.search(
            collection=self.config.collection_name,
            query_vector=query_embedding.embedding,
            top_k=self.config.top_k,
            filter=self.config.metadata_filter,
        )
        retrieve_latency = (time.time() - retrieve_start) * 1000

        sources = [
            RetrievedDocument(
                content=record.metadata.get('content', ''),
                score=record.score,
                metadata=record.metadata,
                citation_id=i + 1,
            )
            for i, record in enumerate(search_result.records)
        ]

        self._add_trace(RAGStep(
            name="retrieval",
            input=query[:100],
            output=f"{len(sources)} documents",
            latency_ms=retrieve_latency,
        ))

        # Step 3: Build context
        context = self._build_context(sources)

        # Step 4: Generate answer
        generate_start = time.time()
        answer = await self._generate_answer(query, context, conversation_history)
        generate_latency = (time.time() - generate_start) * 1000

        self._add_trace(RAGStep(
            name="generation",
            input={"query": query[:100], "context_len": len(context)},
            output=answer[:500],
            latency_ms=generate_latency,
        ))

        total_latency = (time.time() - start_time) * 1000
        self._update_stats(total_latency)

        citations = {s.citation_id: s.metadata.get('source', 'Unknown') for s in sources} if self.config.include_citations else None

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            latency_ms=total_latency,
            citations=citations,
            confidence=self._estimate_confidence(answer, sources),
        )

    def _build_context(self, sources: List[RetrievedDocument]) -> str:
        """Build context from retrieved documents."""
        if not sources:
            return ""

        context_parts = []
        for i, source in enumerate(sources):
            if self.config.include_citations:
                context_parts.append(f"[{i + 1}] {source.content}")
            else:
                context_parts.append(source.content)

        context = "\n\n".join(context_parts)

        # Truncate if needed
        if len(context) > self.config.context_window:
            context = context[:self.config.context_window]

        return context

    async def _generate_answer(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Generate answer using LLM."""
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources using [1], [2], etc. when referencing information from the context.
If the context doesn't contain enough information, say so honestly."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history[-5:])  # Last 5 messages

        messages.append({"role": "user", "content": user_prompt})

        response = await self.llm_client.generate(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.content if hasattr(response, 'content') else str(response)

    def _estimate_confidence(
        self,
        answer: str,
        sources: List[RetrievedDocument],
    ) -> float:
        """Estimate answer confidence."""
        if not sources:
            return 0.1

        # Simple heuristics
        confidence = 0.5

        # More sources = higher confidence
        confidence += min(len(sources) * 0.1, 0.3)

        # Higher scores = higher confidence
        avg_score = sum(s.score for s in sources) / len(sources)
        confidence += avg_score * 0.2

        # Answer length (reasonable length = higher confidence)
        if 50 < len(answer) < 2000:
            confidence += 0.1

        return min(confidence, 1.0)

    async def query_stream(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response."""
        # Retrieve first
        query_embedding = await self.embedding_generator.embed_text(query)
        search_result = await self.vector_store.search(
            collection=self.config.collection_name,
            query_vector=query_embedding.embedding,
            top_k=self.config.top_k,
        )

        sources = [
            RetrievedDocument(
                content=record.metadata.get('content', ''),
                score=record.score,
                metadata=record.metadata,
            )
            for record in search_result.records
        ]

        context = self._build_context(sources)

        # Stream generation
        system_prompt = """You are a helpful assistant that answers questions based on the provided context."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.llm_client, 'generate'):
            result = await self.llm_client.generate(
                messages=messages,
                stream=True,
            )
            async for chunk in result:
                yield chunk.content if hasattr(chunk, 'content') else str(chunk)


# Factory function
def create_orchestrator(
    orchestrator_type: str = "custom",
    **kwargs: Any,
) -> RAGOrchestrator:
    """
    Create RAG orchestrator.

    Args:
        orchestrator_type: Type of orchestrator (langchain, llamaindex, custom)
        **kwargs: Orchestrator-specific arguments

    Returns:
        Configured orchestrator
    """
    orchestrators = {
        "langchain": LangChainOrchestrator,
        "llamaindex": LlamaIndexOrchestrator,
        "custom": CustomRAGOrchestrator,
    }

    orchestrator_class = orchestrators.get(orchestrator_type.lower())
    if not orchestrator_class:
        raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")

    return orchestrator_class(**kwargs)
