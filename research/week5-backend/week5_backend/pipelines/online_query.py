from __future__ import annotations

import uuid
from typing import Any, Dict

from agents.executor import AgentExecutor
from agents.planner import Planner
from agents.tools import Tool, ToolRegistry
from core.factories import (
    create_embeddings_provider,
    create_llm_provider,
    create_routing_policy,
    create_vector_store,
)
from core.settings import load_settings
from rag.answer import generate_answer
from rag.citations import format_citations
from rag.embeddings import EmbeddingService
from rag.reranker import Reranker
from rag.retriever import HybridRetriever


def _rag_tool(question: str) -> str:
    settings = load_settings()
    vector_store = create_vector_store(settings)
    embedder = EmbeddingService(create_embeddings_provider(settings))
    retriever = HybridRetriever(vector_store, embedder)
    chunks = retriever.retrieve(query=question, top_k=8, filters={})
    reranker = Reranker()
    chunks = reranker.rerank(question, chunks)
    provider_name = create_routing_policy(settings).choose(task="qa")
    provider = create_llm_provider(settings, provider_name)
    return generate_answer(question, chunks, provider)


def run_query_pipeline(
    tenant_id: str,
    question: str,
    filters: Dict[str, Any],
    top_k: int,
    mode: str,
) -> Dict[str, Any]:
    _ = (tenant_id, filters, top_k)
    trace_id = str(uuid.uuid4())

    settings = load_settings()

    if mode == "agentic":
        tools = ToolRegistry()
        tools.register(Tool(name="rag", handler=_rag_tool))
        executor = AgentExecutor(Planner(), tools)
        answer = executor.run(question)
        citations = []
        model = "agentic"
    else:
        vector_store = create_vector_store(settings)
        embedder = EmbeddingService(create_embeddings_provider(settings))
        retriever = HybridRetriever(vector_store, embedder)
        chunks = retriever.retrieve(query=question, top_k=top_k, filters=filters)
        reranker = Reranker()
        chunks = reranker.rerank(question, chunks)
        provider_name = create_routing_policy(settings).choose(task="qa")
        provider = create_llm_provider(settings, provider_name)
        answer = generate_answer(question, chunks, provider)
        citations = format_citations(chunks)
        model = provider_name

    return {"answer": answer, "citations": citations, "trace_id": trace_id, "model": model}
