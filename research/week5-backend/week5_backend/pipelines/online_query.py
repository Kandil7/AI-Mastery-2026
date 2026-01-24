from __future__ import annotations

import uuid
from typing import Any, Dict

from agents.executor import AgentExecutor
from agents.planner import Planner
from agents.tooling import build_rag_tool, build_sql_tool, build_web_tool
from agents.tools import ToolRegistry
from agents.verifier import Verifier
from core.factories import (
    create_bm25_index,
    create_embeddings_provider,
    create_llm_provider,
    create_routing_policy,
    create_vector_store,
)
from core.settings import load_settings
from rag.answer import generate_answer, generate_answer_strict
from rag.citations import format_citations
from rag.embeddings import EmbeddingService
from rag.reranker import Reranker
from rag.retriever import FusionConfig, HybridRetriever


def _build_tools(settings) -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(build_rag_tool())

    tool_cfg = settings.raw.get("tools") or {}
    sql_cfg = tool_cfg.get("sql") or {}
    if sql_cfg.get("dsn"):
        tools.register(build_sql_tool(sql_cfg["dsn"], sql_cfg.get("query_template")))

    web_cfg = tool_cfg.get("web") or {}
    if web_cfg.get("base_url"):
        tools.register(build_web_tool(web_cfg["base_url"], web_cfg.get("headers")))

    return tools


def run_query_pipeline(
    tenant_id: str,
    question: str,
    filters: Dict[str, Any],
    top_k: int,
    mode: str,
) -> Dict[str, Any]:
    _ = (top_k,)
    trace_id = str(uuid.uuid4())

    settings = load_settings()
    filters = dict(filters or {})
    if tenant_id:
        filters.setdefault("tenant_id", tenant_id)

    if mode == "agentic":
        tools = _build_tools(settings)
        executor = AgentExecutor(Planner(), tools)
        result = executor.run(question)

        provider_name = create_routing_policy(settings).choose(task="agentic")
        provider = create_llm_provider(settings, provider_name)
        synthesis_prompt = (
            "Synthesize a final answer from the tool outputs.\n"
            f"Question: {question}\n"
            f"Tool outputs:\n{result.output}\n"
            "Answer:"
        )
        answer = provider.generate(synthesis_prompt)
        citations = result.citations
        model = provider_name

        verifier = Verifier(provider)
        if not verifier.verify(answer, citations):
            rag_tool = build_rag_tool()
            fallback = rag_tool.handler(question)
            answer = fallback.output
            citations = fallback.citations
    else:
        retrieval_cfg = settings.raw.get("retrieval") or {}
        fusion_cfg = retrieval_cfg.get("fusion") or {}
        fusion = FusionConfig(
            use_rrf=bool(fusion_cfg.get("use_rrf", True)),
            rrf_k=int(fusion_cfg.get("rrf_k", 60)),
            vector_weight=float(fusion_cfg.get("vector_weight", 1.0)),
            bm25_weight=float(fusion_cfg.get("bm25_weight", 1.0)),
        )
        retrieval_top_k = int(retrieval_cfg.get("top_k", top_k))
        if retrieval_top_k < top_k:
            retrieval_top_k = top_k
        vector_store = create_vector_store(settings)
        embedder = EmbeddingService(create_embeddings_provider(settings))
        bm25 = create_bm25_index(settings) if mode == "hybrid" else None
        retriever = HybridRetriever(vector_store, embedder, bm25_index=bm25, fusion=fusion)
        chunks = retriever.retrieve(query=question, top_k=retrieval_top_k, filters=filters)
        provider_name = create_routing_policy(settings).choose(task="qa")
        provider = create_llm_provider(settings, provider_name)
        reranker_cfg = settings.raw.get("reranker") or {}
        reranker = Reranker(provider if reranker_cfg.get("enabled", True) else None)
        rerank_top_k = int(reranker_cfg.get("top_k", top_k))
        chunks = reranker.rerank(question, chunks, top_k=rerank_top_k)
        answer_cfg = settings.raw.get("answer") or {}
        max_context_words = (
            int(answer_cfg["max_context_words"]) if "max_context_words" in answer_cfg else None
        )
        answer = generate_answer(question, chunks, provider, max_context_words=max_context_words)
        citations = format_citations(chunks)
        model = provider_name
        verification_cfg = settings.raw.get("verification") or {}
        if verification_cfg.get("enabled", True):
            verifier = Verifier(provider)
            if not verifier.verify(answer, citations):
                answer = generate_answer_strict(
                    question,
                    chunks,
                    provider,
                    max_context_words=max_context_words,
                )

    return {"answer": answer, "citations": citations, "trace_id": trace_id, "model": model}
