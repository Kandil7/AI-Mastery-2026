from __future__ import annotations

from typing import Any, Dict

from agents.tools import Tool
from core.factories import create_embeddings_provider, create_llm_provider, create_vector_store
from core.settings import load_settings
from rag.answer import generate_answer
from rag.embeddings import EmbeddingService
from rag.reranker import Reranker
from rag.retriever import HybridRetriever


def build_rag_tool() -> Tool:
    def handler(question: str) -> str:
        settings = load_settings()
        store = create_vector_store(settings)
        embedder = EmbeddingService(create_embeddings_provider(settings))
        retriever = HybridRetriever(store, embedder)
        chunks = retriever.retrieve(query=question, top_k=8, filters={})
        chunks = Reranker().rerank(question, chunks)
        provider = create_llm_provider(settings)
        return generate_answer(question, chunks, provider)

    return Tool(name="rag", handler=handler)


def build_sql_tool(dsn: str, query_template: str | None = None) -> Tool:
    def handler(question: str) -> str:
        import sqlalchemy as sa

        engine = sa.create_engine(dsn)
        query = query_template or question
        with engine.connect() as conn:
            result = conn.execute(sa.text(query))
            rows = [dict(row) for row in result.mappings()]
        return str(rows)

    return Tool(name="sql", handler=handler)


def build_web_tool(base_url: str, headers: Dict[str, Any] | None = None) -> Tool:
    def handler(question: str) -> str:
        import requests

        response = requests.get(base_url, params={"q": question}, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text

    return Tool(name="web", handler=handler)
