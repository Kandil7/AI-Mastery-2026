"""
LLM-Based Reranker
==================

Uses LLM for intelligent reranking with reasoning.

This approach uses an LLM to evaluate and rank documents
based on their relevance to the query, potentially considering
nuances that cross-encoders might miss.

Benefits:
- Can understand complex relevance criteria
- Can explain ranking decisions
- Can handle domain-specific relevance

Limitations:
- Slower than cross-encoders
- More expensive (API costs)
- May be less consistent

Example:
    >>> from src.llm import LLMClient
    >>>
    >>> llm = LLMClient(model="gpt-4")
    >>> reranker = LLMReranker(llm, top_k=5)
    >>> results = reranker.rerank(query, documents)
"""

import time
import json
from typing import Any, Dict, List, Optional, Callable

from .base import BaseReranker, RerankResult, RerankResults


class LLMReranker(BaseReranker):
    """
    LLM-based reranker with reasoning capabilities.

    Uses an LLM to evaluate document relevance and provide
    ranked results with optional explanations.

    Example:
        >>> reranker = LLMReranker(
        ...     llm_client=llm,
        ...     top_k=5,
        ...     include_explanations=True,
        ... )
        >>> results = reranker.rerank(query, documents)
        >>> for r in results:
        ...     print(f"{r.id}: {r.metadata.get('explanation', '')}")
    """

    DEFAULT_PROMPT = """You are an expert at evaluating document relevance. 
Given a query and a list of documents, rank them by how well they answer the query.

Query: {query}

Documents:
{documents}

Rank the documents from most to least relevant. Return ONLY a JSON array with this format:
[
    {{"id": "doc_id", "score": 0.95, "reason": "Brief explanation"}},
    ...
]

Scores should be between 0 and 1."""

    def __init__(
        self,
        llm_client: Any,
        top_k: int = 5,
        include_explanations: bool = False,
        prompt_template: Optional[str] = None,
        max_documents: int = 10,
    ):
        """
        Initialize LLM reranker.

        Args:
            llm_client: LLM client for generating responses
            top_k: Number of results to return
            include_explanations: Include ranking explanations
            prompt_template: Custom prompt template
            max_documents: Maximum documents to rerank at once
        """
        super().__init__(top_k)
        self.llm_client = llm_client
        self.include_explanations = include_explanations
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.max_documents = max_documents

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> RerankResults:
        """
        Rerank documents using LLM.

        Args:
            query: Query text
            results: List of documents to rerank
            top_k: Number of results to return

        Returns:
            RerankResults with new rankings
        """
        top_k = top_k or self.top_k
        start_time = time.time()

        if not results:
            return RerankResults(results=[], original_count=0)

        # Limit documents for LLM
        limited_results = results[: self.max_documents]

        # Format documents for prompt
        doc_strings = "\n".join(
            [f"ID: {d['id']}\nContent: {d['content'][:500]}..." for d in limited_results]
        )

        # Create prompt
        prompt = self.prompt_template.format(
            query=query,
            documents=doc_strings,
        )

        # Get LLM response
        response = self.llm_client.generate(prompt)

        # Parse response
        try:
            # Try to extract JSON from response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                ranking_data = json.loads(response[json_start:json_end])
            else:
                ranking_data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            # Fallback: use original order with LLM confidence scores
            ranking_data = [
                {"id": d["id"], "score": 1.0 - (i * 0.1), "reason": ""}
                for i, d in enumerate(limited_results)
            ]

        # Build reranked results
        id_to_doc = {d["id"]: d for d in limited_results}
        reranked = []

        for item in ranking_data:
            doc_id = item.get("id")
            if doc_id not in id_to_doc:
                continue

            doc = id_to_doc[doc_id]
            metadata = doc.get("metadata", {}).copy()

            if self.include_explanations:
                metadata["explanation"] = item.get("reason", "")

            reranked.append(
                RerankResult(
                    id=doc_id,
                    content=doc["content"],
                    original_score=doc.get("score", 0.0),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=metadata,
                )
            )

        # Sort by rerank score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        # Take top_k
        top_results = reranked[:top_k]

        rerank_time = (time.time() - start_time) * 1000

        return RerankResults(
            results=top_results,
            rerank_time_ms=rerank_time,
            original_count=len(results),
        )

    def rerank_with_criteria(
        self,
        query: str,
        results: List[Dict[str, Any]],
        criteria: str,
        top_k: Optional[int] = None,
    ) -> RerankResults:
        """
        Rerank with custom criteria.

        Args:
            query: Query text
            results: Documents to rerank
            criteria: Custom ranking criteria
            top_k: Number of results

        Returns:
            RerankResults object
        """
        custom_prompt = f"""Rank these documents based on the query AND the following criteria:

Query: {query}
Criteria: {criteria}

Documents:
{chr(10).join(f"ID: {d['id']}\nContent: {d['content'][:300]}..." for d in results[:self.max_documents])}

Return JSON array: [{{"id": "...", "score": 0.0-1.0, "reason": "..."}}]"""

        # Temporarily override prompt
        original_prompt = self.prompt_template
        self.prompt_template = custom_prompt

        try:
            return self.rerank(query, results, top_k)
        finally:
            self.prompt_template = original_prompt

    def __repr__(self) -> str:
        return f"LLMReranker(llm={type(self.llm_client).__name__})"
