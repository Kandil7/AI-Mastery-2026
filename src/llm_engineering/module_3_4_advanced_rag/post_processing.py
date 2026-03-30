"""
Post-Processing Module

Production-ready post-processing for RAG:
- Re-ranking with cross-encoders
- RAG-Fusion for result combination
- Diversity enhancement
- Answer synthesis

Features:
- Multiple reranking strategies
- MMR for diversity
- Answer combination
- Citation management
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """A document with ranking information."""

    content: str
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


class PostProcessor(ABC):
    """Abstract base class for post-processors."""

    @abstractmethod
    async def process(
        self,
        query: str,
        documents: List[RankedDocument],
        **kwargs: Any,
    ) -> List[RankedDocument]:
        """Process and rank documents."""
        pass


class Reranker(PostProcessor):
    """
    Document reranker using cross-encoders.

    Re-ranks retrieved documents using a more accurate
    but slower model than the retrieval embedding.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size

        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking unavailable.")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")

    async def process(
        self,
        query: str,
        documents: List[RankedDocument],
        **kwargs: Any,
    ) -> List[RankedDocument]:
        """Rerank documents."""
        if not self._model or not documents:
            return documents[:self.top_k]

        # Prepare pairs for scoring
        pairs = [[query, doc.content] for doc in documents]

        # Score in batches
        scores = await self._score_batch(pairs)

        # Update documents with new scores
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        # Sort by new scores
        documents.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, doc in enumerate(documents):
            doc.rank = i + 1

        return documents[:self.top_k]

    async def _score_batch(
        self,
        pairs: List[List[str]],
    ) -> List[float]:
        """Score pairs in batches."""
        if not pairs:
            return []

        all_scores = []

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = await loop.run_in_executor(
                None,
                lambda b=batch: self._model.predict(b),
            )
            all_scores.extend(scores)

        return all_scores


class RAGFusion(PostProcessor):
    """
    RAG-Fusion implementation.

    Combines results from multiple queries using
    Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        llm_client: Any,
        num_queries: int = 4,
        rrf_k: int = 60,
        top_k: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.num_queries = num_queries
        self.rrf_k = rrf_k
        self.top_k = top_k

    async def process(
        self,
        query: str,
        documents: List[RankedDocument],
        **kwargs: Any,
    ) -> List[RankedDocument]:
        """Apply RAG-Fusion."""
        # Generate query variations
        variations = await self._generate_variations(query)
        all_queries = [query] + variations

        # Get documents for each query (from kwargs or use provided)
        query_documents = kwargs.get("query_documents", {query: documents})

        # If we only have one query's documents, simulate others
        if len(query_documents) == 1:
            for q in all_queries[1:]:
                # Add slight score perturbation for simulation
                query_documents[q] = self._perturb_scores(documents, q)

        # Apply RRF
        fused = self._reciprocal_rank_fusion(query_documents)

        # Assign ranks
        for i, doc in enumerate(fused):
            doc.rank = i + 1

        return fused[:self.top_k]

    async def _generate_variations(self, query: str) -> List[str]:
        """Generate query variations."""
        prompt = f"""Generate {self.num_queries} different versions of this query.
Each should ask the same thing but use different wording.

Original: {query}

Variations (one per line):"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        content = response.content if hasattr(response, 'content') else str(response)
        variations = [line.strip() for line in content.split("\n") if line.strip()]

        return variations[:self.num_queries]

    def _perturb_scores(
        self,
        documents: List[RankedDocument],
        seed: str,
    ) -> List[RankedDocument]:
        """Perturb scores for simulation."""
        import hashlib

        # Use query as seed for deterministic perturbation
        seed_hash = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)

        perturbed = []
        for doc in documents:
            # Small deterministic perturbation
            perturbation = ((seed_hash % 100) - 50) / 1000
            new_doc = RankedDocument(
                content=doc.content,
                score=doc.score + perturbation,
                metadata=doc.metadata,
                id=doc.id,
            )
            perturbed.append(new_doc)

        return perturbed

    def _reciprocal_rank_fusion(
        self,
        query_documents: Dict[str, List[RankedDocument]],
    ) -> List[RankedDocument]:
        """Apply Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, RankedDocument] = {}

        for query, docs in query_documents.items():
            for rank, doc in enumerate(docs):
                doc_id = doc.id or hash(doc.content)
                rrf_score = 1 / (self.rrf_k + rank + 1)

                if doc_id in scores:
                    scores[doc_id] += rrf_score
                else:
                    scores[doc_id] = rrf_score
                    doc_map[doc_id] = doc

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc.score = scores[doc_id]
            fused.append(doc)

        return fused


class DiversityEnhancer(PostProcessor):
    """
    Enhances diversity in retrieved documents.

    Uses Maximal Marginal Relevance (MMR) to balance
    relevance and diversity.
    """

    def __init__(
        self,
        embedding_generator: Any,
        lambda_param: float = 0.5,
        top_k: int = 5,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.lambda_param = lambda_param  # Balance between relevance and diversity
        self.top_k = top_k

    async def process(
        self,
        query: str,
        documents: List[RankedDocument],
        **kwargs: Any,
    ) -> List[RankedDocument]:
        """Apply MMR for diversity."""
        if len(documents) <= self.top_k:
            return documents

        # Get query embedding
        query_embedding = await self.embedding_generator.embed_text(query)

        # Get document embeddings
        doc_embeddings = []
        for doc in documents:
            emb_result = await self.embedding_generator.embed_text(doc.content)
            doc_embeddings.append(emb_result.embedding)

        # Apply MMR
        selected_indices = self._mmr(
            query_embedding.embedding,
            doc_embeddings,
            [d.score for d in documents],
            self.lambda_param,
            self.top_k,
        )

        # Return selected documents
        selected = [documents[i] for i in selected_indices]

        # Reassign ranks
        for i, doc in enumerate(selected):
            doc.rank = i + 1

        return selected

    def _mmr(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        relevance_scores: List[float],
        lambda_param: float,
        k: int,
    ) -> List[int]:
        """Apply Maximal Marginal Relevance."""
        selected = []
        remaining = list(range(len(doc_embeddings)))

        while len(selected) < k and remaining:
            best_score = float("-inf")
            best_idx = None

            for idx in remaining:
                # Relevance to query
                query_sim = self._cosine_similarity(
                    query_embedding,
                    doc_embeddings[idx],
                )

                # Diversity (similarity to selected)
                max_selected_sim = 0
                if selected:
                    max_selected_sim = max(
                        self._cosine_similarity(
                            doc_embeddings[idx],
                            doc_embeddings[s],
                        )
                        for s in selected
                    )

                # MMR score
                mmr_score = (
                    lambda_param * query_sim -
                    (1 - lambda_param) * max_selected_sim
                )

                # Also consider original relevance
                mmr_score += 0.1 * relevance_scores[idx]

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot / (norm_a * norm_b)


class AnswerSynthesizer:
    """
    Synthesizes final answer from multiple sources.

    Combines information from retrieved documents into
    a coherent, cited answer.
    """

    SYNTHESIS_PROMPT = """Synthesize a comprehensive answer from the following sources.

Sources:
{sources}

Question: {question}

Instructions:
1. Combine information from all relevant sources
2. Cite sources using [1], [2], etc.
3. If sources conflict, acknowledge the disagreement
4. If information is missing, say so
5. Be concise but complete

Answer:"""

    def __init__(
        self,
        llm_client: Any,
        max_tokens: int = 2048,
        include_citations: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.max_tokens = max_tokens
        self.include_citations = include_citations

    async def synthesize(
        self,
        question: str,
        documents: List[RankedDocument],
        **kwargs: Any,
    ) -> str:
        """Synthesize answer from documents."""
        if not documents:
            return "No relevant information found to answer this question."

        # Format sources
        sources_text = self._format_sources(documents)

        prompt = self.SYNTHESIS_PROMPT.format(
            sources=sources_text,
            question=question,
        )

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=self.max_tokens,
        )

        return response.content if hasattr(response, 'content') else str(response)

    def _format_sources(self, documents: List[RankedDocument]) -> str:
        """Format documents as sources."""
        lines = []
        for i, doc in enumerate(documents):
            citation = f"[{i + 1}]" if self.include_citations else ""
            content = doc.content[:1000]  # Truncate long documents
            lines.append(f"{citation} {content}")

        return "\n\n".join(lines)

    async def synthesize_with_reflection(
        self,
        question: str,
        documents: List[RankedDocument],
        num_iterations: int = 2,
    ) -> str:
        """Synthesize with self-reflection for quality."""
        # Initial synthesis
        answer = await self.synthesize(question, documents)

        for _ in range(num_iterations):
            # Reflect and improve
            reflection_prompt = f"""Review and improve this answer.

Question: {question}

Current Answer: {answer}

Instructions:
1. Check if the answer fully addresses the question
2. Verify all claims are supported by the sources
3. Improve clarity and organization
4. Add any missing important information

Improved Answer:"""

            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.2,
                max_tokens=self.max_tokens,
            )

            answer = response.content if hasattr(response, 'content') else str(response)

        return answer


class CitationManager:
    """
    Manages citations in synthesized answers.

    Features:
    - Citation extraction
    - Citation formatting
    - Citation verification
    """

    def __init__(
        self,
        format_style: str = "numbered",  # numbered, inline, footnote
    ) -> None:
        self.format_style = format_style

    def extract_citations(self, text: str) -> List[Tuple[int, str]]:
        """Extract citations from text."""
        import re

        citations = []

        # Find numbered citations [1], [2], etc.
        pattern = r"\[(\d+)\]"
        matches = re.finditer(pattern, text)

        for match in matches:
            citation_num = int(match.group(1))
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()

            citations.append((citation_num, context))

        return citations

    def format_citations(
        self,
        documents: List[RankedDocument],
        style: Optional[str] = None,
    ) -> str:
        """Format citations for display."""
        style = style or self.format_style

        if style == "numbered":
            return self._format_numbered(documents)
        elif style == "inline":
            return self._format_inline(documents)
        elif style == "footnote":
            return self._format_footnote(documents)
        else:
            return self._format_numbered(documents)

    def _format_numbered(self, documents: List[RankedDocument]) -> str:
        """Format as numbered list."""
        lines = ["Sources:"]
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown")
            lines.append(f"[{i + 1}] {source}")

        return "\n".join(lines)

    def _format_inline(self, documents: List[RankedDocument]) -> str:
        """Format as inline citations."""
        citations = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown")
            citations.append(f"({source})")

        return ", ".join(citations)

    def _format_footnote(self, documents: List[RankedDocument]) -> str:
        """Format as footnotes."""
        lines = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown")
            lines.append(f"^{i + 1} {source}")

        return "\n".join(lines)

    def verify_citations(
        self,
        answer: str,
        documents: List[RankedDocument],
    ) -> Dict[str, Any]:
        """Verify that citations are supported by documents."""
        citations = self.extract_citations(answer)

        verification = {
            "total_citations": len(citations),
            "verified": 0,
            "unverified": 0,
            "details": [],
        }

        for citation_num, context in citations:
            if citation_num <= len(documents):
                doc = documents[citation_num - 1]
                # Check if context is supported
                is_supported = self._check_support(context, doc.content)
                verification["details"].append({
                    "citation": citation_num,
                    "supported": is_supported,
                })
                if is_supported:
                    verification["verified"] += 1
                else:
                    verification["unverified"] += 1

        return verification

    def _check_support(self, claim: str, content: str) -> bool:
        """Check if claim is supported by content."""
        # Simple keyword matching
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())

        overlap = len(claim_words & content_words)
        return overlap > len(claim_words) * 0.3  # 30% overlap threshold


class PostProcessingPipeline:
    """
    Pipeline for chaining post-processing steps.

    Features:
    - Configurable pipeline
    - Step tracking
    - Performance metrics
    """

    def __init__(self) -> None:
        self._steps: List[Tuple[str, PostProcessor]] = []
        self._stats: Dict[str, Any] = {}

    def add_step(self, name: str, processor: PostProcessor) -> "PostProcessingPipeline":
        """Add a processing step."""
        self._steps.append((name, processor))
        return self

    async def process(
        self,
        query: str,
        documents: List[RankedDocument],
        **kwargs: Any,
    ) -> List[RankedDocument]:
        """Run pipeline on documents."""
        import time
        start_time = time.time()

        current_docs = documents.copy()
        step_results = {}

        for name, processor in self._steps:
            step_start = time.time()
            current_docs = await processor.process(query, current_docs, **kwargs)
            step_latency = (time.time() - step_start) * 1000

            step_results[name] = {
                "input_count": len(documents),
                "output_count": len(current_docs),
                "latency_ms": step_latency,
            }

        total_latency = (time.time() - start_time) * 1000

        self._stats = {
            "total_latency_ms": total_latency,
            "steps": step_results,
            "input_count": len(documents),
            "output_count": len(current_docs),
        }

        return current_docs

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self._stats.copy()

    @classmethod
    def create_default(
        cls,
        reranker: Optional[Reranker] = None,
        diversity: Optional[DiversityEnhancer] = None,
    ) -> "PostProcessingPipeline":
        """Create default pipeline."""
        pipeline = cls()

        if reranker:
            pipeline.add_step("rerank", reranker)
        if diversity:
            pipeline.add_step("diversity", diversity)

        return pipeline
