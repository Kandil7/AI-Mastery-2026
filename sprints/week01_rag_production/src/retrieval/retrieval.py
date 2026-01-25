"""
Advanced Hybrid Retrieval System for Production RAG Applications

This module implements a sophisticated hybrid retrieval system that combines dense and sparse
retrieval techniques to achieve optimal performance across different query types. The system
leverages both semantic understanding and keyword matching to provide robust information
retrieval capabilities suitable for production environments.

The architecture follows the 2026 RAG production standards with emphasis on:
- Hybrid retrieval combining dense (vector) and sparse (keyword) search
- Reciprocal Rank Fusion (RRF) for optimal result combination
- Persistent storage with ChromaDB
- Configurable fusion strategies
- Comprehensive error handling and performance optimization
- Caching for query embeddings and results
- Deduplication and reranking capabilities

References:
- RRF (Reciprocal Rank Fusion): https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Dense retrieval using Sentence Transformers
- Sparse retrieval using TF-IDF and BM25 variants
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .vector_store import VectorConfig, VectorRecord, VectorStoreFactory


@dataclass(frozen=True)
class Document:
    id: str
    content: str
    source: str = "unknown"
    doc_type: str = "unspecified"
    access_control: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_access(self, user_permissions: Dict[str, str]) -> bool:
        doc_level = self.access_control.get("level", "public")
        user_level = user_permissions.get("level", "public")
        levels = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
        return levels.get(user_level, 0) >= levels.get(doc_level, 0)


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    rank: int
    source: str  # dense|sparse|hybrid


@dataclass(frozen=True)
class QueryOptions:
    top_k: int = 5
    prefilter_k: int = 50
    user_permissions: Dict[str, str] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    allowed_doc_ids: Optional[set[str]] = None
    rerank: bool = True  # Whether to apply reranking
    deduplicate: bool = True  # Whether to deduplicate results


def _doc_allowed(doc: Document, opts: QueryOptions) -> bool:
    if opts.allowed_doc_ids is not None and doc.id not in opts.allowed_doc_ids:
        return False
    if not doc.validate_access(opts.user_permissions):
        return False
    for k, v in (opts.filters or {}).items():
        if doc.metadata.get(k) != v and getattr(doc, k, None) != v:
            return False
    return True


def _deduplicate_results(results: List[RetrievalResult], threshold: float = 0.9) -> List[RetrievalResult]:
    """
    Remove near-duplicate results based on content similarity.

    Args:
        results: List of retrieval results
        threshold: Similarity threshold above which documents are considered duplicates

    Returns:
        List of deduplicated results
    """
    if not results:
        return results

    unique_results = [results[0]]  # Always keep the first result

    for result in results[1:]:
        is_duplicate = False
        for unique_result in unique_results:
            # Simple similarity check based on content overlap
            content_sim = _jaccard_similarity(result.document.content, unique_result.document.content)
            if content_sim > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_results.append(result)

    return unique_results


def _jaccard_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Calculate Jaccard similarity between two texts using n-grams.

    Args:
        text1: First text
        text2: Second text
        n: Size of n-grams

    Returns:
        Jaccard similarity score between 0 and 1
    """
    def get_ngrams(text: str, n: int) -> set:
        text = text.lower()
        return set(text[i:i+n] for i in range(len(text)-n+1))

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))

    return intersection / union if union != 0 else 0.0


def _rerank_results(query: str, results: List[RetrievalResult], top_k: int = None) -> List[RetrievalResult]:
    """
    Rerank results using cross-encoder or other reranking method.

    Args:
        query: Original query
        results: List of retrieval results to rerank
        top_k: Number of top results to return after reranking

    Returns:
        Reranked list of results
    """
    if not results:
        return results

    # For now, implement a simple reranking based on content-query similarity
    # In production, this would use a cross-encoder model
    reranked_results = []
    for result in results:
        # Calculate a simple relevance score based on keyword overlap
        query_words = set(query.lower().split())
        content_words = set(result.document.content.lower().split())
        overlap = len(query_words.intersection(content_words))
        relevance_score = overlap / len(query_words) if query_words else 0.0

        # Combine with original score
        combined_score = 0.3 * result.score + 0.7 * relevance_score
        reranked_results.append(
            RetrievalResult(
                document=result.document,
                score=combined_score,
                rank=result.rank,  # Will be updated after sorting
                source=result.source
            )
        )

    # Sort by combined score and update ranks
    reranked_results.sort(key=lambda x: x.score, reverse=True)
    for i, result in enumerate(reranked_results):
        reranked_results[i] = RetrievalResult(
            document=result.document,
            score=result.score,
            rank=i + 1,
            source=result.source
        )

    # Return top_k if specified
    if top_k is not None:
        reranked_results = reranked_results[:top_k]

    return reranked_results


_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u0600-\u06FF]+", re.UNICODE)
def _tok(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Document] = []
        self._tf: List[Dict[str, int]] = []
        self._dl: List[int] = []
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._doc_by_id: Dict[str, int] = {}

    def index(self, docs: List[Document]) -> None:
        for d in docs:
            if d.id in self._doc_by_id:
                self._docs[self._doc_by_id[d.id]] = d
            else:
                self._doc_by_id[d.id] = len(self._docs)
                self._docs.append(d)
        self._rebuild()

    def _rebuild(self) -> None:
        self._tf, self._dl, self._df = [], [], {}
        for d in self._docs:
            tokens = _tok(d.content)
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self._tf.append(freq)
            self._dl.append(len(tokens))
            for t in freq:
                self._df[t] = self._df.get(t, 0) + 1
        N = len(self._docs)
        self._avgdl = float(sum(self._dl)) / max(1, N)
        self._idf = {t: math.log(1.0 + (N - df + 0.5) / (df + 0.5)) for t, df in self._df.items()}

    def retrieve(self, query: str, opts: QueryOptions) -> List[RetrievalResult]:
        if not self._docs:
            return []
        q = _tok(query)
        if not q:
            return []

        candidates = [i for i, d in enumerate(self._docs) if _doc_allowed(d, opts)]
        if not candidates:
            return []

        scores = np.zeros(len(candidates), dtype=np.float32)
        uq = set(q)
        for term in uq:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for pos, idx in enumerate(candidates):
                tf = self._tf[idx].get(term, 0)
                if tf == 0:
                    continue
                dl = self._dl[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self._avgdl)))
                scores[pos] += idf * (tf * (self.k1 + 1) / denom)

        k = min(opts.top_k, len(candidates))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        out = []
        for rank, p in enumerate(top, start=1):
            di = candidates[int(p)]
            out.append(RetrievalResult(self._docs[di], float(scores[int(p)]), rank, "sparse"))

        # Apply deduplication if enabled
        if opts.deduplicate:
            out = _deduplicate_results(out)

        # Apply reranking if enabled
        if opts.rerank:
            out = _rerank_results(query, out, opts.top_k)

        return out


class DenseRetriever:
    def __init__(
        self,
        vector_config: VectorConfig,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.encoder = SentenceTransformer(model_name)
        self.store = VectorStoreFactory.create(vector_config)
        self._doc_by_id: Dict[str, Document] = {}

        # Cache for query embeddings
        self._query_embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 1000  # Max cache size

    async def initialize(self):
        await self.store.initialize()

    async def index(self, docs: List[Document]) -> None:
        if not docs:
            return
        self._doc_by_id.update({d.id: d for d in docs})
        texts = [d.content for d in docs]
        embs = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        recs = []
        for d, e in zip(docs, embs):
            md = {
                **(d.metadata or {}),
                "source": d.source,
                "doc_type": d.doc_type,
                "access_level": d.access_control.get("level", "public"),
            }
            recs.append(VectorRecord(id=d.id, vector=e.tolist(), metadata=md, document_id=d.id, text_content=None))
        await self.store.upsert(recs)

    def _get_cached_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get cached embedding for a query if available."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self._query_embedding_cache.get(query_hash)

    def _cache_embedding(self, query: str, embedding: np.ndarray):
        """Cache an embedding for a query."""
        if len(self._query_embedding_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._query_embedding_cache))
            del self._query_embedding_cache[oldest_key]

        query_hash = hashlib.md5(query.encode()).hexdigest()
        self._query_embedding_cache[query_hash] = embedding

    async def retrieve(self, query: str, opts: QueryOptions) -> List[RetrievalResult]:
        if not self._doc_by_id:
            return []

        # Check cache first for query embedding
        query_embedding = self._get_cached_embedding(query)
        if query_embedding is None:
            query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            self._cache_embedding(query, query_embedding)

        where = {}
        # Push down filters where possible (metadata only)
        for k, v in (opts.filters or {}).items():
            where[k] = v
        sims = await self.store.search(query_embedding.tolist(), k=opts.prefilter_k, where=where or None)

        # Post-filter ACL + allowed ids
        docs = []
        for doc_id, sim in sims:
            d = self._doc_by_id.get(doc_id)
            if d and _doc_allowed(d, opts):
                docs.append((d, sim))

        docs.sort(key=lambda x: x[1], reverse=True)
        out = []
        for i, (d, s) in enumerate(docs[: opts.top_k], start=1):
            out.append(RetrievalResult(d, float(s), i, "dense"))

        # Apply deduplication if enabled
        if opts.deduplicate:
            out = _deduplicate_results(out)

        # Apply reranking if enabled
        if opts.rerank:
            out = _rerank_results(query, out, opts.top_k)

        return out


def _rrf(dense: List[RetrievalResult], sparse: List[RetrievalResult], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for r in dense:
        scores[r.document.id] = scores.get(r.document.id, 0.0) + 1.0 / (k + r.rank)
    for r in sparse:
        scores[r.document.id] = scores.get(r.document.id, 0.0) + 1.0 / (k + r.rank)
    return scores


def _minmax(m: Dict[str, float]) -> Dict[str, float]:
    if not m:
        return {}
    vals = list(m.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.0 for k in m}
    return {k: (v - lo) / (hi - lo) for k, v in m.items()}


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        fusion: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion.lower()
        self.alpha = float(alpha)
        self.rrf_k = int(rrf_k)

    def index_sparse(self, docs: List[Document]) -> None:
        self.sparse.index(docs)

    async def index_dense(self, docs: List[Document]) -> None:
        await self.dense.index(docs)

    async def retrieve(self, query: str, opts: QueryOptions) -> List[RetrievalResult]:
        pre_k = max(opts.prefilter_k, opts.top_k * 5)
        d_opts = QueryOptions(
            top_k=min(pre_k, 200),
            prefilter_k=min(pre_k, 200),
            user_permissions=opts.user_permissions,
            filters=opts.filters,
            allowed_doc_ids=opts.allowed_doc_ids,
            rerank=False,  # Don't rerank individual results, rerank after fusion
            deduplicate=False,  # Don't deduplicate individual results, deduplicate after fusion
        )
        s_opts = QueryOptions(
            top_k=min(pre_k, 200),
            prefilter_k=min(pre_k, 200),
            user_permissions=opts.user_permissions,
            filters=opts.filters,
            allowed_doc_ids=opts.allowed_doc_ids,
            rerank=False,  # Don't rerank individual results, rerank after fusion
            deduplicate=False,  # Don't deduplicate individual results, deduplicate after fusion
        )

        dense_res = await self.dense.retrieve(query, d_opts)
        sparse_res = self.sparse.retrieve(query, s_opts)

        doc_map: Dict[str, Document] = {}
        for r in dense_res:
            doc_map[r.document.id] = r.document
        for r in sparse_res:
            doc_map.setdefault(r.document.id, r.document)

        if self.fusion == "rrf":
            fused = _rrf(dense_res, sparse_res, k=self.rrf_k)
        elif self.fusion == "weighted":
            d = _minmax({r.document.id: r.score for r in dense_res})
            s = _minmax({r.document.id: r.score for r in sparse_res})
            fused = {}
            for doc_id, v in d.items():
                fused[doc_id] = fused.get(doc_id, 0.0) + self.alpha * v
            for doc_id, v in s.items():
                fused[doc_id] = fused.get(doc_id, 0.0) + (1 - self.alpha) * v
        elif self.fusion == "combsum":
            d = _minmax({r.document.id: r.score for r in dense_res})
            s = _minmax({r.document.id: r.score for r in sparse_res})
            fused = {}
            for doc_id, v in d.items():
                fused[doc_id] = fused.get(doc_id, 0.0) + v
            for doc_id, v in s.items():
                fused[doc_id] = fused.get(doc_id, 0.0) + v
        elif self.fusion == "combmnz":
            d = _minmax({r.document.id: r.score for r in dense_res})
            s = _minmax({r.document.id: r.score for r in sparse_res})
            fused = {}
            cnt = {}
            for doc_id, v in d.items():
                fused[doc_id] = fused.get(doc_id, 0.0) + v
                cnt[doc_id] = cnt.get(doc_id, 0) + 1
            for doc_id, v in s.items():
                fused[doc_id] = fused.get(doc_id, 0.0) + v
                cnt[doc_id] = cnt.get(doc_id, 0) + 1
            for doc_id in list(fused.keys()):
                fused[doc_id] *= cnt.get(doc_id, 1)
        else:
            fused = _rrf(dense_res, sparse_res, k=self.rrf_k)

        ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[: opts.top_k]
        out = []
        for i, (doc_id, score) in enumerate(ranked, start=1):
            d = doc_map.get(doc_id)
            if d is None:
                continue
            out.append(RetrievalResult(d, float(score), i, "hybrid"))

        # Apply deduplication if enabled
        if opts.deduplicate:
            out = _deduplicate_results(out)

        # Apply reranking if enabled
        if opts.rerank:
            out = _rerank_results(query, out, opts.top_k)

        return out
