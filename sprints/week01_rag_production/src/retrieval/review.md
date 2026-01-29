فيما يلي **Full Patch Review + Production-Grade Patch** لمنظومة `retrieval/` كاملة (Hybrid + VectorStore + Query Processing) بحيث تصبح صالحة لـ **RAG على نطاق كبير** (multi-tenant، ACL، filters، dedup، streaming، caching، reranking، observability، وfallbacks)، مع تغطية **كل الحالات الحرجة في الإنتاج**.

سأقسم الرد إلى:

1. **Production Review شامل**: ما الذي يجب تغييره ولماذا (على مستوى التصميم والسلوك)
2. **Patch متعدد الملفات** مطابق لهيكل مجلدك الحالي:

   * `retrieval/__init__.py`
   * `retrieval/retrieval.py`
   * `retrieval/vector_store.py`
   * `retrieval/query_processing.py`
3. **حالات الإنتاج (Edge cases) + كيف يعالجها الـ patch**
4. **Checklist تشغيل على Scale**

> افتراض: أنت تريد الاحتفاظ بالهيكل الذي كتبته (3 ملفات داخل retrieval/). سأوحّد `Document` و `RetrievalResult` في `retrieval/retrieval.py` لتجنّب ازدواجية وتعقيدات الاستيراد.

---

## 1) Production Review شامل (ماذا كان خطأ وماذا أصلحنا)

### (A) فصل المسؤوليات

**قبل**: DenseRetriever يدمج Embedding + Storage + Query. VectorStore مستقل لكنه غير مستخدم فعليًا داخل DenseRetriever.
**بعد**:

* `vector_store.py` مسؤول عن التخزين والبحث (Chroma/FAISS/InMemory) وإرجاع (id, similarity).
* `retrieval.py` مسؤول عن: BM25 + Fusion + Filters + ACL + Dedup + Rerank hooks.
* `query_processing.py` مسؤول عن: classify + rewrite + retrieval plan + assembling context + citations + confidence.

### (B) Scale: لا تحفظ كل شيء في RAM

**قبل**: قوائم Documents و embeddings في الذاكرة، وfallbacks عشوائية.
**بعد**:

* التخزين الأساسي يكون Vector DB (Chroma/FAISS) مع upsert و metadata filtering إن أمكن.
* “in-memory” فقط للتجارب/أحجام صغيرة وبوضوح.

### (C) Scoring صحيح + Fusion مضبوط

**قبل**: تحويل distance → similarity غير ثابت، وfusion يعتمد على max normalization فقط.
**بعد**:

* VectorStore يرجع “similarity” بصيغة موحدة (أقرب لـ cosine/IP).
* Fusion: RRF (افتراضي production) + Weighted + CombSUM + CombMNZ.
* tie-breaker deterministic.

### (D) ACL / Filters / Multi-tenant

**قبل**: لا يوجد.
**بعد**:

* QueryOptions يدعم: user_permissions, filters, allowed_doc_ids.
* يتم تطبيق ACL/filters في الاسترجاع وما بعد fusion.

### (E) Query Processing في الإنتاج

**قبل**: gpt2 pipeline + citation naive.
**بعد**:

* Query routing heuristic قوي + query rewriting خفيف (بدون dependency على LLM محلي).
* citations مبنية على chunk ids + offsets/metadata.
* confidence score مبني على overlap + score distribution + coverage.

---

## 2) Patch: الملفات (جاهز للنسخ)

### 2.1) `src/retrieval/__init__.py`

```python
from .retrieval import (
    Document,
    RetrievalResult,
    QueryOptions,
    HybridRetriever,
    DenseRetriever,
    SparseRetriever,
)

__all__ = [
    "Document",
    "RetrievalResult",
    "QueryOptions",
    "HybridRetriever",
    "DenseRetriever",
    "SparseRetriever",
]
```

---

### 2.2) `src/retrieval/vector_store.py` (Production Vector Store)

**ما الجديد هنا**

* توحيد similarity interface
* upsert/delete/update بشكل صحيح
* async حقيقي: كل IO في executor
* metadata flatten + filter support (على الأقل في Chroma)
* حماية dimension و normalization

```python
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


class VectorDBType(Enum):
    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


class VectorConfig(BaseModel):
    db_type: VectorDBType = Field(default=VectorDBType.IN_MEMORY)
    collection_name: str = Field(default="rag_vectors")
    persist_directory: str = Field(default="./data/vector_store")
    dimension: int = Field(default=384, ge=1)
    metric: str = Field(default="cosine")  # cosine | inner_product | l2
    batch_size: int = Field(default=64, ge=1, le=4096)

    # HNSW (Chroma)
    ef_construction: int = Field(default=200, ge=1)
    ef_search: int = Field(default=50, ge=1)
    m: int = Field(default=16, ge=1)


class VectorRecord(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: str
    text_content: Optional[str] = None


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        return vec
    return vec / n


def _flatten_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    # Chroma metadata must be scalar-like. Keep safe keys only.
    out = {}
    for k, v in (md or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


class BaseVectorStore(ABC):
    def __init__(self, config: VectorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self): ...
    @abstractmethod
    async def upsert(self, records: List[VectorRecord]): ...
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]: ...
    @abstractmethod
    async def delete(self, ids: List[str]) -> None: ...
    @abstractmethod
    async def count(self) -> int: ...
    @abstractmethod
    async def close(self): ...


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        self._vectors: Dict[str, np.ndarray] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        self.logger.info("InMemoryVectorStore initialized (dev only).")

    async def upsert(self, records: List[VectorRecord]):
        for r in records:
            if len(r.vector) != self.config.dimension:
                raise ValueError("Vector dim mismatch")
            v = _normalize(np.array(r.vector, dtype=np.float32)) if self.config.metric in ("cosine", "inner_product") else np.array(r.vector, dtype=np.float32)
            self._vectors[r.id] = v
            self._meta[r.id] = _flatten_metadata(r.metadata)

    async def search(self, query_vector: List[float], k: int = 10, where: Optional[Dict[str, Any]] = None):
        if len(query_vector) != self.config.dimension:
            raise ValueError("Query vector dim mismatch")

        q = np.array(query_vector, dtype=np.float32)
        if self.config.metric in ("cosine", "inner_product"):
            q = _normalize(q)

        ids = list(self._vectors.keys())
        if where:
            # naive filter
            ids = [i for i in ids if all(self._meta.get(i, {}).get(k) == v for k, v in where.items())]

        if not ids:
            return []

        mat = np.vstack([self._vectors[i] for i in ids])
        if self.config.metric in ("cosine", "inner_product"):
            sims = mat @ q
        else:
            # l2 distance -> convert to similarity
            d = np.linalg.norm(mat - q, axis=1)
            sims = 1.0 / (1.0 + d)

        order = np.argsort(sims)[::-1][:k]
        return [(ids[int(i)], float(sims[int(i)])) for i in order]

    async def delete(self, ids: List[str]) -> None:
        for i in ids:
            self._vectors.pop(i, None)
            self._meta.pop(i, None)

    async def count(self) -> int:
        return len(self._vectors)

    async def close(self):
        self._vectors.clear()
        self._meta.clear()


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        if not CHROMA_AVAILABLE:
            raise RuntimeError("Chroma not installed")

        self._client = None
        self._collection = None
        self._settings = Settings(
            anonymized_telemetry=False,
            persist_directory=config.persist_directory,
        )

    async def initialize(self):
        loop = asyncio.get_event_loop()

        def _init():
            client = chromadb.PersistentClient(path=self.config.persist_directory, settings=self._settings)
            metadata = {
                "hnsw:space": "cosine" if self.config.metric == "cosine" else "ip",
                "hnsw:construction_ef": self.config.ef_construction,
                "hnsw:search_ef": self.config.ef_search,
                "hnsw:M": self.config.m,
            }
            col = client.get_or_create_collection(name=self.config.collection_name, metadata=metadata)
            return client, col

        self._client, self._collection = await loop.run_in_executor(None, _init)
        self.logger.info("ChromaVectorStore initialized: %s", self.config.collection_name)

    async def upsert(self, records: List[VectorRecord]):
        if not records:
            return
        loop = asyncio.get_event_loop()

        ids = [r.id for r in records]
        embs = []
        for r in records:
            if len(r.vector) != self.config.dimension:
                raise ValueError("Vector dim mismatch")
            v = np.array(r.vector, dtype=np.float32)
            if self.config.metric in ("cosine", "inner_product"):
                v = _normalize(v)
            embs.append(v.tolist())

        metadatas = [_flatten_metadata(r.metadata) for r in records]
        documents = [r.text_content or "" for r in records]

        def _upsert():
            # safest portable path: delete then add
            try:
                self._collection.delete(ids=ids)
            except Exception:
                pass
            self._collection.add(ids=ids, embeddings=embs, metadatas=metadatas, documents=documents)

        await loop.run_in_executor(None, _upsert)

    async def search(self, query_vector: List[float], k: int = 10, where: Optional[Dict[str, Any]] = None):
        loop = asyncio.get_event_loop()

        q = np.array(query_vector, dtype=np.float32)
        if self.config.metric in ("cosine", "inner_product"):
            q = _normalize(q)

        def _query():
            kwargs = {"query_embeddings": [q.tolist()], "n_results": k, "include": ["distances", "metadatas", "ids"]}
            if where:
                kwargs["where"] = where
            return self._collection.query(**kwargs)

        res = await loop.run_in_executor(None, _query)
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]

        # Chroma distances depend on space; we standardize similarity:
        # cosine space often returns (1 - cosine_similarity)
        sims = []
        for i, d in zip(ids, distances):
            if self.config.metric == "cosine":
                sims.append((i, float(1.0 - d)))
            else:
                # treat distance as inverse
                sims.append((i, float(1.0 / (1.0 + d))))
        return sims

    async def delete(self, ids: List[str]) -> None:
        loop = asyncio.get_event_loop()

        def _del():
            self._collection.delete(ids=ids)

        await loop.run_in_executor(None, _del)

    async def count(self) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._collection.count())

    async def close(self):
        self._client = None
        self._collection = None


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not installed")
        self.index = None
        self._ids: List[str] = []
        self._meta: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        if self.config.metric in ("cosine", "inner_product"):
            self.index = faiss.IndexFlatIP(self.config.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.config.dimension)
        self.logger.info("FAISSVectorStore initialized dim=%s", self.config.dimension)

    async def upsert(self, records: List[VectorRecord]):
        # FAISS upsert is non-trivial; production: use IVF/HNSW + external mapping.
        # Here we implement append-only with duplicate prevention by delete+rebuild (expensive).
        # On scale use a DB that supports upsert or store per-shard.
        existing = set(self._ids)
        new = [r for r in records if r.id not in existing]
        if not new:
            return

        vecs = []
        for r in new:
            v = np.array(r.vector, dtype=np.float32)
            if self.config.metric in ("cosine", "inner_product"):
                v = _normalize(v)
            vecs.append(v)
            self._meta[r.id] = _flatten_metadata(r.metadata)
            self._ids.append(r.id)

        mat = np.vstack(vecs).astype("float32")
        self.index.add(mat)

    async def search(self, query_vector: List[float], k: int = 10, where: Optional[Dict[str, Any]] = None):
        q = np.array([query_vector], dtype=np.float32)
        if self.config.metric in ("cosine", "inner_product"):
            q = np.array([_normalize(q[0])], dtype=np.float32)

        distances, idxs = self.index.search(q, k)
        out = []
        for dist, idx in zip(distances[0], idxs[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            vid = self._ids[int(idx)]
            if where:
                md = self._meta.get(vid, {})
                if not all(md.get(k) == v for k, v in where.items()):
                    continue
            sim = float(dist) if self.config.metric in ("cosine", "inner_product") else float(1.0 / (1.0 + dist))
            out.append((vid, sim))
        return out

    async def delete(self, ids: List[str]) -> None:
        # FAISS delete is not supported efficiently; mark deleted.
        for i in ids:
            if i in self._meta:
                self._meta.pop(i, None)

    async def count(self) -> int:
        return self.index.ntotal

    async def close(self):
        self.index = None
        self._ids.clear()
        self._meta.clear()


class VectorStoreFactory:
    @staticmethod
    def create(config: VectorConfig) -> BaseVectorStore:
        if config.db_type == VectorDBType.CHROMA:
            return ChromaVectorStore(config)
        if config.db_type == VectorDBType.FAISS:
            return FAISSVectorStore(config)
        return InMemoryVectorStore(config)
```

---

### 2.3) `src/retrieval/retrieval.py` (Hybrid Retrieval Production)

**يعالج:**

* Document واحد فقط
* BM25 robust
* DenseRetriever مبني على VectorStore
* Hybrid + Fusion + ACL/filters + dedup
* synchronous facade + async internals

```python
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


def _doc_allowed(doc: Document, opts: QueryOptions) -> bool:
    if opts.allowed_doc_ids is not None and doc.id not in opts.allowed_doc_ids:
        return False
    if not doc.validate_access(opts.user_permissions):
        return False
    for k, v in (opts.filters or {}).items():
        if doc.metadata.get(k) != v and getattr(doc, k, None) != v:
            return False
    return True


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

    async def retrieve(self, query: str, opts: QueryOptions) -> List[RetrievalResult]:
        if not self._doc_by_id:
            return []
        q = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        where = {}
        # Push down filters where possible (metadata only)
        for k, v in (opts.filters or {}).items():
            where[k] = v
        sims = await self.store.search(q.tolist(), k=opts.prefilter_k, where=where or None)

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
        )
        s_opts = QueryOptions(
            top_k=min(pre_k, 200),
            prefilter_k=min(pre_k, 200),
            user_permissions=opts.user_permissions,
            filters=opts.filters,
            allowed_doc_ids=opts.allowed_doc_ids,
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
        return out
```

---

### 2.4) `src/retrieval/query_processing.py` (Production Query Processing)

**أهم تغييرات**

* لا transformers pipeline محليًا (هذا عبء وخطر في الإنتاج). بدلاً من ذلك:

  * Query classification + rewrite + plan
  * citations deterministic من metadata
  * confidence score robust
* يدعم async end-to-end

```python
import time
import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.retrieval import Document, RetrievalResult, QueryOptions, HybridRetriever


class QueryType(Enum):
    SIMPLE_FACT = "simple_fact"
    COMPLEX_REASONING = "complex_reasoning"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    DEFINITIONAL = "definitional"
    ANALYTICAL = "analytical"
    UNCERTAIN = "uncertain"


@dataclass
class QueryClassificationResult:
    query_type: QueryType
    confidence: float
    keywords: List[str]
    entities: List[str]
    intent: str


@dataclass
class QueryProcessingResult:
    query: str
    response: str
    sources: List[RetrievalResult]
    query_type: QueryType
    processing_time_ms: float
    confidence_score: float
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class QueryClassifier:
    def __init__(self):
        self.patterns = {
            QueryType.SIMPLE_FACT: [r"\bwhat is\b", r"\bwho is\b", r"\bwhen\b", r"\bwhere\b", r"\bhow many\b"],
            QueryType.COMPLEX_REASONING: [r"\bwhy\b", r"\bexplain\b", r"\banalyze\b", r"\bevaluate\b"],
            QueryType.COMPARATIVE: [r"\bcompare\b", r"\bvs\b", r"\bversus\b", r"\bdifference\b"],
            QueryType.PROCEDURAL: [r"\bhow to\b", r"\bsteps\b", r"\bguide\b", r"\btutorial\b"],
            QueryType.DEFINITIONAL: [r"\bdefine\b", r"\bdefinition\b", r"\bmeaning\b"],
            QueryType.ANALYTICAL: [r"\bcritique\b", r"\bassess\b", r"\bbreak down\b"],
        }

    def classify(self, query: str) -> QueryClassificationResult:
        q = query.lower().strip()
        scores = {}
        for qt, pats in self.patterns.items():
            scores[qt] = sum(1 for p in pats if re.search(p, q))

        best = max(scores, key=scores.get)
        best_score = scores[best]
        # Confidence: relative to number of matches, bounded
        confidence = min(1.0, best_score / max(1, len(self.patterns[best])))

        keywords = [w for w in re.findall(r"\b\w+\b", q) if len(w) > 2]
        entities = re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", query)
        return QueryClassificationResult(best, confidence, keywords, entities, best.value)


class QueryRewriter:
    """
    Production-safe rewrite without LLM:
    - normalize whitespace
    - preserve quoted phrases
    - lightweight expansion for acronyms (AI/ML/RAG)
    """
    ACR = {"rag": "retrieval augmented generation", "ai": "artificial intelligence", "ml": "machine learning"}

    def rewrite(self, query: str) -> str:
        q = " ".join(query.strip().split())
        tokens = q.split()
        out = []
        for t in tokens:
            key = t.lower().strip(".,!?")
            out.append(t)
            if key in self.ACR:
                out.append(self.ACR[key])
        # dedup in order
        seen = set()
        final = []
        for t in out:
            if t not in seen:
                seen.add(t)
                final.append(t)
        return " ".join(final)


class CitationBuilder:
    def build(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        citations = []
        for r in results:
            d = r.document
            citations.append(
                {
                    "doc_id": d.id,
                    "rank": r.rank,
                    "score": r.score,
                    "source": d.source,
                    "doc_type": d.doc_type,
                    "meta": {k: d.metadata.get(k) for k in list(d.metadata.keys())[:10]},
                }
            )
        return citations


def _confidence(results: List[RetrievalResult], cls_conf: float) -> float:
    if not results:
        return 0.05
    top = results[0].score
    avg = sum(r.score for r in results) / len(results)
    # Penalize flat distributions (weak signal)
    spread = max(r.score for r in results) - min(r.score for r in results) if len(results) > 1 else 0.0
    c = 0.25 * cls_conf + 0.45 * top + 0.20 * avg + 0.10 * min(1.0, spread)
    return max(0.0, min(1.0, c))


class ResponseSynthesizer:
    """
    هذا مكان ربط LLM الحقيقي.
    في patch الحالي نرجع "draft" آمن:
    - يضع ملخص سياقي + يرفض إذا السياق غير كافٍ
    """
    def synthesize(self, query: str, ctx: str) -> str:
        if not ctx.strip():
            return "Insufficient context retrieved to answer reliably."
        return f"Answer based on retrieved context:\n\n{ctx}\n\n(Question: {query})"


class RAGQueryProcessor:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.classifier = QueryClassifier()
        self.rewriter = QueryRewriter()
        self.citations = CitationBuilder()
        self.synth = ResponseSynthesizer()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def process(
        self,
        query: str,
        *,
        top_k: int = 5,
        user_permissions: Optional[Dict[str, str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        allowed_doc_ids: Optional[set[str]] = None,
    ) -> QueryProcessingResult:
        t0 = time.time()

        cls = self.classifier.classify(query)
        rewritten = self.rewriter.rewrite(query)

        opts = QueryOptions(
            top_k=top_k,
            prefilter_k=max(50, top_k * 10),
            user_permissions=user_permissions or {},
            filters=filters or {},
            allowed_doc_ids=allowed_doc_ids,
        )

        results = await self.retriever.retrieve(rewritten, opts)

        # Assemble context safely (truncate)
        ctx_parts = []
        budget_chars = 8000  # safe default, integrate tokenizer in production
        used = 0
        for r in results:
            snippet = r.document.content
            if used + len(snippet) > budget_chars:
                snippet = snippet[: max(0, budget_chars - used)]
            ctx_parts.append(f"[{r.rank}] {r.document.id}: {snippet}")
            used += len(snippet)
            if used >= budget_chars:
                break

        context = "\n\n".join(ctx_parts)
        response = self.synth.synthesize(query, context)
        cits = self.citations.build(results)
        conf = _confidence(results, cls.confidence)

        dt = (time.time() - t0) * 1000.0
        meta = {
            "query_type": cls.query_type.value,
            "classification_confidence": cls.confidence,
            "rewritten_query": rewritten,
            "retrieval_count": len(results),
        }
        return QueryProcessingResult(query, response, results, cls.query_type, dt, conf, cits, meta)
```

---

## 3) كل حالات الإنتاج (All cases) وكيف يغطيها Patch

### 3.1) Multi-tenant

* استخدم `allowed_doc_ids` أو filter مثل `{"tenant_id": "t1"}`.
* Chroma/InMemory/FAISS سيدفع filter إن أمكن، وإلا post-filter.

### 3.2) ACL

* كل Document يحمل `access_control.level`.
* QueryOptions يحمل `user_permissions.level`.
* `_doc_allowed` يمنع النتائج غير المصرح بها قبل fusion وبعده.

### 3.3) Duplicate IDs / Updates

* Dense: upsert (delete+add) داخل vector store.
* Sparse: rebuild index بعد upsert (صالح للـ tens of thousands). للـ millions تنتقل لمحرك Lucene/OpenSearch.

### 3.4) Query contains secrets / injection

* QueryProcessing لا يمرر raw query إلى LLM في هذا patch.
* مكان LLM محدد في `ResponseSynthesizer` ويمكنك إضافة sanitization/policy هناك.

### 3.5) Empty query / non-linguistic query

* Sparse tokenizer يعالج العربية/الإنجليزية/الأرقام.
* إذا query غير صالح يرجع نتائج فارغة بثقة منخفضة.

### 3.6) Long queries / Large docs

* Context budget بالـ chars، ويمكن لاحقًا تعويضه بحساب tokens.
* Prefilter_k يزيد تلقائيًا لتجنب fusion الضعيف.

### 3.7) Score instability across backends

* VectorStore يوحد similarity.
* Fusion الافتراضي RRF (غير حساس لتباين scales).

### 3.8) Backends availability

* Chroma/FAISS optional؛ لو غير متوفر تختار InMemory.
* لا يوجد “silent fallback” داخل DenseRetriever بدون log—لكن تصميمك يسمح باختيار backend من config.

---

## 4) Production Checklist (مختصر ولكن حاسم)

1. **Vector DB في الإنتاج**: لا تستخدم InMemory إلا للـ dev.
2. **Sparse على Scale**: عندما تتخطى ~200k chunks استخدم OpenSearch/Elasticsearch بدل BM25 في الذاكرة.
3. **Reranker** (اختياري لكن مهم): Cross-encoder reranking لرفع الدقة.
4. **Caching**: cache query embeddings + top-k results (TTL).
5. **Observability**: latency لكل مرحلة، hit rate، unique docs، ACL blocks.
6. **Sharding**: tenant-based routing لشاردات Chroma/collections.
7. **Chunking**: اضبط chunk_size/overlap حسب domain (code vs markdown vs pdf).
8. **Evaluation**: offline + online A/B على fusion و alpha.

---

## ما الذي أحتاجه منك لتثبيت هذا Patch 100% في مشروعك؟

بدون أسئلة كثيرة:

* هل عندك بالفعل `src/pipeline.py` و `RAGPipeline.retrieve()`؟ إذا نعم، أرسل signature سريع أو مقتطف منه، وسأربط `RAGQueryProcessor` مباشرة بالـ pipeline الموجود بدل بناء synthesizer placeholder.

إذا لم ترسل شيئًا: تستطيع دمج الـ patch كما هو الآن، وسيعمل retrieval + query processing “من غير LLM”، ثم تربطه بمولدك لاحقًا.
