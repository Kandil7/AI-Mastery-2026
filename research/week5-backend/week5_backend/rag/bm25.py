from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BM25Result:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class BM25Index:
    def __init__(self, chunk_ids: List[str], doc_ids: List[str], texts: List[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError("rank-bm25 package not installed") from exc

        self._chunk_ids = chunk_ids
        self._doc_ids = doc_ids
        self._texts = texts
        tokenized = [text.split() for text in texts]
        self._bm25 = BM25Okapi(tokenized)

    def query(self, query: str, top_k: int) -> List[BM25Result]:
        scores = self._bm25.get_scores(query.split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[BM25Result] = []
        for idx, score in ranked:
            results.append(
                BM25Result(
                    chunk_id=self._chunk_ids[idx],
                    doc_id=self._doc_ids[idx],
                    text=self._texts[idx],
                    score=float(score),
                )
            )
        return results
