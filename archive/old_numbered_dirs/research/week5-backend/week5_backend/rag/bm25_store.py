from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rag.bm25 import BM25Index
from rag.chunking import Chunk


@dataclass(frozen=True)
class BM25Corpus:
    chunk_ids: List[str]
    doc_ids: List[str]
    texts: List[str]


def build_corpus(chunks: List[Chunk]) -> BM25Corpus:
    return BM25Corpus(
        chunk_ids=[chunk.chunk_id for chunk in chunks],
        doc_ids=[chunk.doc_id for chunk in chunks],
        texts=[chunk.text for chunk in chunks],
    )


def save_corpus(corpus: BM25Corpus, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx in range(len(corpus.chunk_ids)):
            record = {
                "chunk_id": corpus.chunk_ids[idx],
                "doc_id": corpus.doc_ids[idx],
                "text": corpus.texts[idx],
            }
            handle.write(json.dumps(record) + "\n")


def load_corpus(path: Path) -> BM25Corpus | None:
    if not path.exists():
        return None
    chunk_ids: List[str] = []
    doc_ids: List[str] = []
    texts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        chunk_ids.append(record["chunk_id"])
        doc_ids.append(record["doc_id"])
        texts.append(record["text"])
    return BM25Corpus(chunk_ids=chunk_ids, doc_ids=doc_ids, texts=texts)


def load_bm25_index(path: Path) -> BM25Index | None:
    corpus = load_corpus(path)
    if not corpus:
        return None
    return BM25Index(corpus.chunk_ids, corpus.doc_ids, corpus.texts)
