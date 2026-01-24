from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.factories import create_embeddings_provider, create_vector_store
from core.settings import load_settings
from rag.chunking import simple_chunk
from rag.embeddings import EmbeddingService
from rag.ingestion import index_text
from rag.bm25_store import build_corpus, save_corpus


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_sources(source: Path) -> list[Path]:
    if source.is_dir():
        files: list[Path] = []
        for ext in (".txt", ".md"):
            files.extend(source.rglob(f"*{ext}"))
        return files
    return [source]


def run_index(source: Path) -> None:
    settings = load_settings()
    embedder = EmbeddingService(create_embeddings_provider(settings))
    store = create_vector_store(settings)
    all_chunks = []
    for path in _iter_sources(source):
        text = _read_text(path)
        chunks = simple_chunk(text=text, doc_id=path.stem)
        all_chunks.extend(chunks)
        index_text(
            doc_id=path.stem,
            text=text,
            embedder=embedder,
            vector_store=store,
            metadata={"path": str(path)},
            chunks=chunks,
        )
    bm25_path = Path(settings.raw.get("bm25_index_path", "data/bm25_index.jsonl"))
    if not bm25_path.is_absolute():
        bm25_path = Path(__file__).resolve().parents[1] / bm25_path
    save_corpus(build_corpus(all_chunks), bm25_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    run_index(Path(args.source))


if __name__ == "__main__":
    main()
