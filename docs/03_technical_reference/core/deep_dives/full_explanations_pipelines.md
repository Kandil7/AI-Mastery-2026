# Pipelines package (pipelines/*.py)

## pipelines/online_query.py

### _build_tools(settings) -> ToolRegistry
- Purpose: builds tool registry for agentic mode.
- Registers RAG tool by default.
- Registers SQL tool if `tools.sql.dsn` is set.
- Registers Web tool if `tools.web.base_url` is set.

### run_query_pipeline(tenant_id, question, filters, top_k, mode) -> Dict[str, Any]
- Purpose: main online query execution path.
- Steps (agentic mode):
  - Build tools and run `AgentExecutor`.
  - Choose provider for synthesis using routing policy.
  - Generate final answer from tool outputs.
  - Verify answer; fallback to RAG tool if verification fails.
- Steps (non-agentic):
  - Build `FusionConfig` from settings.
  - Choose provider for QA.
  - Create vector store, embeddings provider, optional BM25.
  - Optional query rewrite.
  - Retrieve candidates with `HybridRetriever`.
  - Optional rerank with `Reranker`.
  - Generate answer with context budget.
  - Format citations.
  - Optional verification and strict fallback.
- Returns:
  - `answer`, `citations`, `trace_id`, `model`.

## pipelines/offline_index.py

### _read_text(path: Path) -> str
- Reads a file as UTF-8 text.

### _iter_sources(source: Path) -> list[Path]
- If source is a directory, returns all `.txt` and `.md` files recursively.
- If source is a file, returns that file only.

### run_index(source: Path) -> None
- Loads settings and builds embedding and vector store.
- Builds chunker using chunking config.
- Reads files, chunks text, and indexes them.
- Builds and saves BM25 corpus JSONL.

### main() -> None
- CLI entry point.
- Parses `--source` and calls `run_index`.
