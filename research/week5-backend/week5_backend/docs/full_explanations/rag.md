# RAG package (rag/*.py)

## rag/embeddings.py

### EmbeddingService
- Purpose: thin wrapper around an `EmbeddingsProvider`.
- `__init__(provider)` stores provider.
- `embed(texts)` delegates to provider.

## rag/chunking.py

### Chunk (dataclass)
- Represents a chunk of text.
- Fields: `chunk_id`, `doc_id`, `text`.

### simple_chunk(text, doc_id, max_tokens=400) -> List[Chunk]
- Word-based chunker for instructional use.
- Splits words into windows of `max_tokens`.

### structured_chunk(text, doc_id, max_tokens=400, overlap=40)
- Splits text into sections by Markdown headings.
- Preserves headings in each chunk.
- Uses `overlap` to keep context between chunks.

### build_chunker(mode, max_tokens=400, overlap=40) -> Callable
- Returns a callable chunker based on mode.
- `structured` -> `structured_chunk`, else `simple_chunk`.

### _split_sections(text) -> List[Tuple[str, str]]
- Splits text by lines starting with `#`.
- Returns `(heading, body)` pairs.

## rag/ingestion.py

### ingest_document(tenant_id, source_type, uri, metadata) -> str
- Loads settings, builds embedder and store.
- Loads text from source.
- Indexes text into vector store.
- Returns generated `doc_id`.

### index_text(doc_id, text, embedder, vector_store, metadata, chunks=None, chunker=None)
- Chunk text (either provided or computed).
- Embed chunks and upsert to vector store.

### _load_source_text(source_type, uri) -> str
- Supports `web`, `file`, `pdf`.
- Web uses `requests`.
- PDF uses `pypdf` if installed.

## rag/bm25.py

### BM25Result (dataclass)
- Fields: `chunk_id`, `doc_id`, `text`, `score`.

### BM25Index
- Wraps `rank_bm25` implementation.
- `__init__` tokenizes texts and builds BM25 index.
- `query(query, top_k)` returns ranked results.

## rag/bm25_store.py

### BM25Corpus (dataclass)
- Holds `chunk_ids`, `doc_ids`, `texts` lists.

### build_corpus(chunks) -> BM25Corpus
- Converts list of chunks into corpus lists.

### save_corpus(corpus, path) -> None
- Writes JSONL records to disk.

### load_corpus(path) -> BM25Corpus | None
- Reads JSONL into corpus lists.

### load_bm25_index(path) -> BM25Index | None
- Loads corpus and builds BM25Index.

## rag/retriever.py

### RetrievedChunk (dataclass)
- Fields: `chunk_id`, `doc_id`, `text`, `score`, `metadata`.

### FusionConfig (dataclass)
- Controls RRF fusion behavior.
- Fields: `use_rrf`, `rrf_k`, `vector_weight`, `bm25_weight`.

### HybridRetriever
- Combines vector retrieval with optional BM25.
- `retrieve(query, top_k, filters)`:
  - Embeds query and retrieves vector hits.
  - Optionally retrieves BM25 hits.
  - Optionally fuses results with RRF.

### _rrf_fuse(vector_hits, bm25_hits, rrf_k, vector_weight, bm25_weight)
- Computes Reciprocal Rank Fusion scores.
- Returns merged `RetrievedChunk` list.

## rag/reranker.py

### Reranker
- Optional LLM-based reranking.
- `rerank(query, chunks, top_k=None)`:
  - Builds prompt with numbered chunks.
  - Parses JSON `order` or digit list.
  - Returns reordered (and optionally trimmed) chunks.

### _parse_order(response, count) -> List[int]
- Parses JSON or digits from model output.

### _unique_in_range(indices, count)
- Filters unique indices within range.

### _compress_text(text, limit=500)
- Normalizes whitespace and truncates for prompt budget.

## rag/answer.py

### generate_answer(question, chunks, provider, max_context_words=None)
- Builds context string with optional word budget.
- Prompts model to answer using context.

### generate_answer_strict(...)
- Same as generate_answer but enforces "use only context".
- Used when verification fails.

### _build_context(chunks, max_context_words)
- Concatenates chunks until word budget is reached.

## rag/citations.py

### format_citations(chunks) -> List[dict]
- Returns list of citation dictionaries with snippet.

## rag/query_rewrite.py

### QueryRewriter
- Optional rewrite stage to improve retrieval.
- `rewrite(question)` returns original if disabled.

