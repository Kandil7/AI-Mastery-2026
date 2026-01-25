# Core package (core/*.py)

## core/settings.py

### Settings (dataclass)
- Purpose: central configuration container loaded from YAML.
- Fields:
  - `app_version: str`
  - `default_provider: str`
  - `default_vector_store: str`
  - `config_path: Path`
  - `raw: Dict[str, Any]` -> raw config map.

### _load_yaml(path: Path) -> Dict[str, Any]
- Loads YAML from disk and returns a dictionary.
- Returns `{}` if file does not exist.

### load_settings() -> Settings
- Determines config path from `WEEK5_BACKEND_CONFIG` or default.
- Loads YAML using `_load_yaml`.
- Builds and returns a `Settings` object with defaults.

## core/factories.py

### _provider_config(settings, name) -> Dict[str, Any]
- Returns provider configuration block for a given provider name.

### _vector_config(settings, name) -> Dict[str, Any]
- Returns vector store configuration block for a given store name.

### create_llm_provider(settings, name=None) -> LLMProvider
- Creates LLM provider based on settings:
  - `openai` -> `OpenAIProvider`
  - `anthropic` -> `AnthropicProvider`
  - `local_vllm` -> `LocalVLLMProvider`
- Reads API keys from config or environment.
- Raises `ValueError` for unsupported provider.

### create_embeddings_provider(settings, name=None) -> EmbeddingsProvider
- Creates embeddings provider based on settings:
  - `openai` -> `OpenAIEmbeddings`
  - `local_vllm` -> `LocalEmbeddings`
- Reads API keys from config or environment.

### create_vector_store(settings, name=None) -> VectorStore
- Creates vector store backend:
  - `pgvector` -> `PgVectorStore`
  - `qdrant` -> `QdrantStore`
  - `weaviate` -> `WeaviateStore`
- Raises `ValueError` for unsupported store.

### create_bm25_index(settings)
- Resolves BM25 corpus path.
- Loads BM25 index using `load_bm25_index`.

### create_routing_policy(settings) -> RoutingPolicy
- Builds `RoutingPolicy` from settings.

## core/logging.py

### configure_logging(level=logging.INFO) -> None
- Configures root logging with a default format.

## core/telemetry.py

### Trace (dataclass)
- Fields:
  - `trace_id: str`
  - `attributes: Dict[str, str]`

### start_trace(trace_id, attributes=None) -> Trace
- Returns a `Trace` object with attributes defaulted to `{}`.
