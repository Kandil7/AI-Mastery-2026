# Providers package (providers/*.py)

## providers/llm_base.py

### LLMProvider (Protocol)
- Defines a single method: `generate(prompt: str) -> str`.
- Implemented by LLM provider classes.

## providers/embeddings_provider.py

### EmbeddingsProvider (Protocol)
- Defines `embed(texts: List[str]) -> List[List[float]]`.
- Implemented by embedding provider classes.

## providers/openai_provider.py

### OpenAIProvider
- Purpose: wraps OpenAI chat completions for LLM generation.
- `__init__(model, api_key=None, base_url=None)` stores config.
- `generate(prompt: str) -> str`:
  - Creates `OpenAI` client.
  - Calls `chat.completions.create` with temperature 0.2.
  - Returns first message content.

## providers/openai_embeddings.py

### OpenAIEmbeddings
- Purpose: wraps OpenAI embeddings API.
- `__init__(model, api_key=None, base_url=None)` stores config.
- `embed(texts)`:
  - Calls `client.embeddings.create`.
  - Returns embedding vectors.

## providers/anthropic_provider.py

### AnthropicProvider
- Purpose: wraps Anthropic messages API for LLM generation.
- `__init__(model, api_key=None)` stores config.
- `generate(prompt)`:
  - Creates `Anthropic` client.
  - Calls `messages.create` with max_tokens and temperature.
  - Returns text from the first content block.

## providers/local_vllm_provider.py

### LocalVLLMProvider
- Purpose: uses OpenAI-compatible API to talk to local vLLM.
- `__init__(model, base_url, api_key=None)` stores config.
- `generate(prompt)`:
  - Uses OpenAI client with local base URL.
  - Calls chat completions and returns content.

## providers/local_embeddings.py

### LocalEmbeddings
- Purpose: uses OpenAI-compatible embeddings API for local vLLM.
- `__init__(model, base_url, api_key=None)` stores config.
- `embed(texts)`:
  - Calls embeddings endpoint and returns vectors.
