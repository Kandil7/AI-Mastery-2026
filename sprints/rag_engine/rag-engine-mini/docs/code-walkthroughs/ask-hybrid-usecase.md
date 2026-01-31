# 🚶‍♂️ Code Walkthrough: The Hybrid Search Flow

## 🗺️ The Path of a Query

This guide follows a user's question from API to Answer, showing how the hybrid search system works in this RAG implementation.

## 🏗️ Architecture Overview

```
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   User      │    │   API Layer          │    │ Application     │
│   Query     │───▶│  (FastAPI)           │───▶│   Services      │
└─────────────┘    └──────────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐    └─────────┼─────────┐
│  Infrastructure │    │   Domain        │              │         │
│   Layers       │◀───┤   Models        │◀─────────────┼─────────┤
│                │    │                 │              │         │
│ • Vector DB    │    │ • Entities      │              │         │
│ • Keyword DB   │    │ • Value Objects │              │         │
│ • Cache        │    │ • Business Rules│              │         │
└─────────────────┘    └─────────────────┘              │         │
                                                        │         │
                                        ┌───────────────▼─────────┴────────┐
                                        │        Core Logic                  │
                                        │                                  │
                                        │ • Chunking                       │
                                        │ • Embeddings                     │
                                        │ • Fusion (RRF)                   │
                                        │ • Prompt Building                │
                                        └──────────────────────────────────┘
```

## 🧭 Step-by-Step Flow

### 1. API Request (`src/api/v1/routes_queries.py`)

When a user submits a query, it hits the `/ask/hybrid` endpoint:

```python
@router.post("/ask/hybrid")
async def ask_hybrid(
    request: AskHybridRequest,
    use_case: AskQuestionHybridUseCase = Depends(get_ask_hybrid_usecase)
):
    result = await use_case.execute(request)
    return result
```

**Why this design?**
- Clean separation between HTTP concerns and business logic
- Dependency injection allows easy testing and configuration

### 2. Use Case Orchestration (`src/application/use_cases/ask_question_hybrid.py`)

The `AskQuestionHybridUseCase` coordinates the entire process:

```python
class AskQuestionHybridUseCase:
    async def execute(self, request: AskHybridRequest) -> AskResponse:
        # Step 1: Validate and preprocess
        # Step 2: Generate embeddings for the query
        # Step 3: Perform vector search
        # Step 4: Perform keyword search  
        # Step 5: Fuse results using RRF
        # Step 6: Apply reranking if enabled
        # Step 7: Build context and prompt
        # Step 8: Generate response with LLM
        # Step 9: Return formatted response
```

**Why this design?**
- Single responsibility: orchestrates the entire flow
- Easy to test: each step can be mocked independently
- Configurable: can enable/disable components (reranking, keyword search)

### 3. Dual Retrieval Process

#### 3.1 Vector Search (`src/adapters/vector/qdrant_store.py`)
```python
async def search(self, query_embedding: List[float], top_k: int) -> List[ScoredChunk]:
    # Convert embedding to vector search in Qdrant
    # Return top-k most similar chunks
```

#### 3.2 Keyword Search (`src/adapters/persistence/postgres/keyword_store.py`)
```python
async def search(self, query_text: str, top_k: int) -> List[ScoredChunk]:
    # Use PostgreSQL full-text search (FTS) 
    # Return top-k most relevant chunks
```

**Why both?**
- **Vector search**: Captures semantic meaning and synonyms
- **Keyword search**: Handles exact matches and named entities
- **Together**: Better recall and precision than either alone

### 4. Result Fusion (`src/application/services/fusion.py`)

Results from both searches are combined using Reciprocal Rank Fusion (RRF):

```python
def rrf_fusion(
    vector_results: List[ScoredChunk], 
    keyword_results: List[ScoredChunk], 
    k: int = 60
) -> List[ScoredChunk]:
    # Assign ranks to each result from both sources
    # Apply RRF formula: score = sum(1/(k + rank))
    # Return re-ranked results
```

**Why RRF?**
- Parameter-free: no need to tune weights
- Robust: works well regardless of score distributions
- Theoretically sound: based on probability theory

### 5. Re-ranking (Optional) (`src/adapters/rerank/cross_encoder.py`)

If enabled, results are re-ranked using a cross-encoder model:

```python
async def rerank(self, query: str, results: List[ScoredChunk]) -> List[ScoredChunk]:
    # Create query-document pairs
    # Score each pair using cross-encoder
    # Re-sort results by cross-encoder scores
```

**Why cross-encoder?**
- More accurate than simple similarity scores
- Faster than LLM-based re-ranking
- Good balance of quality and performance

### 6. Context Building (`src/application/services/prompt_builder.py`)

Selected chunks are used to build the context for the LLM:

```python
def build_rag_prompt(
    query: str, 
    retrieved_chunks: List[ScoredChunk], 
    max_context_length: int
) -> str:
    # Combine query with retrieved context
    # Truncate if needed to fit context window
    # Apply prompt engineering best practices
```

**Why careful context building?**
- LLM context windows are limited
- Irrelevant context can hurt performance
- Proper formatting improves LLM performance

### 7. Response Generation

Finally, the LLM generates a response based on the context:

```python
response = await llm.generate(prompt)
```

## 🎯 Key Design Decisions

### 1. Clean Architecture
- **Ports/Adapters**: Interfaces abstract external dependencies
- **Dependency Injection**: Makes testing and configuration easier
- **Single Responsibility**: Each component has one job

### 2. Configurability
- Enable/disable components (keyword search, re-ranking)
- Adjust parameters (top-k, chunk size, overlap)
- Swap implementations (different LLMs, embedding models)

### 3. Observability
- Structured logging throughout the pipeline
- Performance metrics for each step
- Error handling and graceful degradation

## 🧪 Debugging Tips

1. **Low recall?** Check if keyword search is enabled
2. **High latency?** Consider disabling re-ranking or reducing top-k
3. **Irrelevant results?** Adjust fusion parameters or re-ranking depth
4. **Hallucinations?** Verify grounding in retrieved context

## 📚 Further Exploration

- `src/core/config.py` - All configurable parameters
- `src/core/bootstrap.py` - How components are wired together
- `src/application/services/chunking.py` - How documents are prepared
- `src/application/services/embedding_cache.py` - How embeddings are cached