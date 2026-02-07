# Extension Development Guide: Building on RAG Engine Mini

## 🎯 Overview

This guide explains how to extend the RAG Engine Mini while maintaining its educational quality and production-ready standards. It covers architectural considerations, implementation patterns, and documentation practices that preserve the system's pedagogical value.

## 🏗️ Understanding the Extension Points

### Primary Extension Areas

The RAG Engine Mini is designed with several key extension points:

1. **Adapters Layer**: Add new LLM providers, vector stores, or file storage backends
2. **Application Services**: Extend core functionality with new use cases
3. **API Layer**: Add new endpoints and data transfer objects
4. **Domain Layer**: Introduce new entities or value objects (with care)
5. **Workers Layer**: Add background processing capabilities

### Architecture-First Approach

Before implementing any extension, consider how it fits into the existing architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   New Feature   │←---│  Application     │←---│    Domain       │
│   API Endpoint  │    │  Service/UseCase │    │    Entities     │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          │               ┌──────▼────────┐              │
          │               │  Dependency   │              │
          │               │   Injection   │              │
          │               │   Container   │              │
          │               └──────┬────────┘              │
          │                      │                       │
          │               ┌──────▼────────┐              │
          │               │  Application  │              │
          │               │   Services    │              │
          │               └──────┬────────┘              │
          │                      │                       │
          │               ┌──────▼────────┐              │
          │               │    Ports &    │              │
          │               │   Adapters    │              │
          │               └──────┬────────┘              │
          │                      │                       │
          │        ┌─────────────┴─────────────┐         │
          │        │                           │         │
          ▼        ▼                           ▼         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ New Integration │    │ Extended DB      │    │ New Queue       │
│   Component     │    │    Schema       │    │   Handler       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🧩 Adding New Adapters

### Example: Adding a New LLM Provider

1. **Define the Port Interface** (in [src/application/ports](../../../src/application/ports/)):
   ```python
   # If not already defined
   class LLMProviderPort(Protocol):
       async def generate(self, prompt: str) -> str: ...
       async def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
   ```

2. **Create the Adapter** (in [src/adapters/llm/](../../../src/adapters/llm/)):
   ```python
   class NewProviderLLMAdapter(LLMProviderPort):
       def __init__(self, api_key: str, model: str):
           self.api_key = api_key
           self.model = model
       
       async def generate(self, prompt: str) -> str:
           # Implementation specific to new provider
           pass
   ```

3. **Register in Dependency Container** (in [src/core/bootstrap.py](../../../src/core/bootstrap.py)):
   ```python
   def get_container():
       container = DIContainer()
       # Register your new adapter
       if settings.LLM_PROVIDER == "new_provider":
           container.register_singleton(LLMProviderPort, NewProviderLLMAdapter)
       return container
   ```

4. **Add Tests** (in [tests/unit/](../../../tests/unit/)):
   ```python
   def test_new_provider_adapter():
       # Test the new adapter specifically
       pass
   ```

5. **Document the Extension**:
   - Add an ADR explaining why you chose this provider
   - Update configuration documentation
   - Add usage examples to the README

### Educational Value Preservation

When adding adapters:
- Include detailed comments explaining the provider-specific nuances
- Add error handling examples specific to the provider
- Document rate limits and best practices
- Provide performance comparisons with existing providers

## 🔧 Extending Application Services

### Adding New Use Cases

Follow these steps to add new functionality:

1. **Define the Use Case** (in [src/application/use_cases/](../../../src/application/use_cases/)):
   ```python
   @dataclass
   class NewUseCaseRequest:
       param1: str
       param2: int
   
   @dataclass
   class NewUseCaseResponse:
       result: str
       metadata: Dict[str, Any]
   
   class NewUseCase:
       def __init__(self, 
                    repo_port: RepositoryPort,
                    llm_port: LLMProviderPort):
           self._repo_port = repo_port
           self._llm_port = llm_port
       
       async def execute(self, request: NewUseCaseRequest) -> NewUseCaseResponse:
           # Business logic goes here
           pass
   ```

2. **Update the Container** (in [src/core/bootstrap.py](../../../src/core/bootstrap.py)):
   ```python
   container.register_transient(NewUseCase, NewUseCase)
   ```

3. **Create API Endpoint** (in [src/api/v1/](../../../src/api/v1/)):
   ```python
   @router.post("/new-feature", response_model=NewUseCaseResponse)
   async def new_feature_endpoint(
       request: NewUseCaseRequest,
       use_case: NewUseCase = Depends(get_new_use_case)
   ):
       return await use_case.execute(request)
   ```

4. **Add Educational Documentation**:
   - Create a notebook demonstrating the new feature
   - Update the relevant layer guide with your changes
   - Add an exercise in the [exercises](../exercises/) directory

## 📚 Maintaining Educational Quality

### Documentation Standards

Every extension should include:

1. **Code Comments**: Explain complex logic and decisions
2. **Type Hints**: Maintain strict typing throughout
3. **Docstrings**: Describe classes, methods, and parameters
4. **Examples**: Show how to use new functionality
5. **Error Handling**: Document potential exceptions

### Educational Content Creation

For each significant extension, create corresponding educational materials:

1. **Architecture Decision Record (ADR)**: Document design decisions
2. **Jupyter Notebook**: Interactive demonstration
3. **Markdown Guide**: Detailed explanation
4. **Exercise**: Practical challenge for learners
5. **Test Cases**: Examples of proper usage

### Example: Adding Semantic Search Enhancement

1. **Create the ADR** ([docs/adr/006-semantic-search-enhancement.md](../../adr/006-semantic-search-enhancement.md)):
   ```
   # ADR 006: Enhanced Semantic Search with Cross-Encoder

   ## Context
   Our current semantic search relies solely on vector similarity...
   ```

2. **Implement the Feature**:
   ```python
   # In src/application/services/semantic_search.py
   class EnhancedSemanticSearchService:
       """Educational implementation of cross-encoder reranking"""
       
       def __init__(self, 
                    vector_adapter: VectorAdapterPort,
                    cross_encoder: CrossEncoderPort):
           self._vector_adapter = vector_adapter
           self._cross_encoder = cross_encoder
       
       async def search(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
           """Demonstrates two-stage retrieval and reranking process"""
           # Stage 1: Retrieve candidates using vector search
           candidates = await self._vector_adapter.search(query, top_k * 2)
           
           # Stage 2: Rerank using cross-encoder
           reranked = await self._cross_encoder.rerank(query, candidates, top_k)
           
           return reranked
   ```

3. **Create Educational Notebook** ([notebooks/learning/semantic_enhancement_demo.ipynb](../../../notebooks/learning/semantic_enhancement_demo.ipynb)):
   ```python
   # Notebook demonstrating the enhancement
   # With visualizations and explanations
   ```

4. **Update Relevant Guides**: Add section to appropriate layer guide

## 🧪 Testing Your Extensions

### Test Organization

Organize tests according to the architecture:

- **Unit Tests** ([tests/unit/](../../../tests/unit/)): Test individual components in isolation
- **Integration Tests** ([tests/integration/](../../../tests/integration/)): Test component interactions
- **End-to-End Tests** ([tests/e2e/](../../../tests/e2e/)): Test complete workflows

### Example Test Structure

```python
# tests/unit/test_new_feature.py
import pytest
from unittest.mock import Mock, AsyncMock

from src.application.use_cases.new_use_case import NewUseCase, NewUseCaseRequest

@pytest.mark.asyncio
class TestNewUseCase:
    async def test_execute_success(self):
        # Arrange
        mock_repo = Mock()
        mock_llm = Mock()
        use_case = NewUseCase(mock_repo, mock_llm)
        
        request = NewUseCaseRequest(param1="test", param2=42)
        
        # Act
        result = await use_case.execute(request)
        
        # Assert
        assert result is not None
        # Additional assertions...
    
    async def test_execute_handles_error_gracefully(self):
        # Test error handling
        pass
```

## 🔍 Code Quality Standards

### Static Analysis

Ensure your extensions pass all static analysis:

- **Type Checking**: Use mypy to verify type annotations
- **Linting**: Follow flake8 and black formatting standards
- **Security**: Use bandit for security vulnerability scanning

### Performance Considerations

When extending the system:

1. **Benchmark Impact**: Measure performance before and after changes
2. **Memory Usage**: Consider memory implications of new features
3. **Async Patterns**: Maintain async/await patterns consistently
4. **Caching Opportunities**: Identify and implement appropriate caching

## 📖 Documentation Requirements

### Required Documentation for Each Extension

1. **Inline Documentation**:
   - Docstrings for all public methods and classes
   - Complex algorithms explained with comments
   - Type hints for all parameters and return values

2. **External Documentation**:
   - API documentation with request/response examples
   - Configuration options documentation
   - Performance characteristics
   - Integration guides

3. **Educational Content**:
   - Jupyter notebook demonstrating usage
   - Exercise for learners to practice
   - Comparison with alternative approaches

### Example Educational Notebook Section

```python
# In your educational notebook
"""
## Understanding Cross-Encoder Reranking

In this section, we'll explore how cross-encoder reranking improves search results.

Traditional vector search computes embeddings for query and documents separately:
- Query embedding: encode(query)
- Document embeddings: encode(doc_1), encode(doc_2), ...
- Similarity: cosine_similarity(query_emb, doc_emb)

Cross-encoders, however, consider query and document together:
- Combined representation: cross_encode([query, document])
- More nuanced understanding of relevance

Let's see this in action:
"""

import numpy as np
from rag_engine.cross_encoder import CrossEncoderService

# Initialize our enhanced search
enhanced_search = CrossEncoderService(model_name="cross-encoder-msmarco")

# Original vector results vs. reranked results
original_results = vector_search.search("machine learning applications", top_k=10)
reranked_results = await enhanced_search.rerank("machine learning applications", original_results)

# Visualize the difference in rankings
plot_comparison(original_results, reranked_results)
```

## 🚀 Advanced Extension Patterns

### Plugin Architecture

For complex extensions, consider implementing a plugin-like pattern:

```python
# src/extensions/base.py
from abc import ABC, abstractmethod
from typing import Protocol

class ExtensionPlugin(ABC):
    """Base class for educational extensions"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the extension"""
        pass
    
    @abstractmethod
    async def initialize(self):
        """Setup logic for the extension"""
        pass
    
    @abstractmethod
    def register_components(self, container: DIContainer):
        """Register extension components in the DI container"""
        pass
```

### Feature Flags

Use feature flags to enable/disable experimental extensions:

```python
# In configuration
NEW_FEATURE_ENABLED = os.getenv("NEW_FEATURE_ENABLED", "false").lower() == "true"

# In service
class ConditionalService:
    async def execute(self, request):
        if settings.NEW_FEATURE_ENABLED:
            return await self._new_approach(request)
        else:
            return await self._traditional_approach(request)
```

## ✅ Pre-Submission Checklist

Before submitting your extension:

- [ ] All tests pass (unit, integration, e2e)
- [ ] Type checking passes (`mypy src/`)
- [ ] Code is formatted (`black src/`)
- [ ] New functionality is documented
- [ ] Educational materials are created
- [ ] Performance impact is measured
- [ ] Security implications are considered
- [ ] Code follows SOLID principles
- [ ] Dependency injection is properly used
- [ ] Error handling is comprehensive
- [ ] Configuration options are documented

## 🎓 Contributing Back to the Community

Once your extension is complete:

1. **Submit a Pull Request** with your changes
2. **Share your learnings** in the community forums
3. **Create a case study** demonstrating your solution
4. **Mentor others** who want to implement similar features

Following these guidelines will ensure your extensions maintain the educational quality that makes RAG Engine Mini valuable for learning while adding meaningful functionality to the system.