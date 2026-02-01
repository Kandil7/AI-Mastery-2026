# RAG Engine Python SDK

[![PyPI version](https://badge.fury.io/py/rag-engine.svg)](https://badge.fury.io/py/rag-engine)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for RAG Engine Mini - A production-ready Retrieval-Augmented Generation (RAG) system.

## Features

- 🔍 **Hybrid Search** - Vector + keyword search with RRF fusion
- 📄 **Document Management** - Upload, search, delete documents
- 💬 **Chat System** - Conversational interface with context
- ⚡ **Async Support** - Full async/await support
- 🛡️ **Type Safe** - Pydantic models for all API interactions
- 🔒 **Error Handling** - Comprehensive exception hierarchy
- 📊 **Query History** - Track and manage past queries

## Installation

```bash
pip install rag-engine
```

For development:

```bash
pip install rag-engine[dev]
```

With documentation support:

```bash
pip install rag-engine[docs]
```

## Quick Start

```python
import asyncio
from rag_engine import RAGClient

async def main():
    # Initialize client
    client = RAGClient(api_key="your-api-key")
    
    # Ask a question
    answer = await client.ask("What is RAG?")
    print(answer.text)
    print(f"Sources: {answer.sources}")
    
    # Upload a document
    doc = await client.upload_document("./document.pdf")
    print(f"Uploaded: {doc.id}")

# Run the async function
asyncio.run(main())
```

## Usage

### Context Manager (Recommended)

```python
from rag_engine import RAGClient

async with RAGClient(api_key="your-api-key") as client:
    answer = await client.ask("What is machine learning?")
    print(answer.text)
    # Client automatically closes
```

### Document Operations

```python
# Upload document
doc = await client.upload_document(
    file_path="./report.pdf",
    title="Annual Report 2024"
)

# Search documents
results = await client.search_documents("machine learning")
for doc in results.documents:
    print(f"{doc.filename}: {doc.status}")

# Delete document
success = await client.delete_document("doc-id")
```

### Advanced Query Options

```python
from rag_engine import RAGClient, QueryOptions

client = RAGClient(api_key="your-api-key")

# Configure search options
options = QueryOptions(
    limit=20,
    use_hybrid_search=True,
    use_reranking=True,
    sort_by="relevance"
)

results = await client.search_documents("AI", options=options)
```

### Error Handling

```python
from rag_engine import RAGClient, AuthenticationError, RateLimitError

client = RAGClient(api_key="your-api-key")

try:
    answer = await client.ask("What is RAG?")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
except Exception as e:
    print(f"Error: {e}")
```

## API Reference

### RAGClient

Main client class for API interactions.

#### Methods

- `ask(question, k=5, use_hybrid=True, rerank=True)` - Ask a question
- `upload_document(file_path, title=None)` - Upload a document
- `search_documents(query, options=None)` - Search documents
- `delete_document(document_id)` - Delete a document
- `get_query_history(limit=20, offset=0)` - Get query history

#### Configuration

```python
client = RAGClient(
    api_key="your-api-key",
    base_url="http://localhost:8000",  # Default
    timeout=30.0  # Request timeout
)
```

### Models

- `Answer` - RAG answer with sources and metadata
- `Document` - Document metadata
- `QueryHistoryItem` - Query history entry
- `QueryOptions` - Search query options
- `SearchResult` - Document search results

### Exceptions

- `RAGEngineError` - Base exception
- `AuthenticationError` - Invalid API key (401)
- `RateLimitError` - Rate limit exceeded (429)
- `ValidationError` - Invalid request (422)
- `ServerError` - Server error (5xx)
- `NotFoundError` - Resource not found (404)

## Examples

See the [examples/](examples/) directory for more detailed examples:

- `basic_usage.py` - Basic SDK usage
- `document_upload.py` - Document management
- `chat_session.py` - Chat functionality
- `error_handling.py` - Error handling patterns

## Development

### Setup

```bash
git clone https://github.com/your-org/rag-engine-mini.git
cd sdk/python
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Formatting
black rag_engine/

# Linting
ruff check rag_engine/

# Type checking
mypy rag_engine/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit (`git commit -m 'feat: add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: https://rag-engine.readthedocs.io/
- Issues: https://github.com/your-org/rag-engine-mini/issues
- Email: team@ragengine.ai

## Changelog

### 1.0.0 (2024-02-01)

- Initial release
- Full API support
- Async/await support
- Pydantic models
- Comprehensive error handling
