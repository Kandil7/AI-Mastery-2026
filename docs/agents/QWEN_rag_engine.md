# RAG Engine Mini - QWEN.md

## Project Overview

RAG Engine Mini is a production-grade Retrieval-Augmented Generation (RAG) starter template that bridges the gap between notebook experiments and real-world AI systems. Built with Clean Architecture principles, it provides a solid foundation for building intelligent document Q&A systems.

### Key Features
- **Hybrid Search**: Combines vector search (semantic) with keyword search (lexical) using RRF (Reciprocal Rank Fusion)
- **Cross-Encoder Reranking**: Improves precision by reranking results using a cross-encoder model
- **Multi-Tenant Design**: Complete user isolation with tenant-specific data separation
- **Async Processing**: Uses Celery for background document indexing
- **Multiple LLM Support**: Adapters for OpenAI, Ollama, Gemini, and Hugging Face
- **Observability**: Prometheus metrics and structured logging
- **Production Ready**: Includes Docker configuration, health checks, and monitoring

### Architecture
The project follows Clean Architecture with four main layers:
1. **Domain Layer**: Pure business logic with no external dependencies
2. **Application Layer**: Use cases, services, and ports (interfaces)
3. **Adapters Layer**: Concrete implementations (DB, vector store, LLM, etc.)
4. **API Layer**: FastAPI routes and controllers

## Building and Running

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key (or Ollama for local LLM)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-org/rag-engine-mini.git
cd rag-engine-mini

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (including Gradio)
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application
```bash
# Start infrastructure (Postgres + Redis + Qdrant)
make docker-up

# Run database migrations
make migrate

# Seed demo user
make seed

# Terminal 1: API Server
make run

# Terminal 2: Celery Worker
make worker

# Terminal 3: Demo UI (Optional)
make demo
```

### Verification
```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

## Development Commands

### Core Commands
- `make install` - Install dependencies
- `make run` - Run FastAPI server (dev mode)
- `make run-prod` - Run FastAPI server (production mode)
- `make worker` - Run Celery worker
- `make demo` - Start Gradio frontend demo
- `make eval` - Run retrieval evaluation script

### Testing Commands
- `make test` - Run all tests
- `make test-cov` - Run tests with coverage
- `make test-unit` - Run unit tests
- `make test-integration` - Run integration tests

### Code Quality
- `make format` - Format code with black + isort
- `make lint` - Lint code with ruff
- `make typecheck` - Type check with mypy
- `make check-all` - Run all checks

### Database Commands
- `make migrate` - Run database migrations
- `make seed` - Seed demo user
- `make migrate-rev` - Create new migration revision

### Docker Commands
- `make docker-up` - Start all services with Docker
- `make docker-down` - Stop all services
- `make docker-logs` - View Docker logs
- `make docker-build` - Build Docker image

## Project Structure

```
rag-engine-mini/
├── src/                        # Source code
│   ├── core/                   # Config, logging, DI
│   ├── domain/                 # Entities, errors
│   ├── application/            # Use cases, ports, services
│   ├── adapters/               # External implementations
│   ├── api/                    # FastAPI routes
│   └── workers/                # Celery tasks
├── tests/                      # Test suite
├── docs/                       # Documentation
├── notebooks/                  # Educational notebooks
├── scripts/                    # Utility scripts
└── docker/                     # Docker configuration
```

## Key Technologies

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: SQL toolkit and ORM
- **Qdrant**: Vector database for semantic search
- **Redis**: Caching and message broker for Celery
- **Celery**: Distributed task queue for background jobs

### AI/ML
- **OpenAI**: LLM and embeddings API
- **Sentence Transformers**: Cross-encoder models for reranking
- **Tiktoken**: Tokenization for chunking
- **Hugging Face**: Alternative LLM and embedding models

### Testing & Quality
- **pytest**: Testing framework
- **mypy**: Static type checking
- **black**: Code formatter
- **ruff**: Fast Python linter

### Infrastructure
- **Docker**: Containerization
- **PostgreSQL**: Relational database
- **Prometheus**: Metrics collection

## Development Conventions

### Code Style
- Python 3.11+ compatible
- Type hints for all functions
- 100 character line limit
- Black formatting with 100 character line length
- MyPy type checking

### Architecture Patterns
- **Clean Architecture**: Clear separation of concerns
- **Ports & Adapters**: Interface-driven design for easy testing and swapping implementations
- **Dependency Injection**: Managed through the bootstrap module
- **Domain-Driven Design**: Business logic isolated in domain layer

### Testing
- Unit tests for pure functions and services
- Integration tests for components that interact with external systems
- Test coverage using pytest-cov
- Type checking with MyPy

### Documentation
- Each function includes brief description
- Mathematical definitions where applicable
- Args and Returns sections
- Example usage

## API Endpoints

### Documents
- `POST /api/v1/documents/upload` - Upload a document for processing
- `GET /api/v1/documents/{document_id}/status` - Get document processing status

### Queries
- `POST /api/v1/queries/ask-hybrid` - Ask a question using hybrid search
- `POST /api/v1/queries/ask-hybrid-stream` - Streamed response for hybrid search
- `GET /api/v1/queries/graph-search` - Query knowledge graph for relationships

### Health & Metrics
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

## Configuration

The application uses Pydantic Settings for configuration management. Settings can be loaded from environment variables or a `.env` file.

Key configuration options include:
- Database URLs and credentials
- LLM provider settings (OpenAI, Ollama, etc.)
- Vector store configuration (Qdrant)
- Redis and Celery settings
- Reranking and retrieval parameters
- File upload limits

## Troubleshooting

### Common Issues
- **Connection refused to Qdrant/Redis/Postgres**: Ensure Docker services are running with `make docker-up`
- **Invalid API key**: Run `python scripts/seed_user.py` to create a demo user
- **No text extracted from file**: Ensure file is PDF, DOCX, or TXT format
- **CUDA out of memory**: Set `CROSS_ENCODER_DEVICE=cpu` in environment
- **Worker not processing tasks**: Ensure worker is running with correct queue

### Logging
The application uses structured logging with different levels based on environment. In development, logs are formatted for readability, while in production they're in JSON format for log aggregation systems.

## Educational Resources

The project includes extensive educational materials:
- **Notebooks**: Step-by-step guides from zero to production
- **Documentation**: Architecture patterns, developer guides, and FAQs
- **ADR Records**: Architecture decision records explaining design choices
- **Failure Modes**: Guides on debugging common RAG issues
- **Code Walkthroughs**: Detailed explanations of key components

## Contributing

Refer to `CONTRIBUTING.md` for detailed contribution guidelines. The project follows standard open-source practices with pull requests, code reviews, and issue tracking.