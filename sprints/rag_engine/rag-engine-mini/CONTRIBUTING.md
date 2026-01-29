# Contributing to RAG Engine Mini

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 Code of Conduct

Be respectful and constructive in all interactions.

## 🐛 Reporting Issues

1. Check existing issues first
2. Use the issue template
3. Include:
   - Python version
   - OS
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages

## 🔧 Development Setup

```bash
# Clone the repo
git clone <repo-url>
cd rag-engine-mini

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Start infrastructure
docker compose -f docker/docker-compose.yml up -d

# Run tests
make test
```

## 📝 Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Add bilingual docstrings where appropriate

### 3. Check Quality

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
```

### 4. Commit Messages

Follow conventional commits:

```
feat: add new feature
fix: resolve bug
docs: update documentation
test: add tests
refactor: improve code structure
chore: update dependencies
```

### 5. Submit PR

- Fill out the PR template
- Link related issues
- Request review

## 🏗️ Architecture Guidelines

### Adding New Features

1. **Domain**: Add entities/errors to `src/domain/`
2. **Port**: Define interface in `src/application/ports/`
3. **Service**: Add pure logic to `src/application/services/`
4. **Use Case**: Create orchestration in `src/application/use_cases/`
5. **Adapter**: Implement in `src/adapters/`
6. **Bootstrap**: Wire in `src/core/bootstrap.py`
7. **Tests**: Add unit and integration tests

### Code Style

- Type hints on all functions
- Docstrings with examples
- Bilingual comments where helpful
- Max line length: 88 (Black default)

## 📚 Documentation

- Update relevant docs in `docs/`
- Add examples for new features
- Update CHANGELOG.md

## 🧪 Testing

### Unit Tests
Test pure logic in isolation:
```python
def test_chunk_text():
    chunks = chunk_text_token_aware("Hello world", ChunkSpec())
    assert len(chunks) >= 1
```

### Integration Tests
Test with mocked or real services:
```python
def test_upload_endpoint(client, auth_headers):
    response = client.post("/api/v1/documents/upload", ...)
    assert response.status_code == 200
```

## 📖 License

By contributing, you agree that your contributions will be licensed under the same license as the project.
