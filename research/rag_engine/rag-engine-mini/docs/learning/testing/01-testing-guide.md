# Testing Strategies: Complete Guide

## Table of Contents
1. [Testing Pyramid](#testing-pyramid)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [Performance Testing](#performance-testing)
5. [Security Testing](#security-testing)
6. [Test Fixtures](#test-fixtures)

---

## Testing Pyramid

### Testing Strategy

```
        /\
       /  \
      / E2E \        (Slowest, Fewest)
     /________\
    / Integration \      (Medium)
   /______________\
  /    Unit Tests   \    (Fastest, Most)
 /_____________________\
```

### Test Coverage

| Type | Purpose | Speed | Coverage |
|-------|----------|--------|
| **Unit** | Test individual functions | Fast (ms) | 80%+ |
| **Integration** | Test component interactions | Medium (s) | 60%+ |
| **E2E** | Test user workflows | Slow (min) | Critical paths |

### Arabic
**هرم الاختبار**: استراتيجية اختبار متعددة المستويات

---

## Unit Testing

### What is Unit Testing?

**Unit testing** tests individual components in isolation.

### Example with Pytest

```python
import pytest
from unittest.mock import Mock, patch

def test_embedding_generation():
    """Test embedding generation returns correct format."""
    # Arrange
    text = "Hello world"
    mock_llm = Mock()
    mock_llm.embed.return_value = [0.1, 0.2, 0.3]
    
    # Act
    result = mock_llm.embed(text)
    
    # Assert
    assert len(result) == 3
    mock_llm.embed.assert_called_once_with(text)


def test_api_request_count_metric():
    """Test API request counter increments."""
    from src.core.observability import API_REQUEST_COUNT
    
    initial = API_REQUEST_COUNT.collect()[0].samples[0].value
    
    API_REQUEST_COUNT.labels(method="POST", endpoint="/ask", status="200").inc()
    
    new_value = API_REQUEST_COUNT.collect()[0].samples[0].value
    assert new_value == initial + 1


@pytest.mark.parametrize("status,expected", [
    (200, "success"),
    (400, "client_error"),
    (500, "server_error"),
])
def test_api_status_codes(status, expected):
    """Test various HTTP status codes."""
    response = make_request(status=status)
    assert response.category == expected
```

### Fixtures

```python
@pytest.fixture
def mock_llm():
    """Fixture providing mocked LLM."""
    mock = Mock()
    mock.embed.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def sample_document():
    """Fixture providing sample document."""
    return {
        "id": "doc-123",
        "text": "This is a sample document.",
        "tenant_id": "tenant-456",
    }


def test_with_fixture(mock_llm, sample_document):
    """Test using fixtures."""
    result = mock_llm.embed(sample_document["text"])
    assert len(result) == 3
```

### Mocking

```python
# Patch external dependencies
@patch('src.adapters.llm.OpenAIClient')
def test_with_openai_mock(mock_openai):
    mock_openai.return_value.embed.return_value = [0.1, 0.2]
    
    result = embed_text("Hello")
    assert len(result) == 3


# Patch HTTP requests
@patch('requests.post')
def test_with_http_mock(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"result": "ok"}
    
    response = call_external_api()
    assert response.status_code == 200
```

---

## Integration Testing

### What is Integration Testing?

**Integration testing** tests how components work together.

### Example with Real Services

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def db_engine():
    """Database engine for integration tests."""
    engine = create_engine(
        "postgresql://postgres:postgres@localhost:5432/rag_test",
        pool_pre_ping=True,
    )
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Database session for integration tests."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.mark.integration
def test_document_crud(db_session):
    """Test document CRUD operations."""
    # Create
    doc = Document(
        id="doc-123",
        user_id="user-456",
        filename="test.pdf",
        content_type="application/pdf",
    )
    db_session.add(doc)
    db_session.commit()
    
    # Read
    retrieved = db_session.query(Document).filter_by(id="doc-123").first()
    assert retrieved is not None
    assert retrieved.filename == "test.pdf"
    
    # Update
    retrieved.filename = "updated.pdf"
    db_session.commit()
    
    # Delete
    db_session.delete(retrieved)
    db_session.commit()
    
    result = db_session.query(Document).filter_by(id="doc-123").first()
    assert result is None


@pytest.mark.integration
def test_vector_search_integration():
    """Test vector search with real Qdrant."""
    client = QdrantClient(host="localhost", port=6333)
    
    # Insert test data
    client.upsert(
        collection_name="test_collection",
        points=[Point(id=1, vector=[0.1, 0.2], payload={"text": "test"})],
    )
    
    # Search
    results = client.search(
        collection_name="test_collection",
        query_vector=[0.1, 0.2],
        limit=1,
    )
    
    assert len(results) == 1
    assert results[0].payload["text"] == "test"
```

### Testcontainers

```python
import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQL container for integration tests."""
    container = PostgresContainer("postgres:15")
    container.start()
    yield container
    container.stop()


@pytest.fixture
def postgres_url(postgres_container):
    """Get connection URL from container."""
    return postgres_container.get_connection_url()


@pytest.mark.integration
def test_with_container(postgres_url):
    """Test with ephemeral PostgreSQL container."""
    engine = create_engine(postgres_url)
    # Run tests...
```

---

## Performance Testing

### What is Performance Testing?

**Performance testing** measures system responsiveness and stability.

### Load Testing with Locust

```python
from locust import HttpUser, task, between


class RAGUser(HttpUser):
    """Simulated RAG API user."""
    wait_time = between(1, 3)
    host = "http://localhost:8000"
    
    @task(3)
    def ask_question(self):
        """Ask a question (80% of requests)."""
        self.client.post(
            "/api/v1/ask",
            json={"question": "What is RAG?"},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
    
    @task(1)
    def upload_document(self):
        """Upload a document (20% of requests)."""
        self.client.post(
            "/api/v1/documents",
            files={"file": ("test.pdf", open("test.pdf", "rb"))},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )


if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10")
```

### Performance Benchmarks

```python
import pytest
import time


@pytest.mark.benchmark
def test_embedding_performance():
    """Benchmark embedding generation."""
    start = time.time()
    
    for _ in range(100):
        embed_text("This is a test document.")
    
    duration = time.time() - start
    avg_time = duration / 100
    
    assert avg_time < 0.5  # < 500ms per embedding


@pytest.mark.benchmark
def test_vector_search_performance():
    """Benchmark vector search."""
    # Insert 10,000 vectors
    insert_test_vectors(count=10000)
    
    start = time.time()
    
    for _ in range(100):
        search_vectors(query=[0.1, 0.2], k=10)
    
    duration = time.time() - start
    avg_time = duration / 100
    
    assert avg_time < 0.1  # < 100ms per search
```

### Performance Metrics

| Metric | Target | Tool |
|---------|---------|-------|
| **P95 Latency** | < 2s | Locust, pytest-benchmark |
| **P99 Latency** | < 5s | Locust |
| **Throughput** | > 100 req/s | Locust |
| **Error Rate** | < 1% | Locust |
| **Memory** | Stable | pytest-memray |

---

## Security Testing

### SQL Injection Tests

```python
import pytest


@pytest.mark.security
def test_sql_injection_prevention():
    """Test that SQL injection is prevented."""
    malicious_input = "'; DROP TABLE users; --"
    
    response = client.post(
        "/api/v1/documents",
        json={"query": malicious_input},
    )
    
    # Should be rejected or sanitized
    assert response.status_code in [400, 403]
    assert "users" not in response.text


@pytest.mark.security
@pytest.mark.parametrize("payload", [
    "1' OR '1'='1",
    "' OR 1=1 --",
    "admin'--",
    "1' UNION SELECT * FROM users--",
])
def test_sql_injection_variants(payload):
    """Test various SQL injection payloads."""
    response = search_documents(query=payload)
    assert response.status_code != 200
```

### XSS Tests

```python
@pytest.mark.security
def test_xss_prevention():
    """Test that XSS is prevented."""
    malicious_input = "<script>alert('XSS')</script>"
    
    response = client.post(
        "/api/v1/questions",
        json={"question": malicious_input},
    )
    
    # Input should be sanitized
    assert "<script>" not in response.text


@pytest.mark.security
@pytest.mark.parametrize("payload", [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "'\"><script>alert('XSS')</script>",
])
def test_xss_variants(payload):
    """Test various XSS payloads."""
    response = submit_question(payload)
    assert "<script>" not in response.text
    assert "alert" not in response.text
```

### Rate Limiting Tests

```python
import pytest
import time


@pytest.mark.security
def test_rate_limiting():
    """Test that rate limiting works."""
    api_key = "test_key"
    
    # Make 100 requests (above limit)
    responses = []
    for i in range(100):
        response = client.get(
            "/api/v1/ask",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        responses.append(response)
    
    # Should hit rate limit
    rate_limited = any(r.status_code == 429 for r in responses)
    assert rate_limited


@pytest.mark.security
def test_api_key_validation():
    """Test API key validation."""
    # Invalid key
    response = client.get(
        "/api/v1/ask",
        headers={"Authorization": "Bearer invalid_key"},
    )
    assert response.status_code == 401
    
    # Missing key
    response = client.get("/api/v1/ask")
    assert response.status_code == 401
```

---

## Test Fixtures

### Factory Pattern

```python
import pytest
from factory import Factory, Faker, LazyAttribute


class UserFactory(Factory):
    """Factory for generating test users."""
    
    class Meta:
        model = User
    
    id = LazyAttribute(lambda _: str(uuid4()))
    email = LazyAttribute(lambda _: Faker().email())
    api_key = LazyAttribute(lambda _: f"sk_{Faker().uuid4()[:24]}")


class DocumentFactory(Factory):
    """Factory for generating test documents."""
    
    class Meta:
        model = Document
    
    id = LazyAttribute(lambda _: str(uuid4()))
    filename = LazyAttribute(lambda _: Faker().file_name())
    content_type = "application/pdf"
    status = "indexed"


def test_with_factories():
    """Test using factory-generated data."""
    user = UserFactory()
    document = DocumentFactory(user_id=user.id)
    
    assert user.id is not None
    assert document.filename is not None
```

### Data Fixtures

```python
@pytest.fixture
def sample_documents(db_session):
    """Provide sample documents for tests."""
    documents = [
        Document(id=f"doc-{i}", filename=f"file-{i}.pdf")
        for i in range(10)
    ]
    db_session.add_all(documents)
    db_session.commit()
    
    return documents


def test_with_sample_data(sample_documents):
    """Test with pre-populated data."""
    results = search_documents(query="file")
    assert len(results) == 10
```

---

## Summary

| Test Type | Purpose | Tool |
|-----------|----------|-------|
| **Unit** | Component isolation | Pytest |
| **Integration** | Component interaction | Testcontainers |
| **Performance** | Responsiveness | Locust |
| **Security** | Vulnerability detection | OWASP ZAP, manual |
| **E2E** | User workflows | Playwright |

### Key Takeaways

1. **Testing pyramid**: More unit tests, fewer E2E tests
2. **Fixtures** reduce test duplication
3. **Mocking** isolates components
4. **Performance** ensures SLA compliance
5. **Security** prevents vulnerabilities

---

## Further Reading

- [Pytest Documentation](https://docs.pytest.org/)
- [Testcontainers Documentation](https://testcontainers-python.readthedocs.io/)
- [Locust Documentation](https://docs.locust.io/)
- [Factory Boy](https://factoryboy.readthedocs.io/)
- `notebooks/learning/06-testing/testing-basics.ipynb` - Interactive notebook
