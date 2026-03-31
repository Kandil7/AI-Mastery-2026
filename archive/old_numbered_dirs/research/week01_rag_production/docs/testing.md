# Testing

This section provides comprehensive information about the testing strategy, test structure, and execution procedures for the Production RAG System.

## Testing Philosophy

The Production RAG System follows a comprehensive testing approach that ensures reliability, performance, and correctness across all components. Our testing strategy includes:

- **Unit Testing**: Individual component testing
- **Integration Testing**: Component interaction testing
- **System Testing**: End-to-end functionality testing
- **Performance Testing**: Response time and throughput testing
- **Quality Testing**: Accuracy and relevance testing
- **Security Testing**: Vulnerability and security assessment
- **Load Testing**: Scalability and stress testing

## Test Structure

### Directory Structure
```
tests/
├── test_rag_system.py     # Main test suite
├── test_retrieval.py      # Retrieval-specific tests
├── test_pipeline.py       # Pipeline-specific tests
├── test_ingestion.py      # Ingestion-specific tests
├── test_api.py            # API endpoint tests
├── test_config.py         # Configuration tests
├── test_eval.py           # Evaluation tests
├── test_chunking.py       # Chunking tests
├── test_services.py       # Service layer tests
└── conftest.py            # Test configuration and fixtures
```

### Test Categories

#### 1. Unit Tests
Unit tests focus on individual functions and classes within each module. These tests:
- Validate individual component functionality
- Test edge cases and error conditions
- Ensure proper input/output behavior
- Verify internal logic and algorithms

#### 2. Integration Tests
Integration tests verify the interaction between different components. These tests:
- Validate component communication
- Test data flow between modules
- Ensure proper error handling across boundaries
- Verify configuration integration

#### 3. System Tests
System tests validate the complete application functionality. These tests:
- Test end-to-end workflows
- Validate API endpoints and responses
- Ensure proper system behavior under various conditions
- Verify integration with external services

#### 4. Performance Tests
Performance tests evaluate system performance under various loads. These tests:
- Measure response times
- Test throughput capabilities
- Validate resource utilization
- Assess scalability characteristics

#### 5. Quality Tests
Quality tests evaluate the accuracy and relevance of system outputs. These tests:
- Validate retrieval accuracy
- Test generation quality
- Assess evaluation metrics
- Verify citation accuracy

## Test Framework

The system uses the following testing tools and frameworks:

### PyTest
- Primary testing framework
- Supports parametrized tests
- Provides fixture management
- Offers comprehensive reporting

### Coverage Analysis
- `pytest-cov` for coverage measurement
- HTML reports for detailed analysis
- Line-by-line coverage tracking
- Integration with CI/CD pipelines

### Mocking
- `unittest.mock` for dependency isolation
- `pytest-mock` for pytest integration
- Mock external services and dependencies
- Control test environment

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rag_system.py

# Run specific test class
pytest tests/test_rag_system.py::TestClass

# Run specific test method
pytest tests/test_rag_system.py::TestClass::test_method
```

### Advanced Test Execution
```bash
# Run tests in parallel
pytest -n auto

# Run tests with specific markers
pytest -m "slow"  # Run slow tests only
pytest -m "not slow"  # Skip slow tests

# Run tests with specific patterns
pytest -k "test_query"  # Run tests matching pattern

# Run tests with detailed output
pytest -v -s

# Run tests and stop on first failure
pytest -x

# Run tests and show local variables on failure
pytest -l
```

### Makefile Commands
The project includes convenient Makefile commands for testing:

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test category
# (define in Makefile based on project needs)
```

## Test Organization

### Test Fixtures
Common test fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id="test_doc_1",
        content="This is a test document for the RAG system...",
        source="unittest",
        doc_type="test",
        metadata={"category": "unittest", "test_field": "value"}
    )

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return RAGConfig(
        generator_model="gpt2",
        dense_model="all-MiniLM-L6-v2",
        alpha=0.5,
        fusion="rrf",
        top_k=3,
        max_new_tokens=100
    )
```

### Test Markers
Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_document_creation():
    """Test creating a document with valid parameters."""
    # Test implementation

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_integration():
    """Test the full pipeline from indexing to querying."""
    # Test implementation

@pytest.mark.performance
def test_retrieval_performance():
    """Test retrieval performance with multiple queries."""
    # Test implementation
```

## Test Coverage

### Coverage Targets
- **Minimum Coverage**: 80% for all modules
- **Critical Components**: 90%+ coverage required
- **New Features**: 100% coverage required before merge

### Coverage Reports
Coverage reports are generated in multiple formats:
- Console output showing coverage percentage
- HTML reports with line-by-line coverage
- XML reports for CI/CD integration

## Test Examples

### Unit Test Example
```python
class TestDocumentClass:
    """Test cases for the Document class."""

    def test_document_creation(self, sample_document):
        """Test creating a document with valid parameters."""
        assert sample_document.id == "test_doc_1"
        assert "RAG system" in sample_document.content
        assert sample_document.source == "unittest"
        assert sample_document.doc_type == "test"
        assert sample_document.metadata["category"] == "unittest"

    def test_document_validation(self):
        """Test document validation with invalid parameters."""
        # Test with empty content
        with pytest.raises(ValueError):
            Document(id="test", content="")

        # Test with empty ID
        with pytest.raises(ValueError):
            Document(id="", content="test content")

    def test_validate_access_public(self, sample_document):
        """Test document access validation for public documents."""
        sample_document.access_control = {"level": "public"}
        user_perms = {"level": "public"}
        assert sample_document.validate_access(user_perms) is True
```

### Integration Test Example
```python
class TestRAGPipeline:
    """Test cases for the RAGPipeline class."""

    @pytest.fixture(autouse=True)
    def setup_mock_transformer(self):
        """Setup mock for sentence transformer."""
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer_class:
            mock_transformer_instance = Mock()
            mock_transformer_instance.encode.return_value = np.random.rand(1, 384)
            mock_transformer_class.return_value = mock_transformer_instance
            yield mock_transformer_instance

    def test_pipeline_initialization(self, mock_config):
        """Test initializing the RAG pipeline."""
        pipeline = RAGPipeline(mock_config)
        assert pipeline.config == mock_config
        assert pipeline.retriever is not None
        assert pipeline.generator is not None

    def test_pipeline_query(self, sample_documents, mock_config):
        """Test the complete query pipeline."""
        pipeline = RAGPipeline(mock_config)

        # Add sample documents
        pipeline.index(sample_documents)

        # Mock both retrieve and generate methods
        mock_retrieve_results = [
            RetrievalResult(
                document=sample_documents[0],
                score=0.9,
                rank=1
            )
        ]

        mock_generated_response = "This is a test response based on the context."

        with patch.object(pipeline, 'retrieve', return_value=mock_retrieve_results):
            with patch.object(pipeline, 'generate', return_value=mock_generated_response):
                result = pipeline.query("What is this?", top_k=1)

                assert result["query"] == "What is this?"
                assert result["response"] == mock_generated_response
                assert len(result["retrieved_documents"]) == 1
                assert result["retrieved_documents"][0]["id"] == sample_documents[0].id
```

### API Test Example
```python
class TestAPIEndpoints:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test the health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "details" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_query_endpoint(self, client):
        """Test the query endpoint."""
        query_data = {
            "query": "What is RAG?",
            "k": 3,
            "include_sources": True,
            "timeout_seconds": 30.0
        }
        
        response = client.post("/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query" in data
        assert "response" in data
        assert "sources" in data
        assert data["query"] == query_data["query"]
```

## Mocking Strategy

### External Dependencies
External dependencies are mocked to ensure test isolation and speed:

```python
# Mock sentence transformers
with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
    # Test code here

# Mock ChromaDB
with patch('chromadb.PersistentClient') as mock_chromadb:
    # Test code here

# Mock MongoDB
with patch('src.ingestion.mongo_storage.MongoStorage') as mock_mongo:
    # Test code here
```

### Configuration Mocking
Configuration values are mocked to test different scenarios:

```python
@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DATABASE__URL", "mongodb://test:test@localhost:27017")
    monkeypatch.setenv("DATABASE__NAME", "test_db")
```

## Performance Testing

### Load Testing
Performance tests validate system behavior under various loads:

```python
@pytest.mark.performance
class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_pipeline_performance_large_documents(self, mock_config):
        """Test pipeline performance with larger documents."""
        pipeline = RAGPipeline(mock_config)

        # Create a larger document
        large_content = "This is a test sentence. " * 100  # 100 sentences
        large_doc = Document(
            id="large_doc",
            content=large_content,
            source="unittest",
            doc_type="test"
        )

        # Time the indexing operation
        import time
        start_time = time.time()
        pipeline.index([large_doc])
        indexing_time = time.time() - start_time

        # Indexing should complete in a reasonable time (under 5 seconds for this test)
        assert indexing_time < 5.0

    @pytest.mark.asyncio
    async def test_retrieval_performance_multiple_queries(self, sample_documents, mock_config):
        """Test retrieval performance with multiple queries."""
        pipeline = RAGPipeline(mock_config)
        pipeline.index(sample_documents)

        # Perform multiple queries and measure performance
        import time
        start_time = time.time()

        for i in range(10):  # 10 queries
            pipeline.retrieve(f"test query {i}", top_k=2)

        total_time = time.time() - start_time
        avg_time_per_query = total_time / 10

        # Each query should complete reasonably quickly (under 1 second for this test)
        assert avg_time_per_query < 1.0
```

## Quality Testing

### Evaluation Testing
Quality tests validate the accuracy and relevance of system outputs:

```python
@pytest.mark.quality
class TestQuality:
    """Quality-related tests."""

    def test_context_recall_quality(self):
        """Test context recall quality."""
        # Implementation for context recall testing
        pass

    def test_generation_faithfulness(self):
        """Test generation faithfulness to context."""
        # Implementation for faithfulness testing
        pass

    def test_answer_relevancy(self):
        """Test answer relevancy to questions."""
        # Implementation for relevancy testing
        pass
```

## Continuous Integration

### CI Configuration
Tests are integrated into CI/CD pipelines with the following checks:

- Unit tests must pass before merge
- Coverage threshold must be maintained
- Performance benchmarks must meet criteria
- Security scans must pass

### Test Reporting
CI systems generate comprehensive test reports including:
- Test execution results
- Coverage reports
- Performance metrics
- Quality metrics

## Test Maintenance

### Regular Updates
- Update tests when functionality changes
- Add new tests for new features
- Remove obsolete tests
- Refactor tests for maintainability

### Test Data Management
- Maintain realistic test data
- Regular updates to test datasets
- Privacy-compliant test data
- Consistent test data across environments

## Best Practices

### Writing Effective Tests
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Test one thing per test
- Use appropriate assertions
- Handle edge cases
- Test error conditions

### Test Organization
- Group related tests in classes
- Use meaningful test class names
- Organize tests by functionality
- Maintain consistent naming conventions
- Separate unit and integration tests

### Performance Considerations
- Minimize test execution time
- Use appropriate test data sizes
- Mock external dependencies
- Parallelize test execution when possible
- Cache expensive operations