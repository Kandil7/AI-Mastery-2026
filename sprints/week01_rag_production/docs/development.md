# Development

This section provides comprehensive guidance for developing with the Production RAG System, including setup, coding standards, testing, and best practices.

## Development Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git version control system
- Docker and Docker Compose (for containerized development)
- A modern code editor or IDE (VS Code, PyCharm, etc.)

### Initial Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-organization/ai-mastery-2026.git
cd ai-mastery-2026/sprints/week01_rag_production

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Install development dependencies
pip install black isort flake8 mypy pytest pytest-cov pytest-asyncio

# 7. Install pre-commit hooks (if available)
pip install pre-commit
pre-commit install
```

## Project Structure

Understanding the project structure is essential for effective development:

```
sprints/week01_rag_production/
├── api.py                 # FastAPI application
├── ui.py                  # Streamlit dashboard
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose orchestration
├── IMPLEMENTATION.md      # Implementation guide
├── Makefile               # Build automation
├── plan.md                # Planning document
├── PRODUCTION_ARCHITECTURE.md # Architecture documentation
├── README.md              # Project overview
├── requirements.txt       # Python dependencies
├── stress_test.py         # Stress testing utilities
├── nginx.conf             # Nginx configuration
├── ssl/                   # SSL certificates directory
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration management
│   ├── pipeline.py        # RAG pipeline
│   ├── chunking/          # Text chunking strategies
│   ├── eval/              # Evaluation module
│   ├── ingestion/         # Document ingestion pipeline
│   ├── observability/     # Monitoring and logging
│   ├── retrieval/         # Retrieval system
│   └── services/          # Business logic services
├── tests/                 # Test suite
├── docs/                  # Documentation
└── .gitignore            # Git ignore patterns
```

## Development Workflow

### 1. Branch Strategy
Follow the GitFlow branching model:

- `main` - Production-ready code
- `develop` - Latest development code
- `feature/*` - Feature branches
- `release/*` - Release preparation
- `hotfix/*` - Urgent fixes

```bash
# Create a feature branch
git checkout -b feature/new-feature-name

# Create a hotfix branch
git checkout -b hotfix/critical-fix
```

### 2. Coding Standards

#### Python Standards
- **Style**: Follow PEP 8 guidelines
- **Formatting**: Use Black with 100 character line length
- **Import Sorting**: Use isort for consistent import ordering
- **Type Hints**: All functions must have type hints
- **Documentation**: All public functions must have docstrings

#### Naming Conventions
- **Classes**: PascalCase (e.g., `RAGPipeline`)
- **Functions**: snake_case (e.g., `process_query`)
- **Constants**: UPPER_CASE (e.g., `MAX_TOKENS`)
- **Variables**: snake_case (e.g., `query_result`)

#### Code Formatting
```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Or use the Makefile command
make format
```

#### Type Checking
```bash
# Run mypy for type checking
mypy src/

# Or use the Makefile command
make lint
```

#### Linting
```bash
# Run flake8 for linting
flake8 src/ tests/

# Or use the Makefile command
make lint
```

### 3. Documentation Standards

Each function should include:
1. Brief description
2. Mathematical definition (using Unicode symbols where applicable)
3. Args and Returns sections
4. Example usage

```python
def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    The cosine similarity is calculated as:
    cos(θ) = (A · B) / (||A|| × ||B||)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity value between -1 and 1

    Example:
        >>> vec1 = [1, 0, 1]
        >>> vec2 = [0, 1, 1]
        >>> calculate_similarity(vec1, vec2)
        0.5
    """
    # Implementation here
```

### 4. Testing Standards

#### Test Structure
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance tests for critical paths
- Quality tests for accuracy validation

#### Test Organization
```python
# tests/test_component.py
import pytest
from unittest.mock import Mock, patch

class TestComponentClass:
    """Test cases for ComponentClass."""

    def test_method_behavior(self):
        """Test specific method behavior."""
        # Test implementation

    @pytest.mark.integration
    def test_integration_with_other_component(self):
        """Test integration with other component."""
        # Test implementation
```

## Development Tools

### Makefile Commands
The project includes a Makefile with common development commands:

```bash
# Install dependencies
make install

# Run the API server
make run-api

# Run tests
make test

# Run tests with coverage
make test-cov

# Run linting
make lint

# Format code
make format

# Clean cache directories
make clean
```

### IDE Configuration

#### VS Code Settings
Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.linting.mypyEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### Recommended Extensions
- Python (Microsoft)
- Pylance (Microsoft)
- Black Formatter
- isort
- GitLens
- Jupyter

## Development Best Practices

### 1. Code Organization
- Keep modules focused on single responsibilities
- Use clear and descriptive names
- Group related functionality in packages
- Maintain consistent code structure

### 2. Error Handling
- Implement comprehensive error handling
- Provide meaningful error messages
- Log errors appropriately
- Gracefully handle edge cases

### 3. Performance Considerations
- Optimize critical paths
- Use efficient data structures
- Implement caching where appropriate
- Monitor resource usage

### 4. Security Practices
- Validate and sanitize all inputs
- Protect against injection attacks
- Use secure coding practices
- Implement proper authentication

## Feature Development

### Adding New Features

1. **Planning Phase**
   - Define feature requirements
   - Design API endpoints if needed
   - Plan data models and schemas
   - Consider backward compatibility

2. **Implementation Phase**
   - Create feature branch
   - Implement core functionality
   - Add comprehensive tests
   - Update documentation
   - Perform code review

3. **Testing Phase**
   - Run all tests
   - Add feature-specific tests
   - Perform integration testing
   - Validate performance impact

4. **Documentation Phase**
   - Update API documentation
   - Add usage examples
   - Document configuration options
   - Update architecture diagrams

### Example Feature Implementation

Let's walk through implementing a new retrieval strategy:

```python
# src/retrieval/retrieval.py
from enum import Enum

class RetrievalStrategy(Enum):
    """Enumeration for different retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    NEW_STRATEGY = "new_strategy"  # Add new strategy

class NewStrategyRetriever:
    """Implementation of the new retrieval strategy."""
    
    def __init__(self, config_param: float = 1.0):
        self.config_param = config_param
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents using the new strategy."""
        # Implementation here
        pass

# Update HybridRetriever to include the new strategy
class HybridRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        # Add logic to handle the new strategy
        if self.strategy == RetrievalStrategy.NEW_STRATEGY:
            # Use NewStrategyRetriever
            pass
```

## Testing During Development

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py

# Run tests with specific marker
pytest -m "unit"

# Run tests in verbose mode
pytest -v
```

### Test-Driven Development (TDD)
1. Write a failing test for the new functionality
2. Implement the minimum code to pass the test
3. Refactor the code while keeping tests passing
4. Repeat for additional functionality

### Mocking External Dependencies
```python
from unittest.mock import Mock, patch
import pytest

@patch('src.external_service.ExternalAPI')
def test_function_with_external_dependency(mock_api):
    """Test function that uses external API."""
    mock_api.return_value.get_data.return_value = {'result': 'success'}
    
    # Test implementation
    result = function_being_tested()
    
    assert result == 'expected_result'
    mock_api.return_value.get_data.assert_called_once()
```

## Debugging

### Debugging Techniques
- Use logging extensively
- Add debug endpoints for development
- Use debugger for complex issues
- Implement circuit breakers for resilience

### Logging Configuration
```python
import logging

logger = logging.getLogger(__name__)

def my_function(param: str) -> str:
    logger.debug(f"Processing parameter: {param}")
    
    try:
        result = complex_operation(param)
        logger.info(f"Operation completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in my_function: {e}", exc_info=True)
        raise
```

### Debug Endpoints (Development Only)
```python
from fastapi import FastAPI
import os

app = FastAPI()

if os.getenv("ENVIRONMENT") == "development":
    @app.get("/debug/config")
    async def debug_config():
        """Debug endpoint to view current configuration."""
        from src.config import settings
        return settings.dict()
```

## Performance Optimization

### Profiling
```python
import cProfile
import pstats

def profile_function():
    """Profile a specific function."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Call the function to profile
    result = function_to_profile()
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
    
    return result
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(input_data: str) -> str:
    """Cache expensive computations."""
    # Implementation here
    pass
```

## Code Review Process

### Checklist for Code Reviews
- [ ] Code follows established patterns and conventions
- [ ] Sufficient test coverage
- [ ] Proper error handling
- [ ] Security considerations addressed
- [ ] Performance implications considered
- [ ] Documentation updated
- [ ] Backward compatibility maintained
- [ ] Configuration options documented

### Review Guidelines
- Be constructive and respectful
- Focus on code quality and functionality
- Consider maintainability and readability
- Verify test coverage
- Check for potential bugs

## Continuous Integration

### Local CI Simulation
```bash
# Run the same checks as CI
make lint
make format
pytest --cov=src
```

### Pre-commit Hooks
Set up pre-commit hooks to run checks before committing:

```bash
pip install pre-commit
pre-commit install

# Example .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.10
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

## Dependency Management

### Managing Dependencies
```bash
# Add new dependency
pip install new-package
pip freeze > requirements.txt

# Update dependencies
pip list --outdated
pip install --upgrade package-name

# Use pip-tools for better dependency management
pip install pip-tools
pip-compile requirements.in  # Generate requirements.txt from requirements.in
```

### Virtual Environment Management
```bash
# Create new virtual environment
python -m venv new_env

# Activate environment
source new_env/bin/activate  # Linux/macOS
new_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Documentation Updates

### Updating Documentation
- Update docstrings when changing functionality
- Update README.md for major changes
- Update architecture documentation
- Add usage examples for new features

### API Documentation
- Update API endpoint documentation
- Add request/response examples
- Document new parameters
- Update error response documentation

## Deployment Preparation

### Pre-deployment Checklist
- [ ] All tests pass
- [ ] Code coverage meets requirements
- [ ] Security scan passes
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Configuration validated
- [ ] Rollback plan prepared

### Environment Validation
```python
from src.config import settings

def validate_configuration():
    """Validate configuration before deployment."""
    errors = []
    
    # Validate database connection
    try:
        # Test database connection
        pass
    except Exception as e:
        errors.append(f"Database connection failed: {e}")
    
    # Validate model availability
    try:
        # Test model loading
        pass
    except Exception as e:
        errors.append(f"Model validation failed: {e}")
    
    if errors:
        raise ConfigurationValidationError("\n".join(errors))
```

This development guide provides a comprehensive foundation for contributing to the Production RAG System. Following these practices will ensure code quality, maintainability, and consistency across the project.