# AI-Mastery-2026: From Math to Production

## Project Overview

AI-Mastery-2026 is a comprehensive AI Engineer Toolkit built from first principles following the **White-Box Approach**. The project focuses on understanding the mathematical foundations before using abstractions, implementing algorithms from scratch with NumPy, and considering production aspects throughout the development process.

The project follows a structured philosophy:
1. **Math First** → Derive equations, understand foundations
2. **Code Second** → Implement from scratch with NumPy
3. **Libraries Third** → Use sklearn/PyTorch knowing what happens underneath
4. **Production Always** → Every concept includes deployment considerations

### Architecture

The project is organized into several key directories:
- `src/` - Core source code with mathematical foundations, ML algorithms, production components, and LLM engineering
- `research/` - Jupyter notebooks for the 17-week learning program
- `tests/` - Unit tests for all modules
- `docs/` - Technical documentation
- `scripts/` - Automation tools
- `benchmarks/` - Performance and cost optimization studies
- `case_studies/` - Real-world applications

## Building and Running

### Prerequisites
- Python 3.10+
- pip
- Docker (optional, for containerized deployment)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# Run setup script to create virtual environment and install dependencies
bash setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Available Commands

The project uses a Makefile for build automation:

```bash
# Install dependencies
make install

# Run tests
make test

# Run tests with coverage
make test-cov

# Run linting and type checking
make lint

# Format code
make format

# Build Docker images
make docker-build

# Run with docker-compose
make docker-run

# Run API server locally
make run

# Generate documentation
make docs
```

Alternatively, you can use the run.sh script for common tasks:

```bash
# Start the FastAPI server
./run.sh api

# Start Jupyter Lab
./run.sh jupyter

# Run tests
./run.sh test

# Run linting
./run.sh lint
```

### Docker Deployment

The project includes Docker support for containerized deployment:

```bash
# Build and run with Docker Compose
make docker-build
make docker-run

# Or using run script
./run.sh docker
```

The Docker container exposes port 8000 for the API and includes health checks.

## Development Conventions

### Code Style
- Python 3.10+ compatible
- Type hints for all functions
- 100 character line limit
- Black formatting with 100 character line length
- MyPy type checking

### Documentation Standards
Each function should include:
1. Brief description
2. Mathematical definition (using Unicode symbols)
3. Args and Returns sections
4. Example usage

### Testing
- Unit tests for all functions
- Test coverage using pytest-cov
- Integration tests for API endpoints
- End-to-end tests for complex workflows

### Directory Structure

```
AI-Mastery-2026/
├── src/                     # Source code
│   ├── core/                # Mathematical foundations
│   ├── ml/                  # Machine Learning algorithms
│   ├── llm/                 # LLM engineering
│   └── production/          # Production components
├── research/                # Jupyter notebooks
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── scripts/                 # Automation tools
├── benchmarks/              # Performance tests
└── Makefile                 # Build automation
```

### Key Dependencies

- numpy, pandas, scipy for scientific computing
- scikit-learn for traditional ML
- pytorch, transformers for deep learning
- fastapi, uvicorn for API development
- langchain for LLM applications
- pytest for testing
- black, mypy for code quality

## Key Features

### Mathematical Foundations
- Linear algebra operations implemented from scratch
- Optimization algorithms (SGD, Adam)
- Probability and statistics

### Machine Learning Algorithms
- Linear/Logistic regression from scratch
- Decision trees and ensemble methods
- Neural networks with complete backpropagation
- Deep learning components

### LLM Engineering
- Transformer architecture implementation
- Multi-head attention mechanisms
- LoRA for fine-tuning
- RAG pipeline components

### Production Components
- FastAPI model serving
- Model caching and optimization
- Monitoring and metrics collection
- Health checks and error handling
- SSE streaming for LLM responses
- Vector databases (HNSW, LSH)

### Testing and Quality
- Comprehensive unit test suite
- Type checking with MyPy
- Code formatting with Black
- Linting with Flake8

The project follows a 6-phase learning roadmap covering Math Foundations, Classical ML, Deep Learning, Transformers, LLM Engineering, and Production considerations. Each implementation includes both from-scratch versions and production-ready components.