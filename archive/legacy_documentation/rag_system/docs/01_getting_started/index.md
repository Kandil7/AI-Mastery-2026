# Getting Started with Arabic Islamic Literature RAG System

**Version**: 1.0.0  
**Last Updated**: March 27, 2026

---

## 🎯 What You'll Learn

- What the RAG system is
- System requirements
- How to install and run
- Your first query

---

## 📋 Prerequisites

### Required Knowledge

- Basic Python programming
- Familiarity with command line
- Understanding of APIs (helpful but not required)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB | 50 GB SSD |
| **Internet** | Required for setup | Required for LLM APIs |

---

## 🚀 Installation

### Step 1: Navigate to RAG System

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026\rag_system
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` - For embeddings
- `fastapi` - For API service
- `uvicorn` - For running the server
- `numpy`, `pydantic` - Core dependencies
- And more...

### Step 4: Verify Installation

```bash
python simple_test.py
```

**Expected Output:**
```
======================================================================
RAG SYSTEM - SIMPLE ARCHITECTURE TEST
======================================================================

[TEST 1] Testing direct submodule imports...
  ✅ Pipeline imports OK
  ✅ Data imports OK
  ✅ Processing imports OK
  ...

======================================================================
ALL TESTS PASSED ✅
======================================================================
```

---

## ⚙️ Configuration

### Environment Variables (Optional)

Create a `.env` file in the root directory:

```bash
# API Keys (for LLM features)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Paths (usually auto-detected)
DATASETS_PATH=K:/learning/technical/ai-ml/AI-Mastery-2026/datasets
OUTPUT_PATH=K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data

# Optional: Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### Default Configuration

The system works out-of-the-box with these defaults:
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **LLM Provider**: `mock` (for testing)
- **Chunk Size**: 512 tokens
- **Retrieval Top-K**: 50 results

---

## 🎓 Your First Query

### Option 1: Python Script

Create `my_first_query.py`:

```python
import asyncio
from src.integration import create_islamic_rag

async def main():
    # Initialize
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Query
    result = await rag.query("ما هو التوحيد في الإسلام؟")
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} sources")

asyncio.run(main())
```

Run it:
```bash
python my_first_query.py
```

### Option 2: Run Demo

```bash
python example_usage_complete.py
```

This runs a comprehensive demo showing all features.

### Option 3: Use API

1. **Start the server:**
   ```bash
   uvicorn src.api.service:app --reload
   ```

2. **Open API docs:**
   ```
   http://localhost:8000/docs
   ```

3. **Try the `/api/v1/query` endpoint**

---

## 📚 Next Steps

### For End Users

- [Quick Start Guide](quickstart.md) - 5-minute tutorial
- [Basic Queries](../04_user_guides/basic_queries.md) - Learn to query
- [Usage Examples](../USAGE_EXAMPLES.md) - Complete examples

### For Developers

- [Architecture Overview](../02_architecture/overview.md) - System design
- [API Reference](../03_api_reference/python_api.md) - Python API
- [Development Setup](../05_development/setup.md) - Dev environment

### For DevOps

- [Deployment Guide](../DEPLOYMENT_GUIDE.md) - Production deployment
- [Docker Setup](../06_deployment/docker.md) - Containerized deployment
- [Monitoring](../06_deployment/monitoring.md) - Observability

---

## 🆘 Getting Help

### Documentation

- [Complete Documentation Index](index.md)
- [FAQ](../07_faq/general.md)
- [Troubleshooting](../07_faq/troubleshooting.md)

### Support Channels

- **GitHub Issues**: Report bugs
- **Documentation**: See other docs in this directory
- **Examples**: Check `example_usage_*.py` files

---

## ✅ Checklist

Before moving to advanced topics, ensure you can:

- [ ] Run `simple_test.py` successfully
- [ ] Execute a basic query
- [ ] Access the API documentation
- [ ] Understand the system architecture at a high level

---

**Ready to learn more?** Continue to [Quick Start Guide](quickstart.md)
