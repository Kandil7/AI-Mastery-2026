# Installation Guide

**Last Updated**: March 27, 2026

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| **CPU** | 4 cores | 8 cores | 16+ cores |
| **RAM** | 8 GB | 16 GB | 32+ GB |
| **Storage** | 10 GB SSD | 50 GB SSD | 100+ GB SSD |
| **GPU** | Optional | NVIDIA 8GB+ | NVIDIA 16GB+ |

### Software Requirements

- **Python**: 3.10 or higher
- **pip**: 21.0 or higher
- **Git**: For version control
- **Docker**: Optional, for containerized deployment

---

## Installation Steps

### Step 1: Clone or Navigate to Project

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026\rag_system
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Core Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install Optional Dependencies

**For GPU support:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For Qdrant vector database:**
```bash
pip install qdrant-client
```

**For advanced features:**
```bash
pip install peft  # Parameter-efficient fine-tuning
pip install cohere  # Cohere embeddings/reranking
```

---

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# ===========================================
# API Keys (Required for LLM features)
# ===========================================

# OpenAI (for GPT-4, embeddings)
OPENAI_API_KEY=sk-your-key-here

# Anthropic (for Claude)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Cohere (optional, for reranking)
COHERE_API_KEY=your-key-here

# ===========================================
# Paths (Auto-detected, override if needed)
# ===========================================

DATASETS_PATH=K:/learning/technical/ai-ml/AI-Mastery-2026/datasets
OUTPUT_PATH=K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data

# ===========================================
# Vector Database (Optional)
# ===========================================

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=  # Leave empty for local

# ===========================================
# Application Settings
# ===========================================

LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
ENABLE_MONITORING=true
```

### Configuration File

Edit `config/config.yaml` for advanced settings:

```yaml
data:
  datasets_path: "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets"
  output_path: "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data"

embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  batch_size: 32
  max_length: 512

vector_db:
  type: "memory"  # or "qdrant", "chroma"
  collection_name: "arabic_islamic_literature"

retrieval:
  top_k: 5
  hybrid_weights:
    semantic: 0.7
    bm25: 0.3

llm:
  provider: "openai"  # or "anthropic", "ollama", "mock"
  model: "gpt-4o"
  temperature: 0.3
```

---

## Verification

### Test Installation

Run the test suite:

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
  ✅ Retrieval imports OK
  ✅ Generation imports OK
  ✅ Specialists imports OK
  ✅ Agents imports OK
  ✅ Evaluation imports OK
  ✅ Monitoring imports OK
  ✅ API imports OK

[TEST 2] Testing component instantiation...
  ✅ Chunker created OK
  ✅ Vector store created OK
  ✅ Agent created OK

[TEST 3] Testing basic functionality...
  ✅ Chunking works: 1 chunks

======================================================================
ALL TESTS PASSED ✅
======================================================================
```

### Test API

Start the API server:

```bash
uvicorn src.api.service:app --reload
```

Access the API docs:
```
http://localhost:8000/docs
```

Try a test query:
```bash
curl -X POST http://localhost:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"ما هو التوحيد؟\", \"top_k\": 5}"
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Error:** `ModuleNotFoundError: No module named 'rag_system'`

**Solution:**
```bash
# Make sure you're in the project directory
cd K:\learning\technical\ai-ml\AI-Mastery-2026\rag_system

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. CUDA/GPU Issues

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Use CPU instead
export EMBEDDING_DEVICE=cpu

# Or reduce batch size
# Edit config/config.yaml
embedding:
  batch_size: 16  # Reduce from 32
```

#### 3. API Key Issues

**Error:** `OpenAI API key not provided`

**Solution:**
```bash
# Set environment variable
export OPENAI_API_KEY=sk-...

# Or add to .env file
OPENAI_API_KEY=sk-your-key-here
```

#### 4. Slow Performance

**Issue:** Queries taking too long

**Solutions:**
1. Use GPU if available
2. Reduce `retrieval_top_k` in config
3. Enable caching
4. Use smaller embedding model

```yaml
embedding:
  model_name: "sentence-transformers/bert-base-multilingual-cased"  # Smaller
  batch_size: 16

retrieval:
  top_k: 20  # Reduce from 50
```

#### 5. Disk Space Issues

**Error:** `No space left on device`

**Solution:**
```bash
# Clear cache
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

# Clear rag_system cache
rm -rf data/embedding_cache/*
```

---

## Next Steps

After successful installation:

1. ✅ **Verify installation** - Run `python simple_test.py`
2. ✅ **Configure environment** - Set up `.env` file
3. ✅ **Test basic functionality** - Run a query
4. ✅ **Read user guide** - See [Basic Queries](../04_user_guides/basic_queries.md)

---

**Having issues?** See [Troubleshooting FAQ](../07_faq/troubleshooting.md)
