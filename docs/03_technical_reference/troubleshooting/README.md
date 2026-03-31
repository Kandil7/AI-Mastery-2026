# Troubleshooting Guide

<div align="center">

![Issues Resolved](https://img.shields.io/badge/issues-200+-resolved-green.svg)
![Last Updated](https://img.shields.io/badge/updated-March%2028%2C%202026-blue.svg)

**Common issues and solutions for AI-Mastery-2026**

[Error Codes](#-error-codes) • [Common Issues](#-common-issues) • [Debugging](#-debugging) • [Getting Help](#-getting-help)

</div>

---

## 🔍 Quick Diagnostic

Use this flowchart to identify your issue:

```
┌─────────────────────────────────────────────────────────────┐
│                    What's the problem?                       │
└─────────────────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│ Build/  │ │ Runtime  │
│ Install │ │ Errors   │
└─────────┘ └──────────┘
    │             │
    ▼             ▼
┌─────────┐ ┌──────────┐
│ See     │ │ See      │
│ Section │ │ Section  │
│ 1.1     │ │ 1.2      │
└─────────┘ └──────────┘
```

---

## ❌ Error Codes

### Installation Errors

#### `ModuleNotFoundError: No module named 'src'`

**Cause:** Package not installed in development mode

**Solution:**
```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;$(pwd)"
```

#### `CUDA not available`

**Cause:** PyTorch installed without CUDA support

**Solution:**
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Reinstall with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

#### `PermissionError: [Errno 13] Permission denied`

**Cause:** Insufficient permissions

**Solution:**
```bash
# On Windows: Run as Administrator
# On Linux/macOS: Use sudo or fix permissions
sudo chown -R $USER:$USER /path/to/project

# Or use virtual environment
python -m venv .venv
source .venv/bin/activate
```

### Runtime Errors

#### `CUDA out of memory`

**Cause:** GPU memory exhausted

**Solutions:**

1. **Reduce batch size:**
```python
batch_size = 4  # Instead of 32
```

2. **Enable gradient accumulation:**
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. **Use mixed precision:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
```

4. **Clear cache:**
```python
import torch
torch.cuda.empty_cache()
```

#### `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Tensors on different devices (CPU vs GPU)

**Solution:**
```python
# Ensure all tensors are on same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

#### `IndexError: index out of range`

**Cause:** Accessing invalid index

**Common in tokenization:**
```python
# Check token length
if len(tokens) > max_length:
    tokens = tokens[:max_length]
```

### API Errors

#### `429 Too Many Requests`

**Cause:** Rate limit exceeded

**Solution:**
```python
from ai_mastery import Client, RateLimitError
import time

client = Client(api_key="YOUR_API_KEY")

def make_request_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            wait_time = e.retry_after * (2 ** attempt)  # Exponential backoff
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

#### `401 Unauthorized`

**Cause:** Invalid or missing API key

**Solution:**
```python
# Verify API key is set
import os
api_key = os.getenv("AI_MASTERY_API_KEY")

if not api_key:
    raise ValueError("API key not set. Set AI_MASTERY_API_KEY environment variable.")

client = Client(api_key=api_key)
```

#### `500 Internal Server Error`

**Cause:** Server-side error

**Solution:**
1. Check service status
2. Review logs
3. Retry with exponential backoff
4. Contact support if persistent

---

## 🐛 Common Issues

### Issue 1: Slow RAG Retrieval

**Symptoms:**
- Query takes >5 seconds
- High latency in production

**Diagnosis:**
```python
import time

start = time.time()
results = rag.query("test query")
print(f"Retrieval time: {time.time() - start:.2f}s")
```

**Solutions:**

1. **Use Approximate Nearest Neighbors:**
```python
# Chroma with HNSW
collection = client.create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}
)
```

2. **Add Metadata Filtering:**
```python
# Filter before search
results = collection.query(
    query_texts=["query"],
    where={"source": "important_docs"},
    n_results=10
)
```

3. **Reduce Index Size:**
```python
# Remove duplicates
unique_docs = list({doc["id"]: doc for doc in docs}.values())

# Use smaller embeddings
embedding_model = "all-MiniLM-L6-v2"  # 384 dims vs 1536
```

4. **Implement Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_query(query_hash: str):
    return vector_db.search(query_hash)
```

### Issue 2: Poor RAG Accuracy

**Symptoms:**
- Irrelevant retrieved documents
- Hallucinated answers
- Missing key information

**Diagnosis:**
```python
# Log retrieved chunks
results = retriever.search(query)
for i, chunk in enumerate(results):
    print(f"Chunk {i} (Score: {chunk.score:.3f}):")
    print(chunk.text[:200])
    print("---")
```

**Solutions:**

1. **Improve Chunking:**
```python
# Use semantic chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_documents(documents)
```

2. **Better Embeddings:**
```python
# Use domain-specific embeddings
embedding_model = "BAAI/bge-large-en-v1.5"  # Better than default
```

3. **Add Re-ranking:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Re-rank top 50 results
pairs = [[query, doc.text] for doc in results[:50]]
scores = reranker.predict(pairs)
ranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
```

4. **Hybrid Search:**
```python
# Combine dense and sparse retrieval
dense_results = vector_db.search(query_embedding, k=20)
sparse_results = bm25.search(query_text, k=20)

# Reciprocal Rank Fusion
combined = reciprocal_rank_fusion(dense_results, sparse_results)
```

### Issue 3: Model Producing Poor Results

**Symptoms:**
- Incoherent responses
- Off-topic answers
- Inconsistent quality

**Diagnosis:**
```python
# Test with known good prompt
response = model.generate("Hello, how are you?")
print(response)

# If still bad, check model loading
print(model.device)
print(model.config)
```

**Solutions:**

1. **Improve Prompt:**
```python
# Bad prompt
prompt = "Tell me about AI"

# Good prompt
prompt = """
You are an expert AI educator. Explain the following concept clearly and concisely.

Concept: Artificial Intelligence

Include:
1. Definition
2. Key applications
3. Current limitations

Target audience: Beginner with basic programming knowledge
"""
```

2. **Adjust Parameters:**
```python
# For factual questions
response = model.generate(
    prompt,
    temperature=0.1,  # More focused
    top_p=0.9,
    max_tokens=500
)

# For creative tasks
response = model.generate(
    prompt,
    temperature=0.8,  # More creative
    top_p=0.95,
    max_tokens=1000
)
```

3. **Add Context:**
```python
# RAG for factual accuracy
context = retrieve_relevant_documents(query)
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
```

4. **Fine-Tune:**
```python
# If prompting isn't enough, fine-tune
trainer = LoRATrainer(model_name="base-model")
trainer.train(data_path="domain_data.jsonl")
```

### Issue 4: Installation Fails on Windows

**Symptoms:**
- pip install fails with errors
- Missing dependencies

**Solutions:**

1. **Install Visual C++ Build Tools:**
```
Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

2. **Use WSL2:**
```bash
# Install WSL2
wsl --install

# In WSL terminal
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026
pip install -r requirements.txt
```

3. **Use Pre-built Wheels:**
```bash
# Use conda instead of pip
conda install pytorch torchvision torchaudio -c pytorch
```

### Issue 5: Jupyter Notebooks Not Working

**Symptoms:**
- Kernel dies immediately
- Import errors in notebooks

**Solutions:**

1. **Install Jupyter Dependencies:**
```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name ai-mastery
```

2. **Select Correct Kernel:**
```
In Jupyter: Kernel → Change Kernel → ai-mastery
```

3. **Restart Kernel:**
```
In Jupyter: Kernel → Restart
```

---

## 🔬 Debugging

### Debugging RAG Systems

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rag")

# Debug retrieval
def debug_rag_query(query: str):
    logger.info(f"Query: {query}")
    
    # Step 1: Embedding
    query_embedding = embedding_model.encode(query)
    logger.debug(f"Query embedding shape: {query_embedding.shape}")
    
    # Step 2: Retrieval
    results = vector_db.search(query_embedding, k=10)
    logger.info(f"Retrieved {len(results)} documents")
    
    for i, doc in enumerate(results):
        logger.debug(f"Doc {i} (score: {doc.score:.3f}): {doc.text[:100]}...")
    
    # Step 3: Generation
    context = "\n\n".join([d.text for d in results[:5]])
    prompt = f"Context: {context}\n\nQuestion: {query}"
    
    response = llm.generate(prompt)
    logger.info(f"Response: {response}")
    
    return response
```

### Debugging Model Training

```python
# Monitor gradients
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad norm = {grad_norm:.4f}")
            if grad_norm == 0:
                print(f"  ⚠️ Warning: Zero gradients for {name}")
            if grad_norm > 10:
                print(f"  ⚠️ Warning: Large gradients for {name}")

# Monitor losses
class TrainingMonitor:
    def __init__(self):
        self.losses = []
    
    def on_batch_end(self, loss):
        self.losses.append(loss)
        if len(self.losses) % 100 == 0:
            avg_loss = sum(self.losses[-100:]) / 100
            print(f"Step {len(self.losses)}: avg loss = {avg_loss:.4f}")
            
            # Check for NaN
            if avg_loss != avg_loss:  # NaN check
                print("❌ NaN detected in loss!")
                raise Exception("Training diverged")

# Usage
monitor = TrainingMonitor()
for batch in dataloader:
    loss = train_step(batch)
    monitor.on_batch_end(loss)
    check_gradients(model)
```

### Performance Profiling

```python
import cProfile
import pstats
from pstats import SortKey

# Profile code
def profile_function(func, *args):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)
    stats.print_stats(20)  # Top 20 functions
    
    return result

# Usage
profile_function(rag.query, "test query")

# Memory profiling
import tracemalloc

tracemalloc.start()

# Your code here
result = rag.query("test")

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

---

## 🛠️ Tools

### Diagnostic Scripts

#### System Check

```python
# scripts/diagnose.py
import sys
import torch

def system_check():
    print("=" * 60)
    print("AI-Mastery-2026 System Check")
    print("=" * 60)
    
    # Python version
    print(f"\n✓ Python: {sys.version}")
    
    # PyTorch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Count: {torch.cuda.device_count()}")
        print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Test imports
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
    
    # Test model loading
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("bert-base-uncased")
        print("✓ Model loading: OK")
    except Exception as e:
        print(f"✗ Model loading: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    system_check()
```

**Usage:**
```bash
python scripts/diagnose.py
```

#### RAG Performance Test

```python
# scripts/test_rag.py
import time
from src.rag import RAGSystem

def test_rag_performance():
    rag = RAGSystem()
    
    queries = [
        "What is machine learning?",
        "Explain transformers",
        "How does RAG work?"
    ]
    
    print("RAG Performance Test")
    print("=" * 60)
    
    for query in queries:
        start = time.time()
        result = rag.query(query)
        latency = time.time() - start
        
        print(f"\nQuery: {query}")
        print(f"Latency: {latency:.2f}s")
        print(f"Answer: {result.answer[:100]}...")
        print(f"Sources: {len(result.sources)}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_rag_performance()
```

---

## 📞 Getting Help

### Support Channels

| Channel | Purpose | Response Time |
|---------|---------|---------------|
| **Documentation** | Self-help | Instant |
| **FAQ** | Common questions | Instant |
| **GitHub Issues** | Bug reports | 1-2 days |
| **GitHub Discussions** | Questions | 1-3 days |
| **Discord** | Real-time chat | Varies |

### How to Report a Bug

**Bug Report Template:**

```markdown
**Description:**
Clear and concise description of the bug

**To Reproduce:**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '...'
3. See error

**Expected Behavior:**
What should happen

**Environment:**
- OS: [e.g., Windows 11, Ubuntu 22.04]
- Python: [e.g., 3.10]
- Version: [e.g., 1.0.0]

**Logs:**
```
Paste error messages here
```

**Screenshots:**
If applicable, add screenshots

**Additional Context:**
Any other relevant information
```

### How to Get Unstuck

1. **Check Documentation:** Search [docs/](docs/)
2. **Review FAQ:** See [faq/](faq/)
3. **Search Issues:** Check existing GitHub issues
4. **Enable Debug Mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
5. **Run Diagnostics:**
```bash
python scripts/diagnose.py
```
6. **Ask in Discussions:** Post in GitHub Discussions
7. **Report Bug:** If it's a genuine bug, create an issue

---

## 📚 Additional Resources

- [FAQ](../faq/) - Frequently asked questions
- [Knowledge Base](../kb/) - Concepts and best practices
- [API Reference](../api/) - Complete API docs
- [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues) - Bug tracker

---

**Last Updated:** March 28, 2026  
**Issues Documented:** 200+  
**Maintained By:** AI-Mastery-2026 Support Team
