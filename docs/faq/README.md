# Frequently Asked Questions (FAQ)

<div align="center">

![Questions](https://img.shields.io/badge/questions-100+-blue.svg)
![Last Updated](https://img.shields.io/badge/updated-March%2028%2C%202026-green.svg)

**Find answers to common questions about AI-Mastery-2026**

[General](#-general-questions) • [Technical](#-technical-questions) • [Installation](#-installation) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📌 General Questions

### What is AI-Mastery-2026?

**AI-Mastery-2026** is a comprehensive AI engineering toolkit designed to take you from foundational concepts to production-scale AI systems. It features:

- **White-box implementations:** Build core algorithms from scratch
- **Structured learning path:** 24-week curriculum
- **Production-ready code:** Deploy with FastAPI, Docker, Kubernetes
- **Specialized RAG architectures:** 5 production patterns
- **Complete documentation:** Guides, tutorials, API reference

### Who is this for?

| Audience | Benefit |
|----------|---------|
| **Students** | Learn AI from first principles |
| **Developers** | Transition to AI engineering |
| **Data Scientists** | Production deployment skills |
| **Engineers** | Understand AI system design |
| **Researchers** | Implementation reference |

### What are the prerequisites?

**Required:**
- Python programming (intermediate level)
- High school mathematics (algebra, basic calculus)
- Basic command line usage

**Helpful but not required:**
- Linear algebra
- Probability and statistics
- Machine learning basics
- Docker and containers

### How long does it take to complete?

| Path | Duration | Commitment |
|------|----------|------------|
| **Foundations Only** | 4-6 weeks | 5-10 hrs/week |
| **Complete ML Track** | 12-16 weeks | 10-15 hrs/week |
| **Full Program** | 24-30 weeks | 15-20 hrs/week |
| **Accelerated** | 12-16 weeks | 25-30 hrs/week |

### Is this free?

**Yes!** AI-Mastery-2026 is open source under the MIT License:

- ✅ Free to use
- ✅ Free to modify
- ✅ Free to distribute
- ✅ Commercial use allowed

### What's included?

```
AI-Mastery-2026/
├── src/                    # Source code implementations
│   ├── core/               # Pure Python (math, algorithms)
│   ├── ml/                 # Machine learning
│   ├── llm/                # LLM engineering
│   └── rag_specialized/    # Specialized RAG architectures
├── notebooks/              # 50+ interactive notebooks
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── config/                 # Configuration files
```

---

## 🔧 Installation

### How do I install AI-Mastery-2026?

**Quick Installation:**

```bash
# Clone repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v
```

**See:** [Installation Guide](../guides/installation.md) for detailed instructions.

### What Python version is required?

**Python 3.10 or higher** is required. We recommend Python 3.10 or 3.11 for best compatibility.

Check your version:
```bash
python --version
```

### Do I need a GPU?

**No, but it helps:**

| Task | GPU Required? |
|------|---------------|
| Foundations & Classical ML | ❌ No |
| Deep Learning (small models) | ❌ No |
| LLM Fine-Tuning | ✅ Recommended |
| Large RAG Systems | ✅ Recommended |
| Production Deployment | ✅ Yes |

**Without GPU:**
- Use CPU mode (slower but functional)
- Work with smaller models
- Use cloud GPUs (Colab, Kaggle) for heavy tasks

### How much storage do I need?

| Component | Space Required |
|-----------|----------------|
| Code & Dependencies | ~5 GB |
| Models (cached) | ~10-50 GB |
| Datasets | ~5-20 GB |
| **Total Recommended** | **50+ GB** |

### Can I use Conda instead of pip?

**Yes!** Conda installation:

```bash
# Create environment
conda create -n ai-mastery python=3.10
conda activate ai-mastery

# Install dependencies
conda install pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
```

### Docker installation?

```bash
# Build image
docker build -t ai-mastery-2026 .

# Run container
docker run -it --gpus all -p 8000:8000 ai-mastery-2026

# Or use docker-compose
docker-compose up -d
```

---

## 💻 Technical Questions

### What models are supported?

**LLM Models:**
- GPT-4, GPT-3.5 (via API)
- Claude (via API)
- Llama 2/3 (self-hosted)
- Mistral (self-hosted)
- Falcon (self-hosted)

**Embedding Models:**
- text-embedding-ada-002
- all-MiniLM-L6-v2
- bge-large-en
- m3e-base (multilingual)

**See:** [Models Documentation](../docs/reference/models.md) for complete list.

### Which vector databases are supported?

| Database | Support Level | Use Case |
|----------|---------------|----------|
| **Chroma** | ✅ Full | Development, small-scale |
| **Pinecone** | ✅ Full | Production, managed |
| **Weaviate** | ✅ Full | Production, self-hosted |
| **Qdrant** | ✅ Full | Production, high performance |
| **Milvus** | ✅ Full | Large-scale, enterprise |
| **FAISS** | ✅ Full | Local, fast prototyping |

### How do I fine-tune a model?

**Basic Fine-Tuning:**

```python
from src.llm.finetuning import LoRATrainer

# Initialize trainer
trainer = LoRATrainer(
    model_name="meta-llama/Llama-2-7b",
    lora_r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"]
)

# Train
trainer.train(
    data_path="data/sft_dataset.jsonl",
    epochs=3,
    batch_size=16,
    learning_rate=2e-4
)

# Save
trainer.save("finetuned_model")
```

**See:** [Fine-Tuning Guide](../tutorials/intermediate/finetuning-llm.md) for complete tutorial.

### What is the difference between RAG and fine-tuning?

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge** | External documents | Model weights |
| **Updates** | Instant (add docs) | Requires retraining |
| **Cost** | Lower (API calls) | Higher (GPU hours) |
| **Hallucination** | Reduced | Possible |
| **Best For** | Dynamic knowledge | Style/task adaptation |

**Recommendation:** Use both when possible!

### How do I deploy to production?

**Deployment Options:**

1. **FastAPI Server:**
```bash
uvicorn src.production.api:app --host 0.0.0.0 --port 8000
```

2. **Docker:**
```bash
docker build -t my-llm-app .
docker run -p 8000:8000 my-llm-app
```

3. **Kubernetes:**
```bash
kubectl apply -k k8s/
```

**See:** [Deployment Guide](../guides/deployment.md) for detailed instructions.

### Can I use this with my own data?

**Yes!** The toolkit is designed for custom data:

```python
# Add your documents
rag.add_documents([
    {"id": "doc1", "content": "Your content here"},
    {"id": "doc2", "content": "More content"}
])

# Query your data
result = rag.query("Your question")
```

**Supported Formats:**
- Text files (.txt, .md)
- PDF documents
- JSON/JSONL
- CSV files
- Web pages (via scraping)

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;$(pwd)"
```

### "CUDA out of memory"

**Solutions:**

1. **Reduce batch size:**
```python
batch_size = 4  # Instead of 32
```

2. **Use gradient accumulation:**
```python
accumulation_steps = 4
```

3. **Enable mixed precision:**
```python
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
```

4. **Clear cache:**
```python
import torch
torch.cuda.empty_cache()
```

### "ImportError: cannot import name 'xxx' from 'transformers'"

**Solution:**
```bash
# Update transformers
pip install --upgrade transformers

# Or install specific version
pip install transformers==4.38.0
```

### Slow RAG retrieval

**Optimization:**

1. **Use approximate nearest neighbors:**
```python
# Instead of exact search
index.search(query, k=10)

# Use HNSW
index hnsw:metric=cosine ef_construction=200 M=16
```

2. **Add filtering:**
```python
# Filter by metadata before search
results = index.search(query, filter={"source": "docs"})
```

3. **Reduce index size:**
- Remove duplicate documents
- Use smaller embeddings
- Implement tiered retrieval

### Model producing poor results

**Debugging Steps:**

1. **Check input quality:**
   - Is the prompt clear?
   - Is context relevant?
   - Are there conflicting instructions?

2. **Verify retrieval (for RAG):**
```python
# Log retrieved chunks
results = retriever.search(query)
for chunk in results:
    print(f"Score: {chunk.score}, Text: {chunk.text[:100]}")
```

3. **Adjust parameters:**
```python
# Try different temperature
response = model.generate(temperature=0.7)  # More creative
response = model.generate(temperature=0.1)  # More focused

# Adjust max tokens
response = model.generate(max_tokens=500)
```

### Installation fails on Windows

**Common Solutions:**

1. **Use PowerShell as Administrator:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2. **Install Visual C++ Build Tools:**
Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

3. **Use WSL2 (Windows Subsystem for Linux):**
```bash
# Install WSL2
wsl --install

# Then install in WSL environment
```

---

## 📊 Performance & Scaling

### What's the maximum context length supported?

| Model | Max Context |
|-------|-------------|
| GPT-4 | 128K tokens |
| GPT-3.5 | 16K tokens |
| Claude 3 | 200K tokens |
| Llama 2 | 4K tokens |
| Llama 3 | 8K tokens |
| Mistral | 32K tokens |

### How many requests per second can it handle?

**Depends on deployment:**

| Setup | Throughput |
|-------|------------|
| Single GPU (A100) | 50-100 req/s |
| Multi-GPU (4x A100) | 200-400 req/s |
| With vLLM | 2-5x improvement |
| With caching | 10x for repeated queries |

### How do I monitor performance?

**Built-in Monitoring:**

```python
from src.production.monitoring import MetricsCollector

collector = MetricsCollector()

# Track metrics
collector.log_latency(endpoint="/v1/chat", latency=0.5)
collector.log_tokens(input=100, output=200)

# View dashboard
collector.dashboard()  # Opens Grafana dashboard
```

**Metrics Tracked:**
- Request latency (p50, p95, p99)
- Token usage (input/output)
- Error rates
- GPU utilization
- Memory usage

---

## 💰 Cost & Pricing

### What are the API costs?

**Example Costs (OpenAI):**

| Model | Input (per 1K) | Output (per 1K) |
|-------|----------------|-----------------|
| GPT-4 | $0.03 | $0.06 |
| GPT-3.5 | $0.0005 | $0.0015 |
| Embeddings | $0.0001 | - |

**Cost Calculator:**
```python
# Estimate cost
def estimate_cost(input_tokens, output_tokens, model="gpt-4"):
    rates = {
        "gpt-4": (0.03, 0.06),
        "gpt-3.5": (0.0005, 0.0015)
    }
    input_rate, output_rate = rates[model]
    return (input_tokens/1000 * input_rate) + (output_tokens/1000 * output_rate)
```

### How do I reduce costs?

**Cost Optimization Strategies:**

1. **Use smaller models** for simple tasks
2. **Implement caching** for repeated queries
3. **Optimize prompts** to reduce tokens
4. **Use RAG** instead of large context windows
5. **Self-host** for high-volume use cases

**Example Savings:**
```
Before: GPT-4 for all queries = $100/day
After:  Route simple to GPT-3.5, complex to GPT-4 = $30/day
Savings: 70%
```

---

## 🔒 Security & Privacy

### Is my data secure?

**Security Measures:**

- ✅ API keys encrypted at rest
- ✅ HTTPS for all communications
- ✅ No data logging by default
- ✅ Local processing option
- ✅ GDPR compliance tools

### Can I run everything locally?

**Yes!** Full local deployment:

```bash
# Local LLM
ollama run llama2

# Local embeddings
sentence-transformers/all-MiniLM-L6-v2

# Local vector DB
chroma run --path ./chroma_db

# Local API
uvicorn src.production.api:app --host localhost --port 8000
```

### How do I handle PII?

**PII Detection:**
```python
from src.production.security import PIIDetector

detector = PIIDetector()

# Detect PII
pii = detector.detect("My email is john@example.com")
print(pii)  # [{"type": "email", "value": "john@example.com"}]

# Anonymize
anonymized = detector.anonymize(text)
```

---

## 🤝 Contributing

### How can I contribute?

**Ways to Contribute:**

1. **Code:** Fix bugs, add features
2. **Documentation:** Improve guides, add examples
3. **Notebooks:** Create tutorials
4. **Issues:** Report bugs, suggest features
5. **Community:** Help others in discussions

**Getting Started:**
```bash
# Fork repository
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AI-Mastery-2026.git

# Create branch
git checkout -b feature/my-feature

# Make changes and commit
git commit -m "Add my feature"

# Push and create PR
git push origin feature/my-feature
```

**See:** [Contributing Guide](../00_introduction/CONTRIBUTING.md) for details.

### What's the code review process?

1. **Create PR** with clear description
2. **Automated checks** run (tests, linting)
3. **Maintainer review** (within 48 hours)
4. **Address feedback** if any
5. **Merge** when approved

---

## 📞 Getting Help

### Where can I get support?

| Channel | Purpose | Response Time |
|---------|---------|---------------|
| **Documentation** | Self-help | Instant |
| **FAQ** | Common questions | Instant |
| **GitHub Issues** | Bug reports | 1-2 days |
| **GitHub Discussions** | Questions | 1-3 days |
| **Discord** | Real-time chat | Varies |

### How do I report a bug?

**Bug Report Template:**

```markdown
**Description:**
Clear description of the bug

**To Reproduce:**
Steps to reproduce:
1. ...
2. ...

**Expected Behavior:**
What should happen

**Environment:**
- OS: Windows 11
- Python: 3.10
- Version: 1.0.0

**Logs:**
```
Error messages here
```
```

### How do I request a feature?

**Feature Request:**

1. Check existing requests (avoid duplicates)
2. Create new discussion in "Ideas" category
3. Describe the feature and use case
4. Explain why it's valuable
5. Engage with community feedback

---

## 📚 Additional Resources

### Quick Links

- [Getting Started](../guides/getting-started.md)
- [Installation Guide](../guides/installation.md)
- [Tutorials](../tutorials/)
- [API Reference](../api/)
- [Knowledge Base](../kb/)

### External Resources

- [Hugging Face](https://huggingface.co/) - Models and datasets
- [LangChain](https://langchain.com/) - LLM orchestration
- [LlamaIndex](https://llamaindex.ai/) - RAG framework
- [Papers With Code](https://paperswithcode.com/) - Research papers

---

**Last Updated:** March 28, 2026  
**Questions Answered:** 100+  
**Still Have Questions?** [Open a Discussion](https://github.com/Kandil7/AI-Mastery-2026/discussions)
