# Quick Start Guide - 5 Minutes to Your First Query

**Last Updated**: March 27, 2026

---

## ⚡ 5-Minute Quick Start

Get up and running with the RAG system in 5 minutes!

---

## Step 1: Install (1 minute)

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026\rag_system
pip install -r requirements.txt
```

---

## Step 2: Test Installation (1 minute)

```bash
python simple_test.py
```

Wait for: **ALL TESTS PASSED ✅**

---

## Step 3: Run Demo (2 minutes)

```bash
python example_usage_complete.py
```

This demonstrates:
- ✅ Basic queries
- ✅ Domain specialists
- ✅ Comparative fiqh
- ✅ Agent system
- ✅ Evaluation
- ✅ Monitoring

---

## Step 4: Your First Query (1 minute)

Create `query.py`:

```python
import asyncio
from src.integration import create_islamic_rag

async def main():
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Basic query
    result = await rag.query("ما هو التوحيد في الإسلام؟")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Sources: {len(result['sources'])} sources")

asyncio.run(main())
```

Run it:
```bash
python query.py
```

---

## 🎯 What's Next?

### For Quick Testing

```bash
# Try the interactive API
uvicorn src.api.service:app --reload

# Open browser to:
# http://localhost:8000/docs
```

### For Learning

- [Full Getting Started Guide](index.md)
- [Usage Examples](../USAGE_EXAMPLES.md)
- [Domain Specialists](../04_user_guides/domain_specialists.md)

### For Production

- [Deployment Guide](../DEPLOYMENT_GUIDE.md)
- [Docker Setup](../06_deployment/docker.md)
- [API Reference](../03_api_reference/rest_api.md)

---

## 📞 Need Help?

- **Installation issues?** → [Installation Guide](installation.md)
- **Query errors?** → [Troubleshooting](../07_faq/troubleshooting.md)
- **General questions?** → [FAQ](../07_faq/general.md)

---

**Congratulations!** You're now using the Arabic Islamic Literature RAG System! 🎉
