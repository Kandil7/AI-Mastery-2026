# RAG System Documentation

**Version**: 1.0.0  
**Status**: ✅ Complete

---

## 📚 Welcome to the RAG System Documentation

This documentation covers the **Arabic Islamic Literature RAG System** - a production-grade Retrieval-Augmented Generation system for querying 8,425+ Islamic books.

---

## 🎯 Find Your Path

### I'm a...

#### 👶 Beginner

Start here:
1. [Getting Started Guide](01_getting_started/index.md)
2. [Quick Start (5 min)](01_getting_started/quickstart.md)
3. [Basic Queries](04_user_guides/basic_queries.md)

#### 💻 Developer

Jump to:
1. [Architecture Overview](02_architecture/overview.md)
2. [Python API Reference](03_api_reference/python_api.md)
3. [Development Setup](05_development/setup.md)

#### 🚀 DevOps

Check out:
1. [Deployment Guide](../DEPLOYMENT_GUIDE.md)
2. [Docker Deployment](06_deployment/docker.md)
3. [Monitoring Setup](06_deployment/monitoring.md)

#### 🔬 Researcher

Explore:
1. [Domain Specialists](04_user_guides/domain_specialists.md)
2. [Agent System](04_user_guides/agent_system.md)
3. [Advanced Features](04_user_guides/advanced_features.md)

---

## 📖 Documentation Sections

### [01_getting_started/](01_getting_started/index.md)

For new users getting started with the RAG system.

- [Index](01_getting_started/index.md) - Overview
- [Installation](01_getting_started/installation.md) - Step-by-step install
- [Quick Start](01_getting_started/quickstart.md) - 5-minute tutorial
- [Configuration](01_getting_started/configuration.md) - Config options

### [02_architecture/](02_architecture/overview.md)

System architecture and design.

- [Overview](02_architecture/overview.md) - High-level architecture
- [Components](02_architecture/components.md) - Component details
- [Data Flow](02_architecture/data_flow.md) - How data flows
- [Design Decisions](02_architecture/design_decisions.md) - Why we built it this way

### [03_api_reference/](03_api_reference/rest_api.md)

API documentation.

- [REST API](03_api_reference/rest_api.md) - HTTP API reference
- [Python API](03_api_reference/python_api.md) - Python client library
- [Examples](03_api_reference/examples.md) - API usage examples

### [04_user_guides/](04_user_guides/basic_queries.md)

User guides for different use cases.

- [Basic Queries](04_user_guides/basic_queries.md) - Getting started with queries
- [Domain Specialists](04_user_guides/domain_specialists.md) - Using specialists
- [Agent System](04_user_guides/agent_system.md) - Working with agents
- [Advanced Features](04_user_guides/advanced_features.md) - Power user features

### [05_development/](05_development/setup.md)

Development guides.

- [Setup](05_development/setup.md) - Dev environment setup
- [Testing](05_development/testing.md) - Testing guide
- [Contributing](05_development/contributing.md) - How to contribute
- [Code Style](05_development/code_style.md) - Coding standards

### [06_deployment/](06_deployment/docker.md)

Production deployment.

- [Docker](06_deployment/docker.md) - Containerized deployment
- [Cloud](06_deployment/cloud.md) - Cloud deployment (AWS, GCP, Azure)
- [Monitoring](06_deployment/monitoring.md) - Observability setup
- [Troubleshooting](06_deployment/troubleshooting.md) - Common issues

### [07_faq/](07_faq/general.md)

Frequently asked questions.

- [General FAQ](07_faq/general.md) - General questions
- [Technical FAQ](07_faq/technical.md) - Technical questions
- [Troubleshooting](07_faq/troubleshooting.md) - Common problems

---

## 📋 Quick Reference

### Installation

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026\rag_system
pip install -r requirements.txt
python simple_test.py
```

### Basic Query

```python
from src.integration import create_islamic_rag

rag = create_islamic_rag()
await rag.initialize()
result = await rag.query("ما هو التوحيد؟")
```

### Start API

```bash
uvicorn src.api.service:app --reload
# Open: http://localhost:8000/docs
```

### Run Demo

```bash
python example_usage_complete.py
```

---

## 🔗 External Resources

- **GitHub Repository**: [AI-Mastery-2026](https://github.com/...)
- **RAG Guide**: [RAG_PIPELINE_COMPLETE_GUIDE_2026.md](../RAG_SYSTEM_COMPLETE_GUIDE.md)
- **Usage Examples**: [USAGE_EXAMPLES.md](../USAGE_EXAMPLES.md)
- **Deployment**: [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)

---

## 🆘 Getting Help

### Documentation Not Helping?

1. **Check FAQ**: [07_faq/troubleshooting.md](07_faq/troubleshooting.md)
2. **Run Tests**: `python simple_test.py`
3. **Check Logs**: `rag_system/logs/`
4. **Examples**: Look at `example_usage_*.py` files

### Still Stuck?

- Review [Troubleshooting Guide](07_faq/troubleshooting.md)
- Check [GitHub Issues](https://github.com/.../issues)
- Read [Architecture Overview](02_architecture/overview.md)

---

## 📊 Documentation Status

| Section | Status | Last Updated |
|---------|--------|--------------|
| Getting Started | ✅ Complete | March 27, 2026 |
| Architecture | ✅ Complete | March 27, 2026 |
| API Reference | ✅ Complete | March 27, 2026 |
| User Guides | ✅ Complete | March 27, 2026 |
| Development | ✅ Complete | March 27, 2026 |
| Deployment | ✅ Complete | March 27, 2026 |
| FAQ | ✅ Complete | March 27, 2026 |

---

## 🎯 Next Steps

**New to RAG?** → [Getting Started](01_getting_started/index.md)

**Want to use the API?** → [API Reference](03_api_reference/rest_api.md)

**Deploying?** → [Deployment Guide](06_deployment/docker.md)

**Having issues?** → [Troubleshooting](07_faq/troubleshooting.md)

---

**Happy learning!** 🚀

---

**Version**: 1.0.0  
**Maintained By**: RAG System Team  
**License**: Educational/Research Use
