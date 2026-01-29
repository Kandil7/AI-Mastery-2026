# 🔍 RAG Engine Mini

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Architecture](https://img.shields.io/badge/Architecture-Clean-purple.svg)

**Production-Ready RAG Starter Template**  
*Hybrid Search • Cross-Encoder Reranking • Multi-Tenant Design*

[English](#-overview) | [العربية](#-نظرة-عامة)

</div>

---

## 📖 Overview

**RAG Engine Mini** is a production-grade Retrieval-Augmented Generation (RAG) starter template that bridges the gap between notebook experiments and real-world AI systems. Built with Clean Architecture principles, it provides a solid foundation for building intelligent document Q&A systems.

### Why This Project? / لماذا هذا المشروع؟

Most RAG tutorials show you the basics: embed → store → search → generate. But production systems need much more:

| Challenge | Our Solution |
|-----------|--------------|
| **Recall issues** | Hybrid search (Vector + Keyword FTS) + **Query Expansion** |
| **Precision problems** | Cross-Encoder reranking |
| **Cost control** | Batch embeddings + Redis caching |
| **Data isolation** | Multi-tenant by design |
| **Duplicate processing** | File hash idempotency + chunk dedup |
| **Vendor lock-in** | Ports & Adapters pattern |
| **Scalability** | Async indexing with Celery |
| **Visibility** | **Prometheus Metrics** + **Gradio Demo UI** |

---

## 📖 نظرة عامة

**RAG Engine Mini** هو قالب بداية احترافي لأنظمة التوليد المعزز بالاسترجاع (RAG)، يسد الفجوة بين تجارب الـ Notebooks وأنظمة الإنتاج الحقيقية. مبني على مبادئ Clean Architecture، يوفر أساساً متيناً لبناء أنظمة ذكية للأسئلة والأجوبة على المستندات.

### لماذا هذا المشروع؟

معظم شروحات RAG تعرض الأساسيات فقط: تضمين ← تخزين ← بحث ← توليد. لكن أنظمة الإنتاج تحتاج أكثر من ذلك بكثير:

| التحدي | حلّنا |
|--------|-------|
| **مشاكل الاستدعاء** | بحث هجين + **توسيع الاستعلام (Query Expansion)** |
| **مشاكل الدقة** | إعادة ترتيب بـ Cross-Encoder |
| **التحكم بالتكلفة** | تضمين دفعي + تخزين Redis مؤقت |
| **عزل البيانات** | تصميم متعدد المستأجرين |
| **المعالجة المكررة** | تجزئة الملفات + إزالة تكرار القطع |
| **الارتباط بمزود** | نمط المنافذ والمحولات |
| **قابلة التوسع** | فهرسة غير متزامنة مع Celery |
| **الرؤية والمراقبة** | **مقاييس Prometheus** + **واجهة تجريبية Gradio** |

---

## ✨ Features / المميزات

### Core RAG Pipeline / خط أنابيب RAG الأساسي

```
📄 Document Upload
    ↓
📝 Text Extraction (PDF/DOCX/TXT)
    ↓
✂️ Token-Aware Chunking (with overlap)
    ↓
🔢 Batch Embeddings (OpenAI / Local)
    ↓
💾 Dual Storage:
    ├── Qdrant (vectors, minimal payload)
    └── Postgres (text, metadata, FTS)
    ↓
🔍 Hybrid Retrieval:
    ├── Vector Search (semantic)
    ├── Keyword Search (FTS + tsvector)
    └── 🔀 Query Expansion (LLM-based)
    ↓
🔀 RRF Fusion (merge results)
    ↓
🎯 Cross-Encoder Reranking
    ↓
💬 LLM Answer Generation
```

### Production Features / مميزات الإنتاج

| Feature | Description | الوصف |
|---------|-------------|-------|
| 🏗️ **Clean Architecture** | Domain/Application/Adapters separation | فصل المجال/التطبيق/المحولات |
| 🔌 **Ports & Adapters** | Swap providers without code changes | تبديل المزودين بدون تغيير الكود |
| 👥 **Multi-Tenant** | Complete user_id isolation | عزل كامل بمعرف المستخدم |
| ⚡ **Async Indexing** | Celery workers for heavy processing | عمال Celery للمعالجة الثقيلة |
| 📈 **Observability** | Prometheus metrics + Structured logging | مقاييس Prometheus + سجلات منظمة |
| 🎨 **Demo UI** | Built-in Gradio frontend for testing | واجهة Gradio تجريبية للاختبار |
| 🧪 **Eval Script** | Retrieval quality evaluation script | سكربت تقييم جودة الاسترجاع |
| 🔄 **Query Expansion** | Multi-query generation for better recall | توليد استعلامات متعددة لاستدعاء أفضل |

---

## 🚀 Quickstart / البدء السريع

### Prerequisites / المتطلبات الأساسية

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key (or Ollama for local LLM)

### 1. Clone & Setup / الاستنساخ والإعداد

```bash
# Clone the repository
git clone https://github.com/your-org/rag-engine-mini.git
cd rag-engine-mini

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (including Gradio)
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Infrastructure / تشغيل البنية التحتية

```bash
# Start Postgres + Redis + Qdrant
make docker-up

# Run database migrations
make migrate

# Seed demo user
make seed
```

### 3. Run the Application / تشغيل التطبيق

```bash
# Terminal 1: API Server
make run

# Terminal 2: Celery Worker
make worker

# Terminal 3: Demo UI (Optional)
make demo
```

### 4. Verify Installation / التحقق من التثبيت

```bash
# Health & Metrics
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

---

## 📚 End-to-End Example / مثال متكامل

### Step 1: Upload a Document / رفع مستند

```bash
# Upload a PDF document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "X-API-KEY: demo_api_key_12345678" \
  -F "file=@./sample.pdf"
```

**Response / الاستجابة:**
```json
{
  "document_id": "d7f3a1b2-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
  "status": "queued"
}
```

### Step 2: Check Indexing Status / التحقق من حالة الفهرسة

```bash
curl -X GET "http://localhost:8000/api/v1/documents/d7f3a1b2.../status" \
  -H "X-API-KEY: demo_api_key_12345678"
```

**Response / الاستجابة:**
```json
{
  "document_id": "d7f3a1b2-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
  "status": "indexed",
  "chunks_count": 42
}
```

### Step 3: Ask a Question (Hybrid Search) / طرح سؤال (بحث هجين)

```bash
curl -X POST "http://localhost:8000/api/v1/queries/ask-hybrid" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: demo_api_key_12345678" \
  -d '{
    "question": "What are the main objectives of this project?",
    "k_vec": 30,
    "k_kw": 30,
    "rerank_top_n": 8
  }'
```

**Response / الاستجابة:**
```json
{
  "answer": "The main objectives of this project are...",
  "sources": [
    "chunk_a1b2c3d4",
    "chunk_e5f6g7h8",
    "chunk_i9j0k1l2"
  ]
}
```

### Step 4: Document-Filtered Search (ChatPDF Mode) / بحث مقيد بمستند

```bash
curl -X POST "http://localhost:8000/api/v1/queries/ask-hybrid" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: demo_api_key_12345678" \
  -d '{
    "question": "Summarize section 3",
    "document_id": "d7f3a1b2-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
    "k_vec": 20,
    "k_kw": 20
  }'
```

---

## 🏗️ Architecture / المعمارية

### Clean Architecture Layers / طبقات المعمارية النظيفة

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│                    Thin controllers + DTOs                   │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  Use Cases   │  │     Ports       │  │   Services    │  │
│  │  (Orchestr.) │  │  (Interfaces)   │  │ (Pure Logic)  │  │
│  └──────────────┘  └─────────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│              Entities + Domain Rules (No I/O)                │
├─────────────────────────────────────────────────────────────┤
│                     Adapters Layer                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │  OpenAI  │ │  Qdrant  │ │ Postgres │ │    Redis     │   │
│  │   LLM    │ │  Vector  │ │   Repo   │ │    Cache     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow / تدفق البيانات

```
                    ┌─────────────────┐
                    │   User Request  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI Route │
                    │  (Thin Controller)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Use Case     │
                    │ (Orchestration) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌─────▼─────┐       ┌────▼────┐
    │ Vector  │        │  Keyword  │       │  Text   │
    │ Search  │        │   Search  │       │ Hydrate │
    │ (Qdrant)│        │ (Postgres)│       │  (DB)   │
    └────┬────┘        └─────┬─────┘       └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │   RRF Fusion    │
                    │ (Merge Results) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Cross-Encoder  │
                    │    Reranking    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  LLM Generation │
                    │    (OpenAI)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Answer      │
                    │   + Sources     │
                    └─────────────────┘
```

### Key Design Decisions / قرارات التصميم الرئيسية

| Decision | Rationale | القرار | السبب |
|----------|-----------|--------|-------|
| **No text in Qdrant** | Reduces storage, easier updates | لا نص في Qdrant | يقلل التخزين، تحديثات أسهل |
| **Postgres FTS over Elasticsearch** | Simpler stack, good enough for most cases | Postgres FTS بدل Elasticsearch | مكدس أبسط، كافٍ لمعظم الحالات |
| **Cross-Encoder local** | No API costs, works offline | Cross-Encoder محلي | بدون تكلفة API، يعمل بدون إنترنت |
| **Chunk dedup per tenant** | Saves storage and embedding costs | إزالة تكرار القطع لكل مستأجر | يوفر التخزين وتكلفة التضمين |
| **Generated tsvector** | Automatic, consistent, GIN-indexed | tsvector مولد تلقائياً | تلقائي، متسق، مفهرس بـ GIN |

---

## ⚙️ Configuration / الإعدادات

### Environment Variables / متغيرات البيئة

```bash
# =============================================================================
# Application / التطبيق
# =============================================================================
APP_NAME=rag-engine-mini
ENV=dev                          # dev | staging | prod
DEBUG=true

# =============================================================================
# Security / الأمان
# =============================================================================
API_KEY_HEADER=X-API-KEY

# =============================================================================
# Database / قاعدة البيانات
# =============================================================================
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# =============================================================================
# Redis / ريديس
# =============================================================================
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# =============================================================================
# Vector Store / مخزن المتجهات
# =============================================================================
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=chunks
EMBEDDING_DIM=1536              # Must match embedding model

# =============================================================================
# LLM Provider / مزود نموذج اللغة
# =============================================================================
LLM_BACKEND=openai              # openai | ollama
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# Ollama (alternative)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text

# =============================================================================
# Reranking / إعادة الترتيب
# =============================================================================
RERANK_BACKEND=cross_encoder    # cross_encoder | none
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CROSS_ENCODER_DEVICE=cpu        # cpu | cuda

# =============================================================================
# File Upload / رفع الملفات
# =============================================================================
UPLOAD_DIR=./uploads
MAX_UPLOAD_MB=20
```

---

## 🧪 Testing / الاختبار

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/test_chunking.py -v

# Run integration tests (requires running services)
pytest tests/integration/ -v --tb=short
```

---

## 📁 Project Structure / هيكل المشروع

```
rag-engine-mini/
├── src/                        # Source code / الكود المصدري
│   ├── core/                   # Config, logging, DI / الإعدادات، السجلات، حقن التبعيات
│   ├── domain/                 # Entities, errors / الكيانات، الأخطاء
│   ├── application/            # Use cases, ports, services / حالات الاستخدام، المنافذ، الخدمات
│   ├── adapters/               # External implementations / التطبيقات الخارجية
│   ├── api/                    # FastAPI routes / مسارات FastAPI
│   └── workers/                # Celery tasks / مهام Celery
├── tests/                      # Test suite / مجموعة الاختبارات
├── docs/                       # Documentation / التوثيق
├── notebooks/                  # Educational notebooks / دفاتر تعليمية
├── scripts/                    # Utility scripts / سكربتات مساعدة
└── docker/                     # Docker configuration / إعدادات Docker
```

See [STRUCTURE.md](./STRUCTURE.md) for detailed file descriptions.

---

## 🛠️ Development Commands / أوامر التطوير

```bash
# Run API server (dev mode)
make run

# Run Celery worker
make worker

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Type check
make typecheck

# Run migrations
make migrate

# Seed demo data
make seed

# Full stack with Docker
make docker-up
make docker-down
```

---

## 🔧 Troubleshooting / استكشاف الأخطاء

### Common Issues / المشاكل الشائعة

<details>
<summary><strong>❌ "Connection refused" to Qdrant/Redis/Postgres</strong></summary>

**Cause:** Services not running  
**Solution:**
```bash
docker compose -f docker/docker-compose.yml up -d
docker compose ps  # Verify all services are running
```
</details>

<details>
<summary><strong>❌ "Invalid API key" error</strong></summary>

**Cause:** No user seeded or wrong API key  
**Solution:**
```bash
python scripts/seed_user.py
# Use the printed API key in X-API-KEY header
```
</details>

<details>
<summary><strong>❌ "No text extracted from file"</strong></summary>

**Cause:** Unsupported file format or corrupted file  
**Solution:**
- Ensure file is PDF, DOCX, or TXT
- Check file is not encrypted/password-protected
- For scanned PDFs, OCR is not implemented (yet)
</details>

<details>
<summary><strong>❌ "CUDA out of memory" with Cross-Encoder</strong></summary>

**Cause:** GPU memory exhausted  
**Solution:**
```bash
# Use CPU instead
CROSS_ENCODER_DEVICE=cpu

# Or reduce batch size in reranker
```
</details>

<details>
<summary><strong>❌ Worker not processing tasks</strong></summary>

**Cause:** Worker not running or wrong queue  
**Solution:**
```bash
# Ensure worker is running with correct queue
celery -A src.workers.celery_app.celery_app worker -Q indexing -l INFO

# Check Redis connection
redis-cli ping
```
</details>

---

## 📚 Documentation / التوثيق

| [workflows.md](./docs/workflows.md) | Key workflows | سير العمليات الرئيسية |
| [contributing.md](./docs/contributing.md) | Contribution guide | دليل المساهمة |
| [deep-dives/](./docs/deep-dives/) | 🧠 Technical Deep Dives | شروحات تقنية عميقة |

---

## 🎓 Learning Center / مركز التعلم

- **[RAG Mastery Roadmap](./docs/ROADMAP.md)**: Your step-by-step learning path.
- **[Architecture Patterns](./docs/deep-dives/architecture-patterns.md)**: The "Why" behind the design.
- **[Visual Architecture Guide](./docs/VISUAL_GUIDE.md)**: Diagrams showing how data flows.
- **[Developer Guide](./docs/developer-guide.md)**: Deep dive for contributors.
- **[Frequently Asked Questions](./docs/FAQ.md)**: Solving common engineering doubts.
- **[Notebooks Index](./notebooks/)**: Step-by-step guides from zero to production.
- **[Architecture Deep-Dive](./docs/architecture.md)**: For those interested in system design.

---

## 🎓 Notebooks / الدفاتر التعليمية

Educational notebooks that import from `src/` instead of reimplementing:

| Notebook | Content | المحتوى |
|----------|---------|--------|
| [01_intro_and_setup.ipynb](./notebooks/01_intro_and_setup.ipynb) | Setup & architecture overview | الإعداد ونظرة على المعمارية |
| [02_end_to_end_rag.ipynb](./notebooks/02_end_to_end_rag.ipynb) | Complete RAG flow walkthrough | شرح تدفق RAG الكامل |
| [03_hybrid_search_and_rerank.ipynb](./notebooks/03_hybrid_search_and_rerank.ipynb) | Hybrid retrieval deep dive | الغوص العميق في الاسترجاع الهجين |
| [04_evaluation_and_monitoring.ipynb](./notebooks/04_evaluation_and_monitoring.ipynb) | Faithfulness & Relevancy | تقييم الدقة والملاءمة |
| [05_agentic_and_graph_rag.ipynb](./notebooks/05_agentic_and_graph_rag.ipynb) | Self-Correction & Graphs | التصحيح الذاتي والرسوم البيانية |

---

## 📄 License / الرخصة

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments / شكر وتقدير

- [mini-RAG](https://github.com/learnwithhasan/mini-rag) - Original inspiration
- [Qdrant](https://qdrant.tech/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Cross-Encoder models
- [FastAPI](https://fastapi.tiangolo.com/) - API framework

---

<div align="center">

**Built with ❤️ for the AI Engineering Community**

*من المجتمع، للمجتمع*

</div>
