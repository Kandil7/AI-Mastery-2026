# RAG System - Complete Usage Examples

**Version**: 1.0.0  
**Date**: March 27, 2026

---

## 📋 Table of Contents

1. [Basic Usage](#basic-usage)
2. [Advanced Queries](#advanced-queries)
3. [Domain Specialists](#domain-specialists)
4. [Agent System](#agent-system)
5. [Indexing Documents](#indexing-documents)
6. [API Usage](#api-usage)
7. [Evaluation](#evaluation)
8. [Monitoring](#monitoring)

---

## Basic Usage

### 1. Quick Start

```python
import asyncio
from rag_system.src.integration import create_islamic_rag

async def main():
    # Create and initialize
    rag = create_islamic_rag()
    await rag.initialize()
    
    # Basic query
    result = await rag.query("ما هو التوحيد في الإسلام؟")
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")

asyncio.run(main())
```

### 2. Configuration

```python
from rag_system.src.integration import IslamicRAGConfig, create_islamic_rag

# Custom configuration
config = IslamicRAGConfig(
    datasets_path="K:/learning/technical/ai-ml/AI-Mastery-2026/datasets",
    output_path="K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data",
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    llm_provider="openai",  # or "anthropic", "ollama", "mock"
    llm_model="gpt-4o",
    chunk_size=512,
    chunk_overlap=50,
    retrieval_top_k=50,
    rerank_top_k=5,
)

rag = create_islamic_rag(config=config)
await rag.initialize()
```

---

## Advanced Queries

### 1. Query with Filters

```python
# Filter by category
result = await rag.query(
    "ما حكم الصلاة؟",
    top_k=5,
    filters={"category": "الفقه العام"}
)

# Filter by author
result = await rag.query(
    "التوحيد",
    filters={"author": "ابن تيمية"}
)
```

### 2. Streaming Response

```python
async for chunk in rag.query_stream("ما هو التوحيد؟"):
    print(chunk, end='', flush=True)
```

### 3. Query with Citations

```python
result = await rag.query("ما أدلة وجود الله؟")

print(f"Answer: {result['answer']}\n")
print("Sources:")
for i, source in enumerate(result['sources'], 1):
    print(f"{i}. {source['book_title']} - {source['author']}")
    print(f"   Category: {source['category']}")
    print(f"   Score: {source['score']:.3f}\n")
```

---

## Domain Specialists

### 1. Tafsir (Quranic Exegesis)

```python
# Query Tafsir specialist
result = await rag.query_tafsir("ما تفسير سورة الإخلاص؟")

print(f"Domain: {result['domain_name']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### 2. Hadith

```python
# Query Hadith specialist
result = await rag.query_hadith("ما حديث إنما الأعمال بالنيات؟")

print(f"Domain: {result['domain_name']}")
print(f"Answer: {result['answer']}")
print(f"Grade: {result.get('hadith_grade', 'Unknown')}")
```

### 3. Fiqh

```python
# Query Fiqh specialist
result = await rag.query_fiqh("ما شروط الصلاة؟")

print(f"Domain: {result['domain_name']}")
print(f"Answer: {result['answer']}")
print(f"Madhhab: {result.get('madhhab', 'General')}")
```

### 4. Custom Domain

```python
# Query as specific domain specialist
result = await rag.query_as_scholar(
    domain="aqeedah",
    question="ما هي أركان الإيمان؟",
    top_k=5
)

print(f"Domain: {result['domain_name']}")
print(f"Answer: {result['answer']}")
```

---

## Comparative Fiqh

### Compare Madhhab Opinions

```python
result = await rag.compare_madhhabs("ما حكم قراءة البسملة في الصلاة؟")

print(f"Question: {result['question']}")
print(f"\nConsensus: {result['consensus']['message']}")

print("\nMadhhab Views:")
for madhhab, madhhab_result in result['madhhab_results'].items():
    if 'error' not in madhhab_result:
        print(f"\n{madhhab.upper()}:")
        print(f"  {madhhab_result['answer'][:200]}...")
        print(f"  Sources: {len(madhhab_result.get('sources', []))}")
```

### Expected Output

```
Question: ما حكم قراءة البسملة في الصلاة؟

Consensus: الإجماع - جميع المذاهب الأربعة على رأي واحد

Madhhab Views:

HANAFI:
  البسملة سنة في أول الفاتحة وسورة...
  Sources: 2

MALIKI:
  البسملة ليست من الفاتحة ولا يجهز بها...
  Sources: 2

SHAFII:
  البسملة آية من الفاتحة وتقرأ جهراً...
  Sources: 2

HANBALI:
  البسملة سنة قبل الفاتحة وسورة...
  Sources: 2
```

---

## Agent System

### 1. Researcher Agent

```python
result = await rag.ask_as_researcher("ما أدلة وجود الله؟")

print(f"Research Results:")
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result.get('sources', []))}")
```

### 2. Student Agent (Learning)

```python
result = await rag.ask_as_student("الصلاة")

print(f"Lesson Plan:")
print(f"Topic: {result.get('topic', 'Unknown')}")
print(f"Objectives: {result.get('objectives', [])}")
print(f"Content: {result.get('content', '')[:200]}...")
print(f"Questions: {result.get('practice_questions', [])}")
```

### 3. Fatwa Research Agent

```python
result = await rag.ask_fatwa("ما حكم الربا؟")

print(f"Fatwa Research:")
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"\n⚠️  Disclaimer: {result.get('disclaimer', '')}")
```

### 4. Custom Agent

```python
# Create specific agent
result = await rag.ask_agent(
    role="historian",
    task="history",
    params={"topic": "الدولة العباسية"}
)

print(f"Historical Analysis:")
print(f"Topic: {result.get('topic', '')}")
print(f"History: {result.get('history', '')[:300]}...")
```

### 5. Multi-Agent Collaboration

```python
result = await rag.collaborate_agents(
    task="ما حكم الزكاة في المال المستفاد؟",
    roles=["researcher", "fatwa", "comparator"]
)

print(f"Collaborative Analysis:")
print(f"Task: {result['task']}")
print(f"\nAgent Results:")
for role, agent_result in result['agent_results'].items():
    print(f"\n{role.upper()}:")
    print(f"  {agent_result.get('answer', '')[:200]}...")

print(f"\nSynthesis:")
print(f"  {result.get('synthesis', '')}")
```

---

## Indexing Documents

### 1. Index with Limit

```python
# Index small sample for testing
await rag.index_documents(limit=10)
print("✅ Indexed 10 documents")
```

### 2. Index by Category

```python
# Index specific categories
await rag.index_documents(
    categories=["التفسير", "كتب السنة", "الفقه العام"],
    limit=100
)
print("✅ Indexed primary sources")
```

### 3. Full Indexing

```python
# Index all documents (may take hours)
print("Starting full indexing...")
await rag.index_documents()
print("✅ Full indexing complete")

# Save indexes
rag._pipeline._save_indexes()
print("✅ Indexes saved")
```

### 4. Indexing with Progress

```python
def progress_callback(current, total):
    percent = (current / total) * 100
    print(f"Progress: {percent:.1f}% ({current}/{total})")

await rag.index_documents(
    limit=1000,
    progress_callback=progress_callback
)
```

---

## API Usage

### 1. Start API Server

```bash
# Development
uvicorn rag_system.src.api.service:app --reload

# Production
uvicorn rag_system.src.api.service:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### 2. Query via HTTP

```bash
# Basic query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما هو التوحيد؟",
    "top_k": 5
  }'

# With filters
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما حكم الصلاة؟",
    "top_k": 5,
    "filters": {"category": "الفقه العام"}
  }'
```

### 3. Streaming Query

```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هو التوحيد؟"}'
```

### 4. Get Statistics

```bash
curl http://localhost:8000/stats
```

### 5. Start Indexing

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"limit": 100}'

# Check status
curl http://localhost:8000/index/status
```

### 6. Python Client

```python
import requests

# Query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"query": "ما هو التوحيد؟", "top_k": 5}
)
result = response.json()

print(f"Answer: {result['answer']}")
print(f"Latency: {result['latency_ms']:.2f}ms")
```

---

## Evaluation

### 1. Run Evaluation

```python
from rag_system.src.evaluation.evaluator import RAGEvaluator, ArabicTestDataset

# Create evaluator
evaluator = RAGEvaluator(pipeline=rag._pipeline)

# Get test samples
samples = ArabicTestDataset.get_samples(
    language="arabic",
    difficulty="easy"
)

# Run evaluation
results = await evaluator.evaluate_dataset(
    samples=samples[:10],
    top_k=5,
    progress_callback=lambda current, total: print(
        f"Progress: {current}/{total}"
    )
)

# Print results
print("\nEvaluation Results:")
print(f"Retrieval Precision@K: {results['retrieval']['precision_at_k']:.3f}")
print(f"Retrieval Recall@K: {results['retrieval']['recall_at_k']:.3f}")
print(f"MRR: {results['retrieval']['mrr']:.3f}")
print(f"Faithfulness: {results['generation']['faithfulness']:.3f}")
print(f"Relevance: {results['generation']['answer_relevance']:.3f}")
```

### 2. Islamic-Specific Metrics

```python
from rag_system.src.evaluation.islamic_metrics import create_islamic_evaluator

evaluator = create_islamic_evaluator(rag._pipeline)

# Evaluate single response
metrics = await evaluator.evaluate_full(
    question="ما هو التوحيد؟",
    answer=result['answer'],
    sources=result['sources']
)

print(f"Source Authenticity: {metrics.source_authenticity:.3f}")
print(f"Evidence Presence: {metrics.evidence_presence:.3f}")
print(f"Citation Quality: {metrics.citation_quality:.3f}")
print(f"Authority Score: {metrics.authority_score:.3f}")

# Generate report
report = evaluator.get_evaluation_report(metrics)
print(report)
```

---

## Monitoring

### 1. Cost Tracking

```python
from rag_system.src.monitoring.monitoring import get_monitor

monitor = get_monitor()

# Log query
monitor.log_query(
    query="ما هو التوحيد؟",
    latency_ms=150,
    tokens_input=100,
    tokens_output=200,
    retrieval_count=5,
    success=True
)

# Check budget
budget = monitor.cost_tracker.check_budget()
print(f"Daily Spent: ${budget['daily']['spent']:.4f}")
print(f"Daily Remaining: ${budget['daily']['remaining']:.4f}")
print(f"Alert: {'⚠️' if budget['daily']['alert'] else '✅'}")
```

### 2. Query Statistics

```python
# Get last 24h statistics
stats = monitor.query_logger.get_statistics(hours=24)

print(f"Queries (24h): {stats['query_count']}")
print(f"Avg Latency: {stats['avg_latency_ms']:.2f}ms")
print(f"Success Rate: {stats['success_rate']:.2%}")
print(f"Total Cost: ${stats['total_cost_usd']:.4f}")
```

### 3. Top Queries

```python
top_queries = monitor.query_logger.get_top_queries(hours=24, limit=10)

print("Top Queries (24h):")
for i, q in enumerate(top_queries, 1):
    print(f"{i}. {q['query']}: {q['count']} times")
```

### 4. Dashboard Metrics

```python
dashboard = monitor.get_dashboard_metrics()

print("Dashboard Metrics:")
print(f"Cost - Daily: ${dashboard['cost']['daily']['spent']:.4f}")
print(f"Queries - Last 24h: {dashboard['queries_last_24h']['query_count']}")
print(f"Top Queries: {len(dashboard['top_queries'])}")
```

---

## Complete Example

```python
"""
Complete RAG System Example
"""

import asyncio
from rag_system.src.integration import create_islamic_rag

async def main():
    print("=" * 70)
    print("ARABIC ISLAMIC RAG SYSTEM - COMPLETE EXAMPLE")
    print("=" * 70)
    
    # Initialize
    print("\n[1] Initializing system...")
    rag = create_islamic_rag()
    await rag.initialize()
    print("✅ System initialized")
    
    # Check stats
    stats = rag.get_stats()
    print(f"📊 Chunks: {stats.get('total_chunks', 0)}")
    print(f"📊 Domains: {len(stats.get('domains_available', []))}")
    print(f"📊 Agents: {len(stats.get('agents_available', []))}")
    
    # Basic query
    print("\n[2] Basic Query...")
    result = await rag.query("ما هو التوحيد في الإسلام؟")
    print(f"✅ Answer: {result['answer'][:100]}...")
    print(f"⏱️  Latency: {result['latency_ms']:.2f}ms")
    
    # Domain specialist
    print("\n[3] Domain Specialist (Tafsir)...")
    result = await rag.query_tafsir("ما تفسير سورة الإخلاص؟")
    print(f"✅ Domain: {result['domain_name']}")
    print(f"✅ Answer: {result['answer'][:100]}...")
    
    # Comparative fiqh
    print("\n[4] Comparative Fiqh...")
    result = await rag.compare_madhhabs("ما حكم القنوت؟")
    print(f"✅ Consensus: {result.get('consensus', {}).get('message', '')}")
    
    # Agent system
    print("\n[5] Agent System (Researcher)...")
    result = await rag.ask_as_researcher("ما أدلة وجود الله؟")
    print(f"✅ Answer: {result['answer'][:100]}...")
    
    # Monitoring
    print("\n[6] Monitoring...")
    from rag_system.src.monitoring.monitoring import get_monitor
    monitor = get_monitor()
    budget = monitor.cost_tracker.check_budget()
    print(f"✅ Daily Cost: ${budget['daily']['spent']:.4f}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Troubleshooting

### No Results

```python
# Check if indexed
stats = rag.get_stats()
if stats.get('total_chunks', 0) == 0:
    print("⚠️  No documents indexed. Run: await rag.index_documents(limit=100)")
```

### Slow Queries

```python
# Reduce top_k
result = await rag.query("السؤال", top_k=3)

# Disable reranking
rag._pipeline.config.enable_reranking = False
```

### Memory Issues

```python
# Reduce batch size
config = IslamicRAGConfig(
    chunk_size=256,  # Smaller chunks
    retrieval_top_k=20  # Fewer results
)
```

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0
