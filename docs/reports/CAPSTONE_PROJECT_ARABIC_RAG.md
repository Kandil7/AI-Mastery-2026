# Capstone Project: Production Arabic RAG Chatbot

## Build a Complete Production-Ready Arabic Question Answering System

**Version:** 1.0  
**Last Updated:** March 24, 2026  
**Difficulty:** Advanced  
**Estimated Time:** 15-20 hours

---

## Project Overview

Build a complete, production-ready Arabic RAG (Retrieval-Augmented Generation) chatbot that:

- ✅ Accepts questions in Arabic (MSA and dialects)
- ✅ Retrieves relevant information from Arabic documents
- ✅ Generates accurate answers in Arabic
- ✅ Handles multi-turn conversations
- ✅ Includes evaluation and monitoring
- ✅ Deployed with Docker and Kubernetes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│              (Streamlit Web App / REST API)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Auth       │  │ Rate Limit   │  │  Monitoring  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Arabic RAG Pipeline                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Dialect    │  │   Query      │  │   Hybrid     │         │
│  │   Detector   │  │  Expansion   │  │  Retrieval   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                              │                                  │
│                    ┌─────────┴─────────┐                        │
│                    ▼                   ▼                        │
│            ┌──────────────┐    ┌──────────────┐               │
│            │  Qdrant      │    │  BM25        │               │
│            │  (Dense)     │    │  (Sparse)    │               │
│            └──────────────┘    └──────────────┘               │
│                    │                   │                       │
│                    └─────────┬─────────┘                       │
│                              ▼                                  │
│                    ┌──────────────┐                            │
│                    │  Reranker    │                            │
│                    │  (Cross-Enc) │                            │
│                    └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Arabic LLM Generation                          │
│              (Jais-13B / AraBERT / AceGPT)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Support Services                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Redis   │  │ Postgres │  │Prometheus│  │ Grafana  │       │
│  │  Cache   │  │ Metadata │  │ Metrics  │  │Dashboard │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
capstone-arabic-rag/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # FastAPI routes
│   │   └── middleware.py          # Auth, rate limiting
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py            # Document chunking
│   │   ├── embeddings.py          # Arabic embeddings
│   │   ├── retrieval.py           # Hybrid retrieval
│   │   └── reranking.py           # Cross-encoder reranking
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── arabic_llm.py          # Arabic LLM wrapper
│   │   └── generation.py          # Answer generation
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── normalization.py       # Arabic normalization
│   │   ├── dialect_detection.py   # Dialect identification
│   │   └── query_expansion.py     # Query expansion
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Configuration
│       └── logging.py             # Logging setup
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── model_evaluation.ipynb
│   └── ablation_studies.ipynb
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
├── kubernetes/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── ingress.yaml
├── data/
│   ├── raw/                       # Raw documents
│   ├── processed/                 # Processed chunks
│   └── test_queries/              # Test queries
├── configs/
│   ├── config.yaml                # Main config
│   └── models.yaml                # Model configurations
├── requirements.txt
├── setup.py
├── README.md
└── Makefile
```

---

## Part 1: Data Preparation

### Step 1.1: Load Arabic Documents

```python
# src/data/load_documents.py
from typing import List, Dict
from pathlib import Path
import json

def load_arabic_documents(data_dir: str) -> List[Dict]:
    """
    Load Arabic documents from various sources
    
    Sources:
    - JSON files
    - PDF documents
    - Web scrapes
    - Arabic Wikipedia
    """
    documents = []
    
    # Load from JSON
    json_files = Path(data_dir).glob("*.json")
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for doc in data:
                documents.append({
                    'id': doc.get('id', f"doc_{len(documents)}"),
                    'text': doc['text'],
                    'metadata': doc.get('metadata', {}),
                    'source': str(file)
                })
    
    return documents


def prepare_test_queries() -> List[Dict]:
    """Prepare test queries for evaluation"""
    return [
        {
            'id': 'q1',
            'question': 'ما هو الذكاء الاصطناعي؟',
            'expected_answer': 'الذكاء الاصطناعي هو فرع من علوم الحاسوب...',
            'category': 'definition'
        },
        {
            'id': 'q2',
            'question': 'كيف يعمل التعلم الآلي؟',
            'expected_answer': 'التعلم الآلي يستخدم الخوارزميات...',
            'category': 'process'
        },
        {
            'id': 'q3',
            'question': 'إزك يا باشا؟',  # Egyptian dialect
            'expected_answer': 'أنا بخير، شكراً!',
            'category': 'greeting',
            'dialect': 'egyptian'
        }
    ]
```

### Step 1.2: Arabic Text Normalization

```python
# src/nlp/normalization.py
import re
import unicodedata

class ArabicNormalizer:
    """Normalize Arabic text for consistent processing"""
    
    def __init__(self):
        self.normalization_map = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alif forms
            'ى': 'ي',  # Alif Maqsura
            'ة': 'ه',  # Ta Marbuta
            'ؤ': 'ء', 'ئ': 'ء',  # Hamza forms
        }
    
    def normalize(self, text: str) -> str:
        """Apply all normalizations"""
        # Character normalization
        for old, new in self.normalization_map.items():
            text = text.replace(old, new)
        
        # Remove diacritics
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        return text
```

---

## Part 2: RAG Pipeline Implementation

### Step 2.1: Document Chunking

```python
# src/rag/chunking.py
from typing import List, Dict
from dataclasses import dataclass
import hashlib

@dataclass
class Chunk:
    """Document chunk"""
    id: str
    text: str
    document_id: str
    chunk_index: int
    metadata: Dict

class ArabicChunker:
    """Chunk Arabic documents with token-aware splitting"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, documents: List[Dict]) -> List[Chunk]:
        """Split documents into chunks"""
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            all_chunks.extend(doc_chunks)
        
        return all_chunks
    
    def _chunk_document(self, doc: Dict) -> List[Chunk]:
        """Chunk single document"""
        text = doc['text']
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_text = ' '.join(words[i:i + self.chunk_size])
            
            chunk = Chunk(
                id=f"{doc['id']}_chunk_{len(chunks)}",
                text=chunk_text,
                document_id=doc['id'],
                chunk_index=len(chunks),
                metadata=doc.get('metadata', {})
            )
            chunks.append(chunk)
        
        return chunks
```

### Step 2.2: Hybrid Retrieval

```python
# src/rag/retrieval.py
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple

class HybridRetriever:
    """Hybrid retrieval: dense + sparse"""
    
    def __init__(self, chunks: List[Chunk], embedder):
        self.chunks = chunks
        self.embedder = embedder
        
        # Build indexes
        self._build_dense_index()
        self._build_sparse_index()
    
    def _build_dense_index(self):
        """Build FAISS index for dense retrieval"""
        embeddings = np.array([chunk.embedding for chunk in self.chunks])
        
        dimension = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.dense_index.add(embeddings)
    
    def _build_sparse_index(self):
        """Build BM25 index"""
        tokenized = [chunk.text.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        """Hybrid retrieval with RRF fusion"""
        # Dense retrieval
        query_embedding = self.embedder.encode(query)
        dense_results = self._dense_search(query_embedding, top_k * 2)
        
        # Sparse retrieval
        sparse_results = self._sparse_search(query, top_k * 2)
        
        # RRF Fusion
        fused = self._rrf_fusion(dense_results, sparse_results)
        
        # Return top-k chunks
        return [self.chunks[idx] for idx, _ in fused[:top_k]]
    
    def _rrf_fusion(self, dense: List, sparse: List) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion"""
        scores = {}
        
        for rank, (idx, _) in enumerate(dense):
            scores[idx] = scores.get(idx, 0) + 1.0 / (rank + 1)
        
        for rank, (idx, _) in enumerate(sparse):
            scores[idx] = scores.get(idx, 0) + 1.0 / (rank + 1)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Step 2.3: Reranking

```python
# src/rag/reranking.py
from sentence_transformers import CrossEncoder
from typing import List

class ArabicReranker:
    """Rerank retrieved chunks"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        """Rerank chunks by relevance"""
        if not chunks:
            return []
        
        # Create pairs
        pairs = [[query, chunk.text] for chunk in chunks]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort
        chunk_score_pairs = list(zip(chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in chunk_score_pairs[:top_k]]
```

---

## Part 3: Arabic LLM Integration

### Step 3.1: Arabic LLM Wrapper

```python
# src/llm/arabic_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ArabicLLM:
    """Arabic LLM wrapper"""
    
    def __init__(self, model_name: str = "inceptionai/jais-13b-chat"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Arabic LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ Model loaded")
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.7) -> str:
        """Generate Arabic text"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Part 4: API and Deployment

### Step 4.1: FastAPI Application

```python
# src/api/routes.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(title="Arabic RAG Chatbot API")

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    latency_ms: float

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process query and return answer"""
    start_time = time.time()
    
    try:
        # RAG pipeline
        chunks = retriever.retrieve(request.question, top_k=request.top_k)
        reranked = reranker.rerank(request.question, chunks, top_k=3)
        
        # Build context
        context = "\n".join([chunk.text for chunk in reranked])
        
        # Generate answer
        prompt = f"Based on: {context}\n\nQuestion: {request.question}\n\nAnswer:"
        answer = llm.generate(prompt)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            sources=[chunk.document_id for chunk in reranked],
            confidence=0.8,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 4.2: Docker Configuration

```dockerfile
# docker/Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip git
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy code
COPY src/ ./src/
COPY configs/ ./configs/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health

# Start
CMD ["uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Part 5: Evaluation

### Step 5.1: Evaluation Metrics

```python
# tests/evaluation.py
class CapstoneEvaluator:
    """Evaluate capstone project"""
    
    def __init__(self):
        self.metrics = {
            'retrieval_precision': [],
            'answer_faithfulness': [],
            'answer_relevance': [],
            'latency_p50': [],
            'latency_p95': []
        }
    
    def evaluate_retrieval(self, query: str, retrieved_chunks: List[Chunk],
                          relevant_chunks: List[Chunk]) -> float:
        """Evaluate retrieval precision"""
        retrieved_ids = {c.id for c in retrieved_chunks}
        relevant_ids = {c.id for c in relevant_chunks}
        
        if not relevant_ids:
            return 0.0
        
        precision = len(retrieved_ids & relevant_ids) / len(retrieved_ids)
        return precision
    
    def evaluate_answer(self, question: str, answer: str,
                       context: List[str], ground_truth: str) -> Dict:
        """Evaluate answer quality"""
        # Faithfulness
        faithfulness = self._check_faithfulness(answer, context)
        
        # Relevance
        relevance = self._check_relevance(question, answer)
        
        # Accuracy
        accuracy = self._check_accuracy(answer, ground_truth)
        
        return {
            'faithfulness': faithfulness,
            'relevance': relevance,
            'accuracy': accuracy,
            'overall': (faithfulness + relevance + accuracy) / 3
        }
```

---

## Deliverables

### Required

1. **Working System**
   - Deployed RAG chatbot
   - API endpoints functional
   - Web interface (Streamlit)

2. **Code Repository**
   - Clean, documented code
   - Tests (unit, integration)
   - Docker configuration

3. **Documentation**
   - README with setup instructions
   - Architecture diagram
   - API documentation

4. **Evaluation Report**
   - Performance metrics
   - Quality evaluation
   - Lessons learned

### Bonus

- Kubernetes deployment
- Monitoring dashboard
- Multi-dialect support
- Advanced RAG patterns (Graph RAG, Agentic RAG)

---

## Grading Rubric

| Component | Weight | Criteria |
|-----------|--------|----------|
| **Functionality** | 30% | System works end-to-end |
| **Code Quality** | 20% | Clean, documented, tested |
| **Arabic Support** | 20% | Dialect handling, normalization |
| **Performance** | 15% | Latency, throughput |
| **Documentation** | 15% | Clear, complete |

---

## Getting Started

```bash
# Clone template
git clone <repository-url> capstone-arabic-rag
cd capstone-arabic-rag

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server
uvicorn src.api.routes:app --reload

# Build Docker image
docker build -f docker/Dockerfile -t arabic-rag .

# Deploy with Docker Compose
docker-compose -f docker/docker-compose.yml up
```

---

## Resources

### Datasets
- [OSIAN](https://www.clarin.eu/) - Arabic news articles
- [Arabic Wikipedia](https://ar.wikipedia.org/)
- [OpenAssistant Arabic](https://huggingface.co/datasets/OpenAssistant/oasst1)

### Models
- [Jais-13B](https://huggingface.co/inceptionai/jais-13b-chat)
- [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv2)
- [MARBERT](https://huggingface.co/UBC-NLP/MARBERTv2)

### Tools
- [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools)
- [vLLM](https://github.com/vllm-project/vllm)
- [Qdrant](https://qdrant.tech/)

---

**Good luck with your capstone project! 🚀**
