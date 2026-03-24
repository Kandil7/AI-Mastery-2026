# LLM Engineering Quick Reference Card

## 🚀 Quick Start Commands

### Setup
```bash
# Create environment
python -m venv llm-env
source llm-env/bin/activate  # Windows: llm-env\Scripts\activate

# Install dependencies
pip install -r requirements-llm-tutorial.txt

# Verify
python -c "import torch; import transformers; print('✓ Ready')"
```

### Load Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
```

### Generate Text
```python
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔧 LoRA Fine-Tuning

### Configuration
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,              # Rank: 8-64
    lora_alpha=32,     # Alpha: 2× rank
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### Training
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=4,
    fp16=True,
    save_steps=100,
    logging_steps=10
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

---

## 🎯 QLoRA (4-bit Quantization)

### Configuration
```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)
```

---

## 📚 RAG Pipeline

### Basic RAG
```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

# Embeddings
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = embedder.encode(documents)

# FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
faiss.normalize_L2(embeddings)
index.add(embeddings)

# BM25
bm25 = BM25Okapi([doc.split() for doc in documents])

# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### Hybrid Search
```python
def hybrid_search(query, query_embedding, top_k=5):
    # Dense search
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    dense_scores, dense_indices = index.search(query_embedding.reshape(1, -1), top_k*2)
    
    # Sparse search
    sparse_scores = bm25.get_scores(query.split())
    sparse_indices = np.argsort(sparse_scores)[::-1][:top_k*2]
    
    # RRF Fusion
    scores = {}
    for rank, idx in enumerate(dense_indices[0]):
        scores[idx] = scores.get(idx, 0) + 1.0 / (rank + 1)
    for rank, idx in enumerate(sparse_indices):
        scores[idx] = scores.get(idx, 0) + 1.0 / (rank + 1)
    
    top_indices = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [documents[i] for i in top_indices]
```

---

## 🇸🇦 Arabic NLP

### Normalization
```python
import re

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)  # Alif
    text = re.sub("ى", "ي", text)        # Alif Maqsura
    text = re.sub("ة", "ه", text)        # Ta Marbuta
    text = re.sub("ؤئ", "ء", text)       # Hamza
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)  # Diacritics
    return text
```

### Arabic Models
```python
# Best Arabic LLMs
ARABIC_MODELS = {
    "Jais-13B": "inceptionai/jais-13b-chat",
    "Jais-30B": "inceptionai/jais-30b-chat",
    "AraBERT": "aubmindlab/bert-base-arabertv2",
    "MARBERT": "UBC-NLP/MARBERTv2",
    "AceGPT": "AceGPT/AceGPT-13B-chat"
}
```

### Dialect Detection
```python
def detect_dialect(text):
    indicators = {
        'egyptian': ['إزك', 'إيه', 'أوي', 'خلاص'],
        'levantine': ['شو', 'كيفك', 'هلق', 'أبدا'],
        'gulf': ['شلونك', 'الحين', 'زين'],
        'maghrebi': ['علاش', 'دابا', 'باش']
    }
    
    for dialect, words in indicators.items():
        if any(word in text for word in words):
            return dialect
    return 'msa'
```

---

## 🏗️ Production Deployment

### vLLM Setup
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)

outputs = llm.generate(prompts, sampling_params)
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install vllm

EXPOSE 8000
CMD ["python3", "-m", "vllm.entrypoints.api_server", \
     "--model", "meta-llama/Meta-Llama-3-8B-Instruct", \
     "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 Evaluation Metrics

### RAG Metrics
```python
def evaluate_rag(question, answer, context, ground_truth=None):
    metrics = {
        'faithfulness': check_faithfulness(answer, context),
        'relevance': check_relevance(question, answer),
        'context_precision': check_context_precision(context, answer),
    }
    
    if ground_truth:
        metrics['context_recall'] = check_context_recall(context, ground_truth)
    
    return metrics
```

### LLM-as-Judge
```python
def llm_judge(question, answer, judge_model):
    prompt = f"""Rate this response (0-1):
Question: {question}
Answer: {answer}
Score:"""
    
    response = judge_model.generate(prompt)
    return float(response.strip())
```

---

## 🔒 Security

### Prompt Injection Detection
```python
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"dan mode",
    r"disregard.*rules",
    r"system prompt"
]

def detect_injection(text):
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
```

### Input Sanitization
```python
import unicodedata

def sanitize_input(text, max_length=4096):
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Strip dangerous characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Length limit
    if len(text) > max_length:
        raise ValueError("Input too long")
    
    return text
```

---

## 💰 Cost Optimization

### Model Routing
```python
def route_query(query):
    if len(query) < 50 and "?" not in query:
        return "llama-3-8b"      # Simple: $0.0001/1K
    elif "reason" in query.lower():
        return "gpt-4-turbo"     # Complex: $0.01/1K
    else:
        return "llama-3-70b"     # Medium: $0.0007/1K
```

### Token Caching
```python
import hashlib
from redis import Redis

cache = Redis()

def cached_generate(prompt, llm):
    key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
    cached = cache.get(key)
    
    if cached:
        return cached.decode()
    
    response = llm.generate(prompt)
    cache.setex(key, 3600, response)  # 1 hour TTL
    return response
```

---

## 🎯 Hyperparameter Guidelines

### LoRA Fine-Tuning
| Parameter | 7B Model | 70B Model | Description |
|-----------|----------|-----------|-------------|
| **r (rank)** | 16-32 | 8-16 | Higher = more expressive |
| **alpha** | 32 | 32 | Scaling factor (α/r) |
| **dropout** | 0.05 | 0.05 | Regularization |
| **learning_rate** | 2e-4 | 2e-4 | Peak learning rate |
| **batch_size** | 8-16 | 4-8 | Per device |
| **epochs** | 3-5 | 3 | Training epochs |

### RAG Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **chunk_size** | 512 | Tokens per chunk |
| **chunk_overlap** | 128 | Overlap between chunks |
| **retrieval_top_k** | 10-20 | Initial retrieval |
| **rerank_top_k** | 3-5 | After reranking |
| **rrf_k** | 60 | RRF constant |

---

## 📁 File Structure

```
AI-Mastery-2026/
├── COMPLETE_LLM_ENGINEERING_TUTORIAL.md  # Main tutorial
├── README_LLM_TUTORIAL.md                 # Getting started
├── TUTORIAL_SUMMARY.md                    # Summary
├── QUICK_REFERENCE.md                     # This file
├── requirements-llm-tutorial.txt          # Dependencies
├── src/
│   ├── arabic/                            # Arabic NLP
│   ├── rag/                               # RAG systems
│   ├── finetuning/                        # Fine-tuning
│   └── ...
└── notebooks/
    ├── 01_arabic_llm_finetuning.ipynb
    └── 02_production_rag_system.ipynb
```

---

## 🔗 Important Links

### Documentation
- Main Tutorial: `COMPLETE_LLM_ENGINEERING_TUTORIAL.md`
- Getting Started: `README_LLM_TUTORIAL.md`
- Summary: `TUTORIAL_SUMMARY.md`

### Notebooks
- Arabic Fine-Tuning: `notebooks/01_arabic_llm_finetuning.ipynb`
- RAG System: `notebooks/02_production_rag_system.ipynb`

### External Resources
- Hugging Face: https://huggingface.co
- LLM University: https://docs.llamaindex.ai/en/stable/
- LangChain Docs: https://python.langchain.com
- vLLM Docs: https://docs.vllm.ai

---

## 📞 Common Issues

### Out of Memory
```python
# Solution: Use QLoRA + gradient accumulation
gradient_accumulation_steps = 8
batch_size = 2  # Effective batch = 16
use_qlora = True
```

### Slow Inference
```python
# Solution: Use vLLM with optimizations
from vllm import LLM
llm = LLM(model="...", enable_prefix_caching=True, gpu_memory_utilization=0.9)
```

### Poor Arabic Performance
```python
# Solution: Use Arabic models + normalization
model_name = "inceptionai/jais-13b-chat"
text = normalize_arabic(text)
```

---

## 🎓 Learning Checklist

### Beginner
- [ ] Understand transformer architecture
- [ ] Load and use Hugging Face models
- [ ] Build simple chatbot
- [ ] Implement basic RAG
- [ ] Use prompt engineering

### Intermediate
- [ ] Fine-tune with LoRA
- [ ] Build production RAG with hybrid search
- [ ] Implement reranking
- [ ] Deploy with vLLM
- [ ] Evaluate LLM performance

### Advanced
- [ ] Fine-tune with QLoRA
- [ ] **Fine-tune Arabic LLMs** ⭐
- [ ] Build multi-agent systems
- [ ] Implement security guardrails
- [ ] Optimize costs
- [ ] Deploy to Kubernetes

---

**Quick Reference Card - Version 1.0 - March 24, 2026**
