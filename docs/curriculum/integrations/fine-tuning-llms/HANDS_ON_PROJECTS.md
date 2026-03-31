# Hands-On Projects

## Fine-Tuning Track Capstone Projects

**Based on:** FineTuningLLMs by dvgodoy  
**Original Repository:** https://github.com/dvgodoy/FineTuningLLMs  
**License:** MIT License

---

## Project Overview

This document defines three hands-on projects that progressively build fine-tuning skills, from beginner to advanced. Each project incorporates techniques from the FineTuningLLMs repository.

### Project Progression

```
Project 1: Fine-Tune a Chatbot (Beginner)
         ↓
         │ Prerequisites: Modules 1-6
         │ Techniques: QLoRA, Basic SFTTrainer
         │ Time: 4-6 hours
         ↓
Project 2: Domain Adaptation (Intermediate)
         ↓
         │ Prerequisites: Modules 7-10
         │ Techniques: LoRA, Flash Attention, Evaluation
         │ Time: 8-12 hours
         ↓
Project 3: Production Deployment (Advanced)
         ↓
         │ Prerequisites: Modules 11-12
         │ Techniques: GGUF, Ollama, API, Monitoring
         │ Time: 12-16 hours
```

---

## Project 1: Fine-Tune a Chatbot (Beginner)

### Project Description

Create a custom chatbot fine-tuned on your own Q&A pairs. This project introduces the complete fine-tuning workflow with a manageable dataset.

### Learning Objectives

By completing this project, you will:
- ✅ Set up a fine-tuning environment
- ✅ Prepare a custom dataset
- ✅ Configure QLoRA (4-bit quantization + LoRA)
- ✅ Train using SFTTrainer
- ✅ Test and evaluate your chatbot
- ✅ Save and share your adapter

### Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| **GPU** | Any GPU with 6GB+ VRAM or Colab Free |
| **Model** | Phi-2 (2.7B) or TinyLlama (1.1B) |
| **Dataset** | 100-500 custom Q&A pairs |
| **Technique** | QLoRA (4-bit) |
| **Time** | 4-6 hours |

### Dataset Requirements

Create a JSON file with your Q&A pairs:

```json
[
    {
        "instruction": "What is [your topic]?",
        "input": "",
        "output": "[Your answer here]"
    },
    {
        "instruction": "Explain [concept]",
        "input": "",
        "output": "[Your explanation]"
    }
]
```

**Example Domains:**
- Personal knowledge base (your notes, expertise)
- Hobby-specific Q&A (cooking, gaming, sports)
- Study aid (course material, exam prep)
- Customer support (FAQ for a product)

### Implementation Steps

#### Step 1: Environment Setup (30 min)

```python
# Install required packages
!pip install transformers>=4.37.0
!pip install peft>=0.7.0
!pip install accelerate>=0.25.0
!pip install bitsandbytes>=0.43.0
!pip install datasets>=2.14.0
!pip install trl

# Verify setup
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

#### Step 2: Prepare Dataset (45 min)

```python
from datasets import load_dataset
import json

# Load your Q&A dataset
dataset = load_dataset("json", data_files="my_qa_dataset.json")

# Format for training
def format_prompt(example):
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
    return {"text": text}

dataset = dataset.map(format_prompt)
dataset = dataset.train_test_split(test_size=0.1)
```

#### Step 3: Load Model with QLoRA (30 min)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = "microsoft/phi-2"  # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

#### Step 4: Train the Model (1-2 hours)

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./chatbot_results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()
```

#### Step 5: Test Your Chatbot (30 min)

```python
def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with your questions
test_questions = [
    "What is [topic from your dataset]?",
    "Explain [concept from your dataset]",
]

for q in test_questions:
    prompt = f"### Instruction:\n{q}\n\n### Response:\n"
    print(f"Q: {q}")
    print(f"A: {chat(prompt)}")
    print("-" * 50)
```

#### Step 6: Save Your Work (15 min)

```python
# Save adapter
model.save_pretrained("./my_chatbot_adapter")
tokenizer.save_pretrained("./my_chatbot_adapter")

# Create a README
with open("./my_chatbot_adapter/README.md", "w") as f:
    f.write(f"""# My Custom Chatbot

## Model Details
- Base Model: {model_name}
- Technique: QLoRA (4-bit + LoRA)
- Dataset: {len(dataset['train'])} Q&A pairs
- Training: 3 epochs

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
model = PeftModel.from_pretrained(model, "./my_chatbot_adapter")
```

## Example Q&A
[Include 3-5 examples from your dataset]
""")
```

### Deliverables

1. **Code Repository** with:
   - Training script/notebook
   - Dataset (or sample if private)
   - README with usage instructions

2. **Trained Adapter** saved and tested

3. **Demo Video/GIF** (optional) showing chatbot in action

4. **Reflection Document** (1 page):
   - What worked well?
   - What challenges did you face?
   - How would you improve it?

### Evaluation Rubric

| Criterion | Weight | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|-----------|--------|---------------|----------|------------------|----------------------|
| **Dataset Quality** | 20% | 100+ diverse, well-formatted Q&A pairs | 50-100 good pairs | 25-50 basic pairs | <25 or poor quality |
| **Code Quality** | 25% | Clean, well-commented, modular | Good structure | Basic working code | Messy or incomplete |
| **Training Success** | 25% | Model converges, good outputs | Model trains, decent outputs | Model trains, mixed outputs | Training issues |
| **Testing** | 15% | Comprehensive testing with examples | Good testing | Basic testing | Minimal testing |
| **Documentation** | 15% | Excellent README, clear instructions | Good documentation | Basic documentation | Poor documentation |

**Total: 100%**

**Passing Score:** 70% (2.8/4.0 average)

---

## Project 2: Domain Adaptation (Intermediate)

### Project Description

Fine-tune a larger model (7B parameters) on domain-specific documents (medical, legal, technical, etc.) with comprehensive evaluation. This project focuses on adapting models to specialized knowledge domains.

### Learning Objectives

By completing this project, you will:
- ✅ Work with larger models (7B+)
- ✅ Prepare domain-specific datasets
- ✅ Implement advanced LoRA configurations
- ✅ Use Flash Attention for optimization
- ✅ Evaluate model performance quantitatively
- ✅ Compare fine-tuned vs base model

### Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| **GPU** | 12GB+ VRAM or Colab Pro / RunPod |
| **Model** | Mistral-7B or Llama-2-7B |
| **Dataset** | 1,000-5,000 domain documents |
| **Technique** | LoRA + Flash Attention |
| **Time** | 8-12 hours |

### Domain Options

Choose one domain:

| Domain | Dataset Sources | Example Tasks |
|--------|-----------------|---------------|
| **Medical** | PubMed abstracts, medical Q&A | Diagnosis assistance, medical Q&A |
| **Legal** | Legal documents, case summaries | Contract analysis, legal research |
| **Technical** | Documentation, StackOverflow | Code explanation, troubleshooting |
| **Scientific** | Research papers, arXiv | Paper summarization, research Q&A |
| **Financial** | Earnings reports, news | Market analysis, financial Q&A |

### Implementation Steps

#### Step 1: Dataset Collection & Preparation (2 hours)

```python
from datasets import load_dataset, concatenate_datasets
import pandas as pd

# Option A: Load from Hugging Face
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

# Option B: Load custom documents
def load_domain_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), 'r') as f:
            content = f.read()
            documents.append({"text": content, "source": file})
    return documents

# Create instruction-style dataset
def create_instructions(documents):
    instructions = []
    for doc in documents:
        # Generate Q&A from documents
        instructions.append({
            "instruction": "Summarize the following document:",
            "input": doc["text"][:2000],  # Truncate if needed
            "output": f"Summary: [Create summary]"
        })
        instructions.append({
            "instruction": "What are the key points in this document?",
            "input": doc["text"][:2000],
            "output": f"Key points: [Extract key points]"
        })
    return instructions
```

#### Step 2: Model Setup with Flash Attention (1 hour)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "mistralai/Mistral-7B-v0.1"

# Check Flash Attention availability
try:
    from flash_attn import flash_attn_func
    USE_FLASH_ATTN = True
    print("Flash Attention available!")
except ImportError:
    USE_FLASH_ATTN = False
    print("Flash Attention not available, using standard attention")

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2" if USE_FLASH_ATTN else "eager",
)

# Advanced LoRA config
lora_config = LoraConfig(
    r=32,  # Higher rank for better quality
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],  # All linear layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

#### Step 3: Training with Optimization (2-4 hours)

```python
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir="./domain_adapter",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch size = 16
    learning_rate=1e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    optim="paged_adamw_8bit",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    gradient_checkpointing=USE_FLASH_ATTN,  # Enable with Flash Attention
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=1024,  # Longer sequences for documents
    packing=True,  # Pack multiple examples
)

trainer.train()
```

#### Step 4: Comprehensive Evaluation (2 hours)

```python
import evaluate
import torch
from tqdm import tqdm

# Load metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def evaluate_model(model, tokenizer, test_dataset, num_samples=50):
    """Evaluate model on test dataset."""
    predictions = []
    references = []
    
    for i, example in enumerate(tqdm(test_dataset[:num_samples])):
        # Generate prediction
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        prediction = prediction.split("### Response:")[-1].strip()
        
        predictions.append(prediction)
        references.append(example['output'])
    
    # Calculate metrics
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "predictions": predictions[:5],  # Sample predictions
        "references": references[:5],
    }

# Evaluate fine-tuned model
ft_results = evaluate_model(model, tokenizer, dataset["test"])

# Evaluate base model for comparison
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
base_results = evaluate_model(base_model, tokenizer, dataset["test"])

# Compare results
print("=== Evaluation Results ===")
print(f"Fine-tuned ROUGE-L: {ft_results['rougeL']:.4f}")
print(f"Base Model ROUGE-L: {base_results['rougeL']:.4f}")
print(f"Improvement: {(ft_results['rougeL'] - base_results['rougeL']) / base_results['rougeL'] * 100:.1f}%")
```

#### Step 5: Benchmark Comparison (1 hour)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create comparison table
comparison = pd.DataFrame({
    "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
    "Base Model": [
        base_results["rouge1"],
        base_results["rouge2"],
        base_results["rougeL"]
    ],
    "Fine-Tuned": [
        ft_results["rouge1"],
        ft_results["rouge2"],
        ft_results["rougeL"]
    ]
})

comparison["Improvement (%)"] = (
    (comparison["Fine-Tuned"] - comparison["Base Model"]) / 
    comparison["Base Model"] * 100
)

print(comparison.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
x = range(len(comparison))
plt.bar([i - 0.2 for i in x], comparison["Base Model"], 0.4, label="Base Model")
plt.bar([i + 0.2 for i in x], comparison["Fine-Tuned"], 0.4, label="Fine-Tuned")
plt.xticks(x, comparison["Metric"])
plt.ylabel("Score")
plt.title("Model Comparison: Base vs Fine-Tuned")
plt.legend()
plt.savefig("model_comparison.png")
```

### Deliverables

1. **Code Repository** with:
   - Complete training pipeline
   - Evaluation scripts
   - Benchmark comparison

2. **Technical Report** (3-5 pages):
   - Domain description and dataset
   - Training configuration
   - Evaluation results
   - Analysis of improvements

3. **Model Comparison**:
   - Quantitative metrics (ROUGE, BLEU)
   - Qualitative examples
   - Performance charts

4. **Working Demo**:
   - Script to run inference
   - Sample inputs/outputs

### Evaluation Rubric

| Criterion | Weight | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|-----------|--------|---------------|----------|------------------|----------------------|
| **Dataset Quality** | 15% | 1000+ diverse domain documents | 500-1000 good documents | 250-500 basic documents | <250 or poor quality |
| **Implementation** | 25% | Clean, optimized, Flash Attention | Good implementation | Basic working code | Issues with implementation |
| **Evaluation** | 25% | Comprehensive metrics, baseline comparison | Good evaluation | Basic evaluation | Minimal evaluation |
| **Performance Improvement** | 20% | >20% improvement | 10-20% improvement | 5-10% improvement | <5% or no improvement |
| **Documentation** | 15% | Excellent report, clear analysis | Good report | Basic report | Poor documentation |

**Total: 100%**

**Passing Score:** 70% (2.8/4.0 average)

---

## Project 3: Production Deployment (Advanced)

### Project Description

Take your fine-tuned model from Project 2 and deploy it as a production-ready service. This includes GGUF conversion, Ollama deployment, API wrapper, and monitoring.

### Learning Objectives

By completing this project, you will:
- ✅ Convert models to GGUF format
- ✅ Deploy with Ollama
- ✅ Create REST API wrappers
- ✅ Implement monitoring and logging
- ✅ Handle production concerns (rate limiting, error handling)
- ✅ Document for production use

### Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| **GPU** | For conversion (can use CPU for inference) |
| **Model** | Fine-tuned model from Project 2 |
| **Tools** | llama.cpp, Ollama, FastAPI |
| **Time** | 12-16 hours |

### Implementation Steps

#### Step 1: Merge and Convert to GGUF (2 hours)

```python
# merge_and_convert.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import subprocess
import os

def merge_adapter(base_model_name, adapter_path, output_path):
    """Merge LoRA adapter with base model."""
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    print("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    print("Merge complete!")
    return output_path

def convert_to_gguf(model_path, output_name, quantization="q4_0"):
    """Convert to GGUF format using llama.cpp."""
    llama_cpp_path = "/path/to/llama.cpp"
    
    # Convert to GGUF (f16 first)
    print("Converting to GGUF (f16)...")
    subprocess.run([
        "python", f"{llama_cpp_path}/convert.py",
        model_path,
        "--outfile", f"{output_name}.f16.gguf",
        "--outtype", "f16"
    ])
    
    # Quantize
    print(f"Quantizing to {quantization}...")
    subprocess.run([
        f"{llama_cpp_path}/quantize",
        f"{output_name}.f16.gguf",
        f"{output_name}.{quantization}.gguf",
        quantization
    ])
    
    # Clean up
    os.remove(f"{output_name}.f16.gguf")
    
    print(f"GGUF conversion complete: {output_name}.{quantization}.gguf")
    return f"{output_name}.{quantization}.gguf"

# Usage
if __name__ == "__main__":
    merged_path = merge_adapter(
        "mistralai/Mistral-7B-v0.1",
        "./domain_adapter",
        "./merged_domain_model"
    )
    
    gguf_path = convert_to_gguf(
        merged_path,
        "./domain_model",
        "q4_k_m"  # Good quality/size balance
    )
```

#### Step 2: Deploy with Ollama (1 hour)

```dockerfile
# Modelfile
FROM ./domain_model.q4_k_m.gguf

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER num_batch 512

# System prompt for domain expertise
SYSTEM """You are an expert assistant in [YOUR DOMAIN].
Provide accurate, well-reasoned responses based on your training.
If you're unsure about something, acknowledge it.
Keep responses concise but informative."""

# Template (adjust for your model)
TEMPLATE """### Instruction:
{{ .Prompt }}

### Response:
{{ .Response }}"""
```

```bash
# Import model
ollama create domain-expert -f Modelfile

# Test
ollama run domain-expert "Explain [domain concept]"

# Run as server (default on port 11434)
ollama serve
```

#### Step 3: Create API Wrapper (3 hours)

```python
# api/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Domain Expert API",
    description="API for fine-tuned domain expert model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    stream: bool = Field(default=False)

class GenerateResponse(BaseModel):
    response: str
    model: str
    created_at: str
    total_tokens: int

class HealthResponse(BaseModel):
    status: str
    model: str
    timestamp: str

# Ollama client
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "domain-expert"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            
        return HealthResponse(
            status="healthy",
            model=MODEL_NAME,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a response from the model."""
    start_time = datetime.utcnow()
    logger.info(f"Generate request: prompt_length={len(request.prompt)}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": MODEL_NAME,
                "prompt": request.prompt,
                "stream": request.stream,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            }
            
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
        end_time = datetime.utcnow()
        latency = (end_time - start_time).total_seconds()
        
        logger.info(
            f"Generate complete: latency={latency}s, "
            f"tokens={result.get('eval_count', 0)}"
        )
        
        return GenerateResponse(
            response=result.get("response", ""),
            model=MODEL_NAME,
            created_at=start_time.isoformat(),
            total_tokens=result.get("eval_count", 0)
        )
        
    except httpx.TimeoutException:
        logger.error("Request timeout")
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: GenerateRequest):
    """Chat endpoint with conversation history support."""
    # Similar to generate but with chat format
    pass

# Rate limiting (simple implementation)
from collections import defaultdict
from datetime import timedelta

rate_limit_store = defaultdict(list)
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = timedelta(minutes=1)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = datetime.utcnow()
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip]
        if now - t < RATE_WINDOW
    ]
    
    # Check rate limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    rate_limit_store[client_ip].append(now)
    
    return await call_next(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Step 4: Add Monitoring (2 hours)

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Metrics definitions
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'API request latency',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

TOKEN_COUNT = Counter(
    'api_tokens_total',
    'Total tokens processed',
    ['type']  # prompt, completion
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time to load model'
)

ACTIVE_CONNECTIONS = Gauge(
    'api_active_connections',
    'Number of active connections'
)

def monitor_endpoint(endpoint_name):
    """Decorator to monitor endpoint metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time
                REQUEST_COUNT.labels(endpoint=endpoint_name, status=status).inc()
                REQUEST_LATENCY.labels(endpoint=endpoint_name).observe(latency)
        
        return wrapper
    return decorator

# Start metrics server
def start_metrics_server(port=8001):
    start_http_server(port)
    print(f"Metrics server started on port {port}")
    print(f"Metrics available at http://localhost:{port}/metrics")
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - ollama
    volumes:
      - ./logs:/app/logs

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./models:/root/.ollama
      - ollama_data:/root/.ollama

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  ollama_data:
  prometheus_data:
  grafana_data:
```

#### Step 5: Testing & Documentation (2 hours)

```python
# tests/test_api.py
import pytest
import httpx
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_generate():
    """Test generate endpoint."""
    response = client.post(
        "/generate",
        json={
            "prompt": "What is machine learning?",
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert len(response.json()["response"]) > 0

def test_rate_limiting():
    """Test rate limiting."""
    # Make many requests
    for _ in range(100):
        client.post("/generate", json={"prompt": "test", "max_tokens": 10})
    
    # Should be rate limited
    response = client.post("/generate", json={"prompt": "test", "max_tokens": 10})
    assert response.status_code == 429
```

```markdown
# API Documentation

## Domain Expert API

### Endpoints

#### POST /generate
Generate a response from the model.

**Request:**
```json
{
    "prompt": "Your question here",
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
}
```

**Response:**
```json
{
    "response": "Model response here",
    "model": "domain-expert",
    "created_at": "2026-03-30T12:00:00Z",
    "total_tokens": 150
}
```

#### GET /health
Check API health status.

**Response:**
```json
{
    "status": "healthy",
    "model": "domain-expert",
    "timestamp": "2026-03-30T12:00:00Z"
}
```

### Rate Limits
- 100 requests per minute per IP
- 429 Too Many Requests when exceeded

### Error Codes
- 400: Bad Request (invalid input)
- 429: Rate Limit Exceeded
- 500: Internal Server Error
- 503: Service Unavailable
- 504: Gateway Timeout
```

### Deliverables

1. **Deployed Service**:
   - Working API endpoint
   - Ollama deployment
   - Docker Compose configuration

2. **Code Repository**:
   - API code with error handling
   - Monitoring setup
   - Tests

3. **Documentation**:
   - API documentation
   - Deployment guide
   - Operations runbook

4. **Monitoring Dashboard**:
   - Prometheus metrics
   - Grafana dashboard (optional)

### Evaluation Rubric

| Criterion | Weight | Excellent (4) | Good (3) | Satisfactory (2) | Needs Improvement (1) |
|-----------|--------|---------------|----------|------------------|----------------------|
| **Deployment** | 25% | Fully deployed, working API | Deployed with minor issues | Basic deployment | Deployment issues |
| **API Quality** | 25% | Production-ready, error handling | Good API design | Basic API | Poor API design |
| **Monitoring** | 20% | Comprehensive metrics, dashboard | Good monitoring | Basic logging | Minimal monitoring |
| **Documentation** | 15% | Excellent docs, runbook | Good documentation | Basic docs | Poor documentation |
| **Testing** | 15% | Comprehensive tests | Good test coverage | Basic tests | Minimal testing |

**Total: 100%**

**Passing Score:** 70% (2.8/4.0 average)

---

## Project Submission Guidelines

### Repository Structure

```
project-name/
├── README.md              # Project overview
├── requirements.txt       # Dependencies
├── src/
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── inference.py      # Inference script
├── notebooks/
│   └── exploration.ipynb # Exploratory analysis
├── data/
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── models/
│   └── [saved models]
├── tests/
│   └── test_*.py         # Test files
└── docs/
    └── [documentation]
```

### Submission Checklist

```
[ ] Code runs without errors
[ ] All dependencies documented
[ ] README with setup instructions
[ ] Example usage provided
[ ] Tests included
[ ] Documentation complete
[ ] Attribution to FineTuningLLMs included
```

---

## Attribution

These projects are based on techniques and examples from the FineTuningLLMs repository:

- **Original Author:** dvgodoy (Daniel Godoy)
- **Repository:** https://github.com/dvgodoy/FineTuningLLMs
- **License:** MIT License
- **Book:** "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

---

*Document Version: 1.0*  
*Created: March 30, 2026*  
*For: AI-Mastery-2026 Curriculum*
