# Fine-Tuning LLMs on Consumer GPUs

## A Practical Guide for Students

**Based on:** FineTuningLLMs by dvgodoy  
**Original Repository:** https://github.com/dvgodoy/FineTuningLLMs  
**License:** MIT License

---

## Table of Contents

1. [Why Fine-Tune?](#1-why-fine-tune)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Quick Start (5-Minute Setup)](#3-quick-start-5-minute-setup)
4. [Step-by-Step Fine-Tuning Workflow](#4-step-by-step-fine-tuning-workflow)
5. [Common Pitfalls & Solutions](#5-common-pitfalls--solutions)
6. [Cost Estimates](#6-cost-estimates)
7. [Next Steps](#7-next-steps)

---

## 1. Why Fine-Tune?

### 1.1 What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained language model and continuing its training on a specific dataset to adapt it to a particular task or domain.

```
Pre-training (General Knowledge) → Fine-tuning (Specialized Knowledge) → Your Custom Model
     Billions of tokens                    Hundreds/thousands of examples        Ready to use!
```

### 1.2 Business Value of Fine-Tuning

| Use Case | Business Value | Example |
|----------|---------------|---------|
| **Customer Support** | Reduce response time, improve accuracy | Auto-respond to common queries |
| **Domain Expertise** | Specialized knowledge in legal, medical, technical | Legal document analysis |
| **Brand Voice** | Consistent tone and style | Marketing content generation |
| **Task Automation** | Automate repetitive text tasks | Report generation, summarization |
| **Data Privacy** | Keep sensitive data on-premises | Internal document processing |

### 1.3 When to Fine-Tune vs Prompt Engineering

```
                    ┌─────────────────────────────────────────┐
                    │         Start with Prompt Engineering   │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │  Does it work well enough?              │
                    └─────────────────┬───────────────────────┘
                          Yes │               │ No
                              │               │
                              │               ▼
                              │    ┌──────────────────────────┐
                              │    │ Consider Fine-Tuning if: │
                              │    │ • Need consistent output │
                              │    │ • Domain-specific knowledge │
                              │    │ • Prompt too long/complex │
                              │    │ • Cost of long prompts high │
                              │    └──────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────────────────┐
                    │         Use Prompt Engineering          │
                    └─────────────────────────────────────────┘
```

### 1.4 Fine-Tuning Success Stories

| Company | Use Case | Result |
|---------|----------|--------|
| **GitHub** | Code completion (Copilot) | 40% of code auto-completed |
| **Jasper** | Marketing content | 10x faster content creation |
| **Harvey AI** | Legal document analysis | 50% time reduction |
| **Midjourney** | Custom style adaptation | Unique artistic outputs |

---

## 2. Hardware Requirements

### 2.1 Minimum Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU** | GTX 1660 (6GB) | RTX 3060 (12GB) | RTX 4090 (24GB) |
| **VRAM** | 6 GB | 12 GB | 24 GB |
| **RAM** | 16 GB | 32 GB | 64 GB |
| **Storage** | 50 GB SSD | 100 GB SSD | 500 GB NVMe |
| **CPU** | 4 cores | 8 cores | 16 cores |

### 2.2 What You Can Fine-Tune by VRAM

| VRAM | Model Size | Quantization | Notes |
|------|------------|--------------|-------|
| **6 GB** | Up to 3B | 4-bit | Entry-level fine-tuning |
| **8 GB** | Up to 7B | 4-bit | Good for learning |
| **12 GB** | Up to 7B | 8-bit / 4-bit | Sweet spot for most |
| **16 GB** | Up to 13B | 8-bit / 4-bit | Colab T4 |
| **24 GB** | Up to 13B | Full / 8-bit | RTX 3090/4090 |
| **40 GB+** | Up to 70B | 8-bit / 4-bit | A100, multi-GPU |

### 2.3 Cloud GPU Options

| Provider | GPU | VRAM | Price/Hour | Best For |
|----------|-----|------|------------|----------|
| **Google Colab Free** | T4 | 16 GB | Free | Learning, small models |
| **Google Colab Pro** | A100 | 40 GB | ~$3/hr | Larger models |
| **RunPod** | RTX 4090 | 24 GB | ~$0.70/hr | Cost-effective |
| **Lambda Labs** | A100 | 80 GB | ~$1.50/hr | Production |
| **AWS (g5.xlarge)** | A10G | 24 GB | ~$1.00/hr | Enterprise |

### 2.4 Memory Calculation

**Quick Formula:**
```
VRAM Needed = Model Size × Precision Multiplier × Overhead

Precision Multipliers:
- Full (FP32): 4 bytes/parameter
- Half (FP16): 2 bytes/parameter
- 8-bit: 1 byte/parameter
- 4-bit: 0.5 bytes/parameter

Overhead: ~20-30% for gradients, optimizer states

Example (7B model, 4-bit):
7,000,000,000 × 0.5 bytes = 3.5 GB
3.5 GB × 1.3 (overhead) = ~4.5 GB VRAM needed
```

---

## 3. Quick Start (5-Minute Setup)

### 3.1 Option A: Google Colab (Recommended for Beginners)

**Step 1:** Open Colab
```
1. Go to https://colab.research.google.com
2. Click "New Notebook"
3. Go to Runtime → Change runtime type → GPU → T4
```

**Step 2:** Install Dependencies
```python
!pip install transformers>=4.37.0
!pip install peft>=0.7.0
!pip install accelerate>=0.25.0
!pip install bitsandbytes>=0.43.0
!pip install datasets>=2.14.0
```

**Step 3:** Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 3.2 Option B: Local Setup

**Step 1:** Create Virtual Environment
```bash
# Windows
python -m venv ft-env
ft-env\Scripts\activate

# Mac/Linux
python3 -m venv ft-env
source ft-env/bin/activate
```

**Step 2:** Install PyTorch with CUDA
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 3:** Install Fine-Tuning Libraries
```bash
pip install transformers peft accelerate bitsandbytes datasets
pip install sentencepiece protobuf
```

**Step 4:** Verify Installation
```python
import torch
import transformers
import peft

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

## 4. Step-by-Step Fine-Tuning Workflow

### Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Choose    │ →  │   Prepare   │ →  │  Configure  │ →  │   Train     │
│   Model     │    │   Data      │    │   LoRA      │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ↓                                                        ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Deploy    │ ←  │   Convert   │ ←  │   Evaluate  │ ←  │   Save      │
│   Model     │    │   to GGUF   │    │   Results   │    │   Adapter   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 4.1 Step 1: Choose Your Model

**Popular Choices for Fine-Tuning:**

| Model | Parameters | VRAM (4-bit) | Best For |
|-------|------------|--------------|----------|
| **TinyLlama-1.1B** | 1.1B | ~2 GB | Learning, testing |
| **Phi-2** | 2.7B | ~3 GB | Reasoning tasks |
| **Mistral-7B** | 7B | ~5 GB | General purpose |
| **Llama-2-7B** | 7B | ~5 GB | General purpose |
| **Llama-2-13B** | 13B | ~9 GB | Higher quality |

**Code: Load Model with Quantization**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "mistralai/Mistral-7B-v0.1"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

### 4.2 Step 2: Prepare Your Dataset

**Dataset Format (Instruction Tuning):**
```json
[
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": "Machine learning is a subset of artificial intelligence..."
    },
    {
        "instruction": "Translate to French",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    }
]
```

**Code: Prepare Dataset**
```python
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("json", data_files="your_dataset.json")

# Or use a pre-built dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

# Format for training
def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    }

dataset = dataset.map(format_example)
```

### 4.3 Step 3: Configure LoRA

**What is LoRA?**
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds small adapter layers to the model, reducing trainable parameters by 100-1000x.

```
Original Model (Frozen)     LoRA Adapters (Trainable)
┌─────────────────────┐     ┌─────────────────────┐
│  Attention Layers   │     │  Small rank matrices │
│  Feed-Forward       │  +  │  (r = 8, 16, 32)    │
│  Embeddings         │     │  ~0.1% of params     │
└─────────────────────┘     └─────────────────────┘
```

**Code: Configure LoRA**
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# LoRA configuration
lora_config = LoraConfig(
    r=16,                    # Rank (higher = more params, better quality)
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Layers to adapt
    lora_dropout=0.05,       # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM",
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 7,241,732,096 || trainable%: 0.0579%
```

### 4.4 Step 4: Train with SFTTrainer

**Code: Training Setup**
```python
from transformers import TrainingArguments
from trl import SFTTrainer

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512,
)

# Start training
trainer.train()
```

### 4.5 Step 5: Test Your Model

**Code: Inference Testing**
```python
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
prompt = "### Instruction:\nExplain quantum computing\n\n### Response:\n"
print(generate_response(prompt))
```

### 4.6 Step 6: Save Your Adapter

**Code: Save LoRA Adapter**
```python
# Save adapter
model.save_pretrained("./my_finetuned_adapter")
tokenizer.save_pretrained("./my_finetuned_adapter")

# Save config
import json
with open("./my_finetuned_adapter/training_info.json", "w") as f:
    json.dump({
        "base_model": model_name,
        "lora_config": lora_config.to_dict(),
        "training_args": training_args.to_dict(),
    }, f, indent=2)
```

### 4.7 Step 7: Convert to GGUF

**Step 7a: Install llama.cpp**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

**Step 7b: Convert Model**
```bash
# First, merge adapter with base model
python merge_adapter.py --base_model mistralai/Mistral-7B-v0.1 \
                        --adapter ./my_finetuned_adapter \
                        --output ./merged_model

# Convert to GGUF
python convert.py ./merged_model --outfile ./model.gguf --outtype f16
```

**Step 7c: Quantize (Optional)**
```bash
# Quantize to 4-bit
./quantize ./model.gguf ./model_q4_0.gguf q4_0
```

### 4.8 Step 8: Deploy Locally with Ollama

**Step 8a: Install Ollama**
```bash
# Mac/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

**Step 8b: Create Modelfile**
```dockerfile
FROM ./model_q4_0.gguf

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

# Set system prompt
SYSTEM """You are a helpful assistant specialized in [your domain].
Provide accurate, concise responses."""
```

**Step 8c: Import and Run**
```bash
# Import model
ollama create my-custom-model -f Modelfile

# Run model
ollama run my-custom-model "What is machine learning?"

# Or use API
curl http://localhost:11434/api/generate -d '{
    "model": "my-custom-model",
    "prompt": "What is machine learning?"
}'
```

---

## 5. Common Pitfalls & Solutions

### 5.1 Out of Memory (OOM) Errors

**Problem:** `CUDA out of memory`

**Solutions:**
```python
# 1. Use lower precision
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)

# 2. Reduce batch size
training_args = TrainingArguments(per_device_train_batch_size=1, ...)

# 3. Use gradient accumulation
training_args = TrainingArguments(gradient_accumulation_steps=8, ...)

# 4. Reduce sequence length
trainer = SFTTrainer(max_seq_length=256, ...)  # Instead of 512

# 5. Enable gradient checkpointing
training_args = TrainingArguments(gradient_checkpointing=True, ...)
```

### 5.2 Poor Model Quality

**Problem:** Model produces low-quality or irrelevant outputs

**Solutions:**
| Issue | Solution |
|-------|----------|
| Overfitting | Add dropout, reduce training epochs, use more data |
| Underfitting | Train longer, increase learning rate, use more epochs |
| Catastrophic forgetting | Use lower learning rate, freeze more layers |
| Inconsistent outputs | Improve data quality, add more examples |

**Debug Checklist:**
```
[ ] Is training loss decreasing?
[ ] Is validation loss similar to training loss?
[ ] Are outputs coherent?
[ ] Does the model follow instructions?
[ ] Have you tested with diverse prompts?
```

### 5.3 Slow Training

**Problem:** Training takes too long

**Solutions:**
```python
# 1. Enable Flash Attention (if supported)
# Install: pip install flash-attn
model.config._attn_implementation = "flash_attention_2"

# 2. Use larger batch sizes (if VRAM allows)
training_args = TrainingArguments(per_device_train_batch_size=8, ...)

# 3. Optimize data loading
training_args = TrainingArguments(dataloader_num_workers=4, ...)

# 4. Use mixed precision
training_args = TrainingArguments(fp16=True, ...)
```

### 5.4 Model Won't Load

**Problem:** Errors loading model or tokenizer

**Solutions:**
```python
# 1. Add trust_remote_code
model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)

# 2. Specify cache directory
model = AutoModelForCausalLM.from_pretrained(..., cache_dir="./cache")

# 3. Clear cache and re-download
# Delete ~/.cache/huggingface and try again

# 4. Check internet connection
# Some models require internet for first load
```

### 5.5 GGUF Conversion Issues

**Problem:** Conversion fails or produces invalid GGUF

**Solutions:**
```bash
# 1. Ensure model is properly merged
python merge_adapter.py --verify

# 2. Use correct conversion script version
# Match llama.cpp version to model format

# 3. Check disk space
df -h  # Ensure enough space for conversion

# 4. Try different quantization
./quantize ./model.gguf ./model_q4_k_m.gguf q4_k_m  # More compatible
```

---

## 6. Cost Estimates

### 6.1 Training Cost Calculator

**Formula:**
```
Cost = GPU Hours × Price/Hour

GPU Hours = (Dataset Size × Epochs) / (Batch Size × Tokens/Second / Sequence Length)
```

**Example Calculation:**
```
Dataset: 1,000 examples
Epochs: 3
Batch Size: 4
Sequence Length: 512
Tokens/Second: 100 (T4 GPU)

GPU Hours = (1000 × 3) / (4 × 100 / 512) = 3,000 / 0.78 = ~1 hour

Cost on Colab Pro ($3/hr): ~$3
Cost on RunPod ($0.70/hr): ~$0.70
```

### 6.2 Cost Comparison Table

| Dataset Size | Epochs | Colab Free | Colab Pro | RunPod | Lambda Labs |
|--------------|--------|------------|-----------|--------|-------------|
| 100 examples | 3 | Free* | ~$0.50 | ~$0.15 | ~$0.30 |
| 1,000 examples | 3 | Free* | ~$3 | ~$0.70 | ~$1.50 |
| 10,000 examples | 3 | Limited | ~$30 | ~$7 | ~$15 |
| 100,000 examples | 3 | Not feasible | ~$300 | ~$70 | ~$150 |

*Colab Free has usage limits (~12 hours/day)

### 6.3 Cost Optimization Tips

1. **Start Small:** Test with 100 examples before full dataset
2. **Use 4-bit:** Reduces VRAM, enables cheaper GPUs
3. **Fewer Epochs:** Often 1-3 epochs is sufficient
4. **Colab Free:** Great for learning and small experiments
5. **Spot Instances:** RunPod/Lambda offer discounted spot pricing

---

## 7. Next Steps

### 7.1 Continue Learning

| Topic | Resource |
|-------|----------|
| Advanced LoRA | [LoRA Paper](https://arxiv.org/abs/2106.09685) |
| RLHF | [InstructGPT Paper](https://arxiv.org/abs/2203.02155) |
| DPO | [DPO Paper](https://arxiv.org/abs/2305.18290) |
| Model Merging | [MergeKit](https://github.com/cg123/mergekit) |

### 7.2 Advanced Techniques

1. **Multi-LoRA:** Train multiple adapters for different tasks
2. **Model Merging:** Combine multiple fine-tuned models
3. **DPO (Direct Preference Optimization):** Align with human preferences
4. **Continual Fine-Tuning:** Incrementally update with new data

### 7.3 Production Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Checklist                      │
├─────────────────────────────────────────────────────────────┤
│ [ ] Model evaluated on test set                             │
│ [ ] Latency meets requirements (<500ms typical)             │
│ [ ] Throughput sufficient for expected load                 │
│ [ ] Error handling implemented                              │
│ [ ] Monitoring and logging configured                       │
│ [ ] API documentation complete                              │
│ [ ] Rate limiting implemented                               │
│ [ ] Backup/recovery plan in place                           │
└─────────────────────────────────────────────────────────────┘
```

### 7.4 Join the Community

- **Hugging Face Discord:** https://hf.co/join/discord
- **r/LocalLLaMA:** https://reddit.com/r/LocalLLaMA
- **FineTuningLLMs Issues:** https://github.com/dvgodoy/FineTuningLLMs/issues

---

## Quick Reference Card

### Essential Commands

```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install core libraries
pip install transformers peft accelerate bitsandbytes datasets trl

# Run training
python train.py --config config.yaml

# Convert to GGUF
python convert.py ./model --outfile ./model.gguf

# Run with Ollama
ollama run my-model "Your prompt here"
```

### Key Parameters

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| `r` (LoRA rank) | 8, 16, 32 | Higher = more params, better quality |
| `learning_rate` | 1e-4 to 2e-4 | Higher = faster but less stable |
| `epochs` | 1-5 | More = better fit, risk overfitting |
| `batch_size` | 1-8 | Higher = faster, more VRAM |
| `sequence_length` | 256-512 | Longer = more context, more VRAM |

---

## Attribution

This guide is based on the FineTuningLLMs repository by dvgodoy:
- **Repository:** https://github.com/dvgodoy/FineTuningLLMs
- **License:** MIT License
- **Book:** "A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face"

Adapted for AI-Mastery-2026 curriculum with additional examples and explanations.

---

*Document Version: 1.0*  
*Created: March 30, 2026*  
*For: AI-Mastery-2026 Students*
