# Balygh (بليغ) - User Guide

## دليل المستخدم الشامل

**Version**: 3.0.0  
**Purpose**: Complete user guide for Balygh v3.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Processing](#data-processing)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)
8. [FAQ](#faq)

---

## Introduction

### What is Balygh?

Balygh (بليغ) is a specialized Arabic LLM system designed for:
- **Islamic Sciences** (Fiqh, Hadith, Tafsir)
- **Arabic Linguistics** (Nahw, Balagha, Sarf)
- **RAG & Tool Use** (Grounded responses with citations)

### Key Features

- ✅ 29 specialized roles
- ✅ 76 linguistic & Islamic skills
- ✅ 5 integrated data sources
- ✅ 300K training examples
- ✅ Production-ready pipeline

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU VRAM | 16 GB | 24 GB |
| Disk Space | 50 GB | 100 GB |
| Python | 3.10 | 3.11+ |

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/youruser/arabic-llm.git
cd arabic-llm
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Using conda
conda create -n arabic-llm python=3.11
conda activate arabic-llm
```

### Step 3: Install Dependencies

```bash
# Install package
pip install -e .

# Install dev dependencies (optional)
pip install -e ".[dev]"

# Verify installation
python -c "from arabic_llm.core.schema import Role; print(f'✅ {len(Role)} roles')"
```

### Step 4: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Check GPU (if available)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Run quick test
pytest tests/test_schema.py -v
```

---

## Quick Start

### 5-Minute Pipeline

```bash
# 1. Audit data sources (5 min)
python scripts/processing/complete_data_audit.py

# 2. Process all sources (60 min)
python scripts/run_pipeline.py --process

# 3. Merge datasets (10 min)
python scripts/utilities/merge_all_datasets.py

# 4. Train model (36 hours)
python scripts/training/train.py
```

### One-Command Pipeline

```bash
python scripts/run_pipeline.py --all
```

---

## Data Processing

### Step 1: Audit Data Sources

```bash
python scripts/processing/complete_data_audit.py
```

**Output**:
```
✅ Arabic Web: found (10 GB)
✅ Extracted Books: found (16.4 GB, 8,424 books)
✅ Metadata: found (8,424 entries)
✅ Sanadset: found (368K narrators)
✅ System Books: found (5 DBs)
```

### Step 2: Process Each Source

**Arabic Web**:
```bash
python scripts/processing/process_arabic_web.py
```

**Extracted Books**:
```bash
python scripts/processing/process_books.py
```

**Sanadset Hadith**:
```bash
python scripts/processing/process_sanadset.py
```

### Step 3: Generate Dataset

```bash
python scripts/generation/build_balygh_sft.py --target-examples 300000
```

### Step 4: Refine with LLM (Optional)

```bash
export DEEPSEEK_API_KEY="sk-..."
python scripts/generation/refine_with_llm.py
```

### Step 5: Merge & Deduplicate

```bash
python scripts/utilities/merge_all_datasets.py
```

**Output**: `data/jsonl/balygh_final_sft.jsonl` (300K examples)

---

## Training

### Step 1: Prepare Configuration

Edit `configs/training.yaml`:

```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "float16"

lora:
  r: 64
  alpha: 128
  dropout: 0.05

training:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2.0e-4
  epochs: 3
  max_seq_length: 4096
```

### Step 2: Start Training

```bash
python scripts/training/train.py \
  --config configs/training.yaml \
  --dataset data/jsonl/balygh_final_sft.jsonl \
  --output-dir models/balygh-v3
```

### Step 3: Monitor Training

**Watch logs**:
```bash
tail -f models/balygh-v3/training.log
```

**Check checkpoints**:
```bash
ls -lh models/balygh-v3/checkpoint-*
```

### Step 4: Training Complete

Expected output:
```
✅ Training complete!
Model saved to: models/balygh-v3
Training time: 36 hours
Final loss: 1.234
```

---

## Evaluation

### Step 1: Prepare Evaluation

```bash
export BALYGH_MODEL_DIR="models/balygh-v3"
```

### Step 2: Run Evaluation

```bash
python scripts/training/prepare_eval.py
```

### Step 3: View Results

```bash
cat evaluation/results/balygh_score.json
```

**Expected Output**:
```json
{
  "balygh_score": 0.78,
  "fiqh_f1": 0.76,
  "hadith_f1": 0.72,
  "nahw_score": 0.82,
  "balagha_score": 0.74
}
```

---

## Deployment

### Option 1: Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_id = "models/balygh-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Push to hub
model.push_to_hub("youruser/balygh-v3")
tokenizer.push_to_hub("youruser/balygh-v3")
```

### Option 2: Gradio Demo

```python
# examples/gradio_demo.py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "models/balygh-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

def generate(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(fn=generate, inputs="text", outputs="text")
iface.launch()
```

**Run**:
```bash
python examples/gradio_demo.py
```

### Option 3: REST API

```python
# deployment/api/fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Prompt(BaseModel):
    text: str
    max_tokens: int = 256

@app.post("/generate")
async def generate(prompt: Prompt):
    # Load model and generate
    return {"response": "Generated text"}
```

**Run**:
```bash
uvicorn deployment.api.fastapi_app:app --reload
```

---

## FAQ

### Q: How long does training take?

**A**: On RTX 3090 (24GB): ~36 hours for 300K examples, 3 epochs.

### Q: Can I train on CPU?

**A**: Technically yes, but extremely slow (weeks). GPU recommended.

### Q: How much VRAM do I need?

**A**: Minimum 16GB, recommended 24GB for 7B model with QLoRA.

### Q: Can I use a different base model?

**A**: Yes! Edit `configs/training.yaml` and change `model.base`.

### Q: How do I add custom roles?

**A**: Edit `arabic_llm/core/schema.py` and add to `Role` enum.

### Q: Where can I get help?

**A**: 
1. Check documentation: `docs/`
2. Review examples: `examples/`
3. Open issue: GitHub
4. Community: Discord/Slack (when available)

---

**Status**: ✅ **COMPLETE**  
**Next**: See specific guides for detailed walkthroughs

---

<div align="center">

# بليغ (Balygh) v3.0

**دليل المستخدم**

[Installation](#installation) | [Quick Start](#quick-start) | [Training](#training)

</div>
