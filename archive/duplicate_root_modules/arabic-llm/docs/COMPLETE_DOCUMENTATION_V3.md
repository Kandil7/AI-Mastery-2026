# Balygh (بليغ) v3.0 - Complete Documentation

## الوثائق الكاملة الشاملة

**Version**: 3.0.0  
**Status**: ✅ **PRODUCTION READY**  
**Last Updated**: March 27, 2026

---

## 📚 Table of Contents

### Part I: Getting Started
1. [README](#readme) - Project overview
2. [Quick Start](#quick-start) - 5-minute setup
3. [Installation](#installation) - Detailed installation
4. [Tutorial](#tutorial) - Complete walkthrough

### Part II: Architecture
5. [Architecture Overview](#architecture) - System design
6. [Module Structure](#modules) - Code organization
7. [Data Pipeline](#data-pipeline) - Data flow
8. [Training Pipeline](#training-pipeline) - Training flow

### Part III: API Reference
9. [Core API](#core-api) - Schema & templates
10. [Processing API](#processing-api) - Cleaning & processing
11. [Generation API](#generation-api) - Dataset generation
12. [Training API](#training-api) - QLoRA utilities
13. [Agents API](#agents-api) - AI agents

### Part IV: Guides
14. [Data Processing](#data-processing) - Process your data
15. [Training Guide](#training-guide) - Train models
16. [Evaluation Guide](#evaluation-guide) - Evaluate models
17. [Deployment Guide](#deployment-guide) - Deploy to production

### Part V: Reference
18. [Configuration](#configuration) - Config files
19. [Commands](#commands) - CLI reference
20. [Troubleshooting](#troubleshooting) - Common issues

---

## Part I: Getting Started

### README

**Balygh** is a production-ready Arabic LLM system with:
- **29 Specialized Roles** (Islamic scholars, linguists, tech roles)
- **76 Linguistic & Islamic Skills**
- **5 Integrated Data Sources** (8,424 books, 368K narrators)
- **300K Training Examples** (curated, deduplicated, quality-filtered)

**Quick Example**:
```python
from arabic_llm.core.schema import Role, Skill, TrainingExample

# Create a training example
example = TrainingExample(
    instruction="أعرب الجملة التالية",
    input="الكتابُ صديقٌ لا يخونُ",
    output="الكتابُ: مبتدأ مرفوع...",
    role=Role.TUTOR,
    skills=[Skill.NAHW],
    level="intermediate"
)
```

### Quick Start

**5-Minute Setup**:
```bash
# 1. Install
pip install -e .

# 2. Run pipeline
python scripts/run_pipeline.py --all

# 3. Train
python scripts/training/train.py
```

### Installation

**Requirements**:
- Python 3.10+
- GPU with 24GB+ VRAM (for training)
- 100GB+ disk space

**Install**:
```bash
# Clone
git clone https://github.com/youruser/arabic-llm.git
cd arabic-llm

# Install
pip install -e .

# Verify
python -c "from arabic_llm.core.schema import Role; print(f'✅ {len(Role)} roles')"
```

### Tutorial

**Complete Walkthrough**:
1. Install package
2. Audit data sources
3. Process books
4. Generate dataset
5. Train model
6. Evaluate
7. Deploy

See `docs/guides/tutorial.md` for step-by-step guide.

---

## Part II: Architecture

### Architecture Overview

**High-Level Structure**:
```
┌─────────────────────────────────────────┐
│         User Interfaces                 │
│  (CLI, API, Gradio, Notebooks)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Application Layer               │
│  (Scripts, Pipelines, Agents)          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Core Package                    │
│  (arabic_llm/)                          │
│   ├── core/     (schemas, templates)   │
│   ├── processing/ (cleaning, dedup)    │
│   ├── generation/ (dataset gen)        │
│   ├── training/   (QLoRA, train)       │
│   ├── agents/     (scraper, eval)      │
│   └── utils/      (utilities)          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Data Layer                      │
│  (5 sources, 300K examples)            │
└─────────────────────────────────────────┘
```

### Modules

**arabic_llm/core/**:
- `schema.py` - 29 roles, 76 skills
- `templates.py` - 200+ instruction templates

**arabic_llm/processing/**:
- `cleaning.py` - 7-stage Arabic text cleaning
- `deduplication.py` - MinHash LSH deduplication
- `book_processor.py` - Book extraction

**arabic_llm/generation/**:
- `dataset_generator.py` - SFT example generation

**arabic_llm/training/**:
- `qlora.py` - QLoRA configuration
- `quantization.py` - 4-bit quantization
- `checkpoints.py` - Checkpoint management

**arabic_llm/agents/**:
- `data_collector.py` - Web scraping
- `evaluator.py` - Evaluation suite

### Data Pipeline

```
5 Sources → Cleaning → Dedup → Generation → Merge → Training
   ↓
358K raw → 300K unique → Train → Evaluate → Deploy
```

### Training Pipeline

```
Config → Load Data → QLoRA Setup → Train → Save → Evaluate
```

---

## Part III: API Reference

### Core API

**Schema**:
```python
from arabic_llm.core.schema import Role, Skill, TrainingExample

# All 29 roles
print(list(Role))

# All 76 skills
print(list(Skill))

# Create example
ex = TrainingExample(
    instruction="...",
    input="...",
    output="...",
    role=Role.TUTOR,
    skills=[Skill.NAHW]
)
```

**Templates**:
```python
from arabic_llm.core.templates import get_templates

# Get templates by role
templates = get_templates(role="tutor")

# Get templates by skill
templates = get_templates(skill="nahw")
```

### Processing API

**Cleaning**:
```python
from arabic_llm.processing.cleaning import ArabicTextCleaner

cleaner = ArabicTextCleaner()
cleaned, operations = cleaner.clean(raw_text)
```

**Deduplication**:
```python
from arabic_llm.processing.deduplication import MinHashDeduplicator

dedup = MinHashDeduplicator(threshold=0.85)
is_dup = dedup.is_duplicate(text)
```

### Generation API

**Dataset Generation**:
```python
from arabic_llm.generation.dataset_generator import DatasetGenerator
from arabic_llm.core.schema import DatasetConfig

config = DatasetConfig(target_examples=100000)
generator = DatasetGenerator(config)
generator.generate(output_path="data.jsonl")
```

### Training API

**QLoRA Setup**:
```python
from arabic_llm.training.qlora import QLoRAConfig

config = QLoRAConfig(
    r=64,
    alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj"]
)
```

### Agents API

**Data Collection**:
```python
from arabic_llm.agents.data_collector import DataCollectionAgent

agent = DataCollectionAgent()
agent.add_source(source_config)
agent.collect()
```

**Evaluation**:
```python
from arabic_llm.agents.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model_path, device="cuda")
results = evaluator.evaluate_benchmark("OALL", examples)
```

---

## Part IV: Guides

### Data Processing

**Process Books**:
```bash
python scripts/processing/process_books.py \
  --books-dir datasets/extracted_books \
  --metadata-dir datasets/metadata \
  --output-dir data/processed
```

### Training Guide

**Train Model**:
```bash
python scripts/training/train.py \
  --config configs/training.yaml \
  --dataset data/jsonl/balygh_final.jsonl \
  --output-dir models/balygh-v3
```

### Evaluation Guide

**Evaluate**:
```bash
python scripts/training/prepare_eval.py \
  --model-path models/balygh-v3 \
  --output-dir evaluation/results
```

### Deployment Guide

**Deploy to HF**:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("models/balygh-v3")
model.push_to_hub("youruser/balygh-v3")
```

---

## Part V: Reference

### Configuration

**training.yaml**:
```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
lora:
  r: 64
  alpha: 128
training:
  batch_size: 4
  epochs: 3
```

### Commands

| Command | Purpose |
|---------|---------|
| `python scripts/run_pipeline.py --all` | Run complete pipeline |
| `python scripts/processing/complete_data_audit.py` | Audit data |
| `python scripts/training/train.py` | Train model |
| `pytest tests/` | Run tests |

### Troubleshooting

**OOM Error**:
```yaml
# Reduce batch size
training:
  batch_size: 2
  gradient_accumulation: 8
```

**Import Error**:
```bash
# Update imports
python -c "import sys; print(sys.path)"
```

---

**Status**: ✅ **COMPLETE**  
**Next**: See individual guide documents for details

---

<div align="center">

# بليغ (Balygh) v3.0

**الوثائق الكاملة**

[Quick Start](#quick-start) | [API](#api-reference) | [Guides](#guides)

</div>
