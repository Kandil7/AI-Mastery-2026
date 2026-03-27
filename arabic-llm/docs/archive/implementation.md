# Arabic LLM Implementation Guide

## دليل التطبيق العملي

A comprehensive step-by-step guide for implementing the Arabic LLM fine-tuning system based on the LLM_Arabic_plan.md specification.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Data Pipeline Details](#data-pipeline-details)
5. [Training Configuration](#training-configuration)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### Project Goals

This implementation creates an Arabic language model with three core capabilities:

| Role | Arabic | Capabilities |
|------|--------|--------------|
| Tutor | معلم اللغة | Grammar (نحو), Rhetoric (بلاغة), Teaching |
| Proofreader | المصحح اللغوي | Error correction, Style editing |
| Poet | الشاعر | Poetry composition, Literary criticism |
| Investigator | المحقق | Text verification, Classical analysis |

### Dataset Source

- **Shamela Library**: 8,423 extracted Arabic books
- **Total Size**: ~16.4 GB of text
- **Categories**: 41 subject areas
- **Authors**: 3,146 scholars

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Arabic LLM Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Step 1     │    │   Step 2     │    │   Step 3     │      │
│  │   Process    │───▶│   Generate   │───▶│   Train      │      │
│  │   Books      │    │   Dataset    │    │   Model      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 8,423 Books  │    │ 50K Examples │    │ QLoRA Model  │      │
│  │   → Segments │    │   → JSONL    │    │   → Weights  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Flow

```
extracted_books/ ──┐
                   ├──▶ BookProcessor ──▶ TextSegments
metadata/ ─────────┘
                              │
                              ▼
                    InstructionTemplates
                              │
                              ▼
                    ExampleGenerator ──▶ TrainingExamples
                              │
                              ▼
                    JSONL Dataset ──▶ QLoRA Training
```

---

## Step-by-Step Implementation

### Step 1: Process Books

**Script**: `scripts/01_process_books.py`

**Purpose**: Extract and segment text from Shamela books.

**Command**:
```bash
cd arabic-llm

python scripts/01_process_books.py \
    --books-dir ../datasets/extracted_books \
    --metadata-dir ../datasets/metadata \
    --output-dir data/raw \
    --max-books 1000
```

**What it does**:
1. Loads book metadata from SQLite/JSON
2. Reads extracted text files
3. Segments text by type (prose, poetry, hadith, verse)
4. Saves processed segments with metadata

**Output**:
- `data/raw/processed_segments.json`: Segmented text
- `data/raw/processing_report.json`: Processing statistics

**Segment Types**:
| Type | Description | Source Categories |
|------|-------------|-------------------|
| prose | Regular text | Most categories |
| poetry | Verse lines | الشعر, الأدب |
| hadith | Prophetic traditions | كتب السنة |
| verse | Quranic verses | التفسير |

---

### Step 2: Generate Dataset

**Script**: `scripts/02_generate_dataset.py`

**Purpose**: Convert segments to instruction-tuning examples.

**Command**:
```bash
python scripts/02_generate_dataset.py \
    --books-dir ../datasets/extracted_books \
    --metadata-dir ../datasets/metadata \
    --input-dir data/raw \
    --output-dir data/jsonl \
    --config configs/data_config.yaml \
    --target-examples 50000
```

**What it does**:
1. Loads processed segments
2. Applies instruction templates based on role/skill
3. Generates examples with proper formatting
4. Creates train/val/test splits

**Role Distribution** (configurable):
```yaml
role_distribution:
  tutor: 0.40          # 20,000 examples
  proofreader: 0.25    # 12,500 examples
  poet: 0.15           # 7,500 examples
  muhhaqiq: 0.15       # 7,500 examples
  assistant_general: 0.05  # 2,500 examples
```

**Output**:
- `data/jsonl/training_data.jsonl`: Full dataset
- `data/jsonl/train.jsonl`: Training split (90%)
- `data/jsonl/val.jsonl`: Validation split (5%)
- `data/jsonl/test.jsonl`: Test split (5%)
- `data/jsonl/generation_report.json`: Statistics

**Example Training Data**:
```json
{
  "id": "ar-tut-a1b2c3d4e5f6",
  "instruction": "أعرب الجملة التالية ثم وضّح الصورة البلاغية فيها",
  "input": "العلمُ نورٌ يبدّدُ ظلماتِ الجهلِ.",
  "output": "أولاً: الإعراب: العلمُ: مبتدأ مرفوع...",
  "role": "tutor",
  "skills": ["nahw", "balagha"],
  "level": "intermediate",
  "domain": "education",
  "difficulty": 2,
  "source": "extracted_books",
  "book_id": 123,
  "book_title": "كتاب البلاغة"
}
```

---

### Step 3: Fine-Tune Model

**Script**: `scripts/03_train_model.py`

**Purpose**: Train model using QLoRA.

**Command**:
```bash
python scripts/03_train_model.py \
    --dataset data/jsonl/train.jsonl \
    --output-dir models/arabic-linguist-v1 \
    --config configs/training_config.yaml
```

**What it does**:
1. Loads base model (Qwen2.5-7B-Instruct)
2. Applies 4-bit quantization (QLoRA)
3. Adds LoRA adapters
4. Trains on instruction dataset

**QLoRA Configuration**:
```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"

lora:
  r: 64              # LoRA rank
  alpha: 128         # Scaling factor
  dropout: 0.05
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

**Hardware Requirements**:
| Model | VRAM (QLoRA) | Training Time |
|-------|--------------|---------------|
| 7B | 24 GB | ~12 hours |
| 8B | 24 GB | ~15 hours |

**Output**:
- `models/arabic-linguist-v1/`: Trained model
- `models/arabic-linguist-v1/training_metrics.json`: Training stats

---

## Data Pipeline Details

### Book Processing Pipeline

```
┌─────────────────┐
│   Book File     │  (e.g., 1_الفواكه_العذاب.txt)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Content Type   │  (poetry/prose/hadith/verse)
│    Detection    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Segmentation  │  (Split into training-ready chunks)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Metadata     │  (book_id, category, author)
│   Attachment    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   TextSegment   │  (Structured output)
└─────────────────┘
```

### Template Application

For each segment, templates are selected based on:

1. **Role**: Which capability to train
2. **Skill**: Which linguistic skill
3. **Content Type**: Poetry vs prose vs hadith
4. **Level**: Difficulty adjustment

**Template Example**:
```python
Template(
    id="tutor_nahw_002",
    role="tutor",
    skill="nahw",
    level="intermediate",
    instruction_template="أعرب الجملة التالية إعراباً مفصلاً: \"{sentence}\"",
    output_format="الإعراب المفصل: [كلمة بكلمة]",
    tags=["i3rab", "detailed_grammar"],
)
```

---

## Training Configuration

### Key Hyperparameters

```yaml
training:
  batch_size: 4           # Per device
  gradient_accumulation: 4 # Effective batch: 16
  learning_rate: 2.0e-4
  epochs: 3
  max_seq_length: 2048
  warmup_ratio: 0.03
  lr_scheduler: "cosine"
  weight_decay: 0.01
```

### Memory Optimization

```yaml
optimization:
  gradient_checkpointing: true  # Save memory
  fp16: true                     # Mixed precision
  num_workers: 4                 # Data loading
```

### Expected Training Metrics

| Epoch | Loss | Learning Rate |
|-------|------|---------------|
| 1 | 2.5 → 1.8 | 2e-4 → 1.8e-4 |
| 2 | 1.8 → 1.2 | 1.8e-4 → 1.2e-4 |
| 3 | 1.2 → 0.8 | 1.2e-4 → 0.4e-4 |

---

## Evaluation

### Automatic Benchmarks

1. **Open Arabic LLM Leaderboard (OALL)**
   - General Arabic understanding
   - Compare with base models

2. **Custom Linguistic Tests**:
   - Grammar analysis accuracy
   - Rhetoric identification
   - Poetry quality metrics

### Manual Evaluation

Create evaluation sets for each role:

```python
# Example evaluation prompt for tutor role
{
    "instruction": "أعرب الجملة التالية",
    "input": "المعلمُ مجتهدٌ",
    "expected_output": "المعلمُ: مبتدأ مرفوع...",
}
```

**Evaluation Metrics**:
- Exact match for grammar terms
- BLEU/ROUGE for explanations
- Human rating for poetry quality

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
```bash
# Reduce batch size
--batch-size 2

# Enable CPU offload
# Edit configs/training_config.yaml:
hardware:
  offload_optimizer: true
  offload_params: true

# Use gradient checkpointing (already enabled)
```

#### 2. Slow Data Loading

**Symptoms**: GPU utilization low, waiting for data

**Solutions**:
```yaml
optimization:
  num_workers: 8  # Increase workers
  pin_memory: true
```

#### 3. Poor Convergence

**Symptoms**: Loss not decreasing

**Solutions**:
```yaml
training:
  learning_rate: 1.0e-4  # Lower learning rate
  warmup_ratio: 0.1      # More warmup
  epochs: 5              # More epochs
```

#### 4. Garbage Output

**Symptoms**: Model produces nonsensical text

**Causes**:
- Data quality issues
- Insufficient training
- Wrong template format

**Solutions**:
1. Validate dataset quality
2. Check template formatting
3. Increase training epochs
4. Verify tokenizer compatibility

---

## Next Steps

After completing the initial training:

1. **Evaluate**: Run evaluation scripts
2. **Iterate**: Adjust based on results
3. **Specialize**: Fine-tune for specific roles
4. **Deploy**: Set up inference API

---

## File Reference

```
arabic-llm/
├── README.md                    # Project overview
├── requirements.txt             # Python dependencies
├── configs/
│   ├── training_config.yaml    # Training hyperparameters
│   └── data_config.yaml        # Data generation config
├── src/
│   ├── __init__.py             # Package exports
│   ├── schema.py               # Data models
│   ├── instruction_templates.py # Templates for all roles
│   ├── book_processor.py       # Book processing
│   └── dataset_generator.py    # Dataset generation
├── scripts/
│   ├── 01_process_books.py     # Step 1
│   ├── 02_generate_dataset.py  # Step 2
│   └── 03_train_model.py       # Step 3
└── docs/
    └── implementation.md       # This file
```

---

**Version**: 1.0.0  
**Last Updated**: March 25, 2026  
**Status**: Ready for Implementation
