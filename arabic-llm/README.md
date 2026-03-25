# Arabic LLM Fine-Tuning Project

## مشروع تدريب نموذج لغوي عربي متخصص

A comprehensive system for fine-tuning Arabic LLMs to become expert linguists, poets, and language investigators using the Shamela books dataset (8,423 books).

---

## 🎯 Project Goals

This project implements the complete pipeline from the [LLM_Arabic_plan.md](../LLM_Arabic_plan.md) to create an Arabic language model with three core capabilities:

1. **عالم لغوي (Linguistics Scholar)**: Expert in Arabic grammar (نحو), morphology (صرف), and rhetoric (بلاغة)
2. **شاعر (Poet)**: Compose weighted poetry (بحور الخليل) and literary criticism
3. **محقق لغوي (Language Investigator)**: Text verification, error correction, and manuscript analysis

---

## 📚 Dataset

### Source: Shamela Library (المكتبة الشاملة)

- **Total Books**: 8,423 extracted texts
- **Total Size**: ~16.4 GB
- **Authors**: 3,146 scholars
- **Categories**: 41 subject areas
- **Time Period**: Classical to modern Arabic texts

### Key Categories for Linguistic Training

| Category | Books | Relevance |
|----------|-------|-----------|
| كتب اللغة | ~400 | Core linguistics |
| التفسير | 270 | Classical Arabic |
| كتب السنة | 1,226 | Prophetic Arabic |
| الأدب | 415 | Literary Arabic |
| البلاغة | ~100 | Rhetoric science |
| النحو | ~150 | Grammar science |
| الشعر | ~200 | Poetry corpus |

---

## 🏗️ Architecture

```
arabic-llm/
├── src/
│   ├── __init__.py
│   ├── schema.py              # JSONL data schema
│   ├── book_processor.py      # Book extraction & processing
│   ├── instruction_templates.py # Templates for all roles
│   ├── dataset_generator.py   # Generate JSONL datasets
│   └── fine_tuning.py         # QLoRA training scripts
├── configs/
│   ├── training_config.yaml   # Training hyperparameters
│   ├── model_config.yaml      # Base model selection
│   └── data_config.yaml       # Data sampling ratios
├── data/
│   ├── raw/                   # Processed book texts
│   ├── processed/             # Intermediate formats
│   ├── jsonl/                 # Final training datasets
│   └── evaluation/            # Test sets & benchmarks
├── scripts/
│   ├── 01_process_books.py    # Step 1: Process books
│   ├── 02_generate_dataset.py # Step 2: Generate JSONL
│   ├── 03_train_model.py      # Step 3: Fine-tune
│   └── 04_evaluate.py         # Step 4: Evaluate
├── notebooks/
│   └── exploration.ipynb      # Data analysis
└── docs/
    ├── implementation.md      # Detailed implementation
    └── evaluation.md          # Evaluation metrics
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Process Books

```bash
python scripts/01_process_books.py \
  --books-dir ../datasets/extracted_books \
  --metadata-dir ../datasets/metadata \
  --output-dir data/raw
```

### Step 2: Generate Training Dataset

```bash
python scripts/02_generate_dataset.py \
  --input-dir data/raw \
  --output-dir data/jsonl \
  --config configs/data_config.yaml
```

### Step 3: Fine-Tune Model

```bash
python scripts/03_train_model.py \
  --dataset data/jsonl/training_data.jsonl \
  --output-dir models/arabic-linguist-v1 \
  --config configs/training_config.yaml
```

### Step 4: Evaluate

```bash
python scripts/04_evaluate.py \
  --model-path models/arabic-linguist-v1 \
  --test-set data/evaluation/test_set.jsonl
```

---

## 📊 JSONL Schema

Each training example follows this schema:

```json
{
  "id": "ar-tutor-0001",
  "instruction": "أعرب الجملة التالية ثم وضّح الصورة البلاغية فيها",
  "input": "الكتابُ صديقٌ لا يخونُ صاحِبَه.",
  "output": "أولاً: الإعراب: الكتابُ: مبتدأ مرفوع...",
  "role": "tutor",
  "skills": ["nahw", "balagha"],
  "level": "beginner",
  "domain": "education",
  "style": "fusha_classical",
  "task_type": "analysis_and_explanation",
  "difficulty": 1,
  "source": "extracted_books",
  "tags": ["i3rab", "tashbih"]
}
```

### Roles

| Role | Description | Training % |
|------|-------------|------------|
| `tutor` | Language teacher | 40% |
| `proofreader` | Grammar corrector | 25% |
| `poet` | Poetry composition | 15% |
| `muhhaqiq` | Text investigator | 15% |
| `assistant_general` | General assistant | 5% |

### Skills

- `nahw` - Grammar (النحو)
- `sarf` - Morphology (الصرف)
- `balagha` - Rhetoric (البلاغة)
- `orthography` - Spelling (الإملاء)
- `poetry` - Poetry (الشعر)
- `heritage` - Classical texts (التراث)

---

## 🎓 Base Model Selection

### Recommended Models

| Model | Size | Arabic Focus | VRAM (QLoRA) |
|-------|------|--------------|--------------|
| ALLaM-7B-Instruct | 7B | High | 24GB |
| Qwen2.5-7B-Instruct | 7B | Good | 24GB |
| Qwen2.5-8B-Instruct | 8B | Good | 24GB |

### Selection Criteria

For "لغوي عربي" focus: **ALLaM-7B** (trained on 1.2T Arabic tokens)
For general reasoning + Arabic: **Qwen2.5-7B**

---

## 🔧 Training Configuration

### QLoRA Settings (7B Model)

```yaml
# configs/training_config.yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "float16"

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  epochs: 3
  max_seq_length: 2048
  warmup_ratio: 0.03
  lr_scheduler: "cosine"

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
```

---

## 📈 Evaluation

### Benchmarks

1. **Open Arabic LLM Leaderboard (OALL)**
2. **Arabic Benchmarks (ABB)**
3. **Custom Linguistic Tests**:
   - Grammar analysis (إعراب)
   - Rhetoric identification (بلاغة)
   - Poetry composition (شعر)
   - Text correction (تصحيح)

### Test Sets

| Test Set | Examples | Focus |
|----------|----------|-------|
| Grammar | 500 | إعراب، قواعد |
| Rhetoric | 300 | تشبيه، استعارة |
| Poetry | 200 | نظم، نقد |
| Correction | 400 | تصحيح لغوي |
| Heritage | 300 | نصوص تراثية |

---

## 📝 Implementation Details

### Book Processing Pipeline

1. **Load metadata** from `books.json`
2. **Filter by category** (linguistics, literature, etc.)
3. **Extract text segments** suitable for instruction tuning
4. **Apply templates** based on content type
5. **Generate JSONL** with balanced role distribution

### Instruction Templates

Templates are organized by:
- **Role**: tutor, proofreader, poet, muhhaqiq
- **Skill**: nahw, balagha, poetry, etc.
- **Level**: beginner, intermediate, advanced
- **Content Type**: verse, prose, hadith, poetry

Example template for grammar analysis:
```
Instruction: "أعرب الجملة التالية إعراباً مفصلاً"
Input: {sentence_from_book}
Output: {detailed_parse}
```

---

## 🔬 Advanced Features

### 1. Multi-Role Training

The model learns to switch between roles based on instruction context.

### 2. Difficulty Grading

Examples are tagged with difficulty levels for curriculum learning.

### 3. Source Attribution

Each example tracks its source book for quality analysis.

### 4. Synthetic Data Augmentation

LLM-generated examples supplement rare patterns.

---

## 📊 Expected Results

After training on 50K-100K examples:

| Capability | Target Performance |
|------------|-------------------|
| Grammar Analysis | 85%+ accuracy |
| Rhetoric ID | 80%+ accuracy |
| Poetry Quality | Human-evaluated |
| Error Correction | 90%+ precision |

---

## 🛠️ Troubleshooting

### OOM Errors

```bash
# Reduce batch size
--batch_size 2

# Enable gradient checkpointing
--gradient_checkpointing true

# Use CPU offload
--offload_optimizer true
```

### Slow Training

```bash
# Enable Flash Attention 2 (if GPU supports it)
--attn_implementation "flash_attention_2"

# Increase num_workers for data loading
--num_workers 4
```

---

## 📄 License

This project uses:
- **Shamela Books**: Public domain Islamic texts
- **Code**: MIT License
- **Generated Datasets**: CC BY-SA 4.0

---

## 🤝 Contributing

Contributions welcome for:
- Additional instruction templates
- Better evaluation metrics
- Model improvements
- Documentation translations

---

## 📞 Support

For issues or questions:
1. Check existing issues
2. Review documentation
3. Open new issue with details

---

**Version:** 1.0.0  
**Last Updated:** March 25, 2026  
**Status:** Implementation Phase
