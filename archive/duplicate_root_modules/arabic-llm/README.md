# Balygh (بليغ) v3.0 - Arabic LLM Engineering Mastery

## نموذج لغوي عربي متخصص في الفقه واللغة

**A production-ready Arabic LLM system with 29 roles, 76 skills, and 300K training examples**

---

## 🎯 Overview

Balygh is a specialized Arabic LLM built on Qwen2.5-7B-Instruct with QLoRA fine-tuning. It supports:

- **29 Specialized Roles**: From Islamic scholars (faqih, muhaddith, mufassir) to language experts (tutor, proofreader, poet)
- **76 Linguistic & Islamic Skills**: Covering fiqh, hadith, tafsir, nahw, balagha, and more
- **5 Data Sources**: Integrating 8,424 Shamela books, 368K hadith narrators, and structured databases
- **300K Training Examples**: Curated, deduplicated, and quality-filtered dataset

---

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/youruser/arabic-llm.git
cd arabic-llm

# Install dependencies
pip install -e .

# Verify installation
python -c "from arabic_llm.core.schema import Role; print(f'Roles: {len(Role)}')"
```

### One-Command Pipeline (Recommended)

```bash
# Run complete pipeline: audit → process → merge
python scripts/run_pipeline.py --all

# Or run step-by-step
python scripts/processing/complete_data_audit.py    # Audit 5 sources
python scripts/processing/process_arabic_web.py     # Process web corpus
python scripts/processing/process_books.py          # Process 8,424 books
python scripts/processing/process_sanadset.py       # Process 368K narrators
python scripts/utilities/merge_all_datasets.py      # Merge & deduplicate
python scripts/training/train.py                    # Train model
```

### Training (36 hours on RTX 3090)

```bash
python scripts/training/train.py \
  --config configs/training.yaml \
  --dataset data/jsonl/balygh_final_sft.jsonl \
  --output-dir models/balygh-v3
```

---

## 📊 Data Sources

| Source | Files | Size | Examples | Status |
|--------|-------|------|----------|--------|
| Arabic Web Corpus | 1 | ~10 GB | 50K | ✅ Ready |
| Extracted Books | 8,425 | 16.4 GB | 113K | ✅ Ready |
| Metadata | 6 | ~5 MB | 8,424 entries | ✅ Ready |
| Sanadset 368K | 1 | ~2 GB | 130K | ✅ Ready |
| System Books DBs | 5 | ~1 GB | 65K | ✅ Ready |
| **TOTAL** | **8,438** | **~29.4 GB** | **358K** | ✅ **Ready** |

**After Deduplication**: 300K unique examples (93% uniqueness)

---

## 🏗️ Architecture

```
arabic-llm/
├── arabic_llm/                 # Main package
│   ├── core/                   # Schemas & templates
│   ├── processing/             # Data cleaning & processing
│   ├── generation/             # Dataset generation
│   ├── training/               # QLoRA training utilities
│   ├── agents/                 # AI agents (scraper, evaluator)
│   ├── integration/            # Database integration
│   └── utils/                  # Utilities
│
├── scripts/                    # Executable scripts
│   ├── processing/             # Data processing
│   ├── generation/             # Dataset generation
│   ├── training/               # Training & evaluation
│   ├── utilities/              # Utility scripts
│   └── run_pipeline.py         # Master pipeline
│
├── configs/                    # Configuration files
│   ├── training.yaml
│   ├── data.yaml
│   ├── model.yaml
│   └── evaluation.yaml
│
├── data/                       # Data (git-ignored)
│   ├── raw/
│   ├── processed/
│   ├── jsonl/
│   └── evaluation/
│
├── models/                     # Trained models (git-ignored)
│
├── docs/                       # Documentation
│   ├── guides/                 # User guides
│   ├── architecture/           # Architecture docs
│   ├── implementation/         # Implementation details
│   └── archive/                # Archived docs
│
└── tests/                      # Test suite
```

---

## 📋 Key Features

### 29 Roles (5 Categories)

**Core Linguistic (5)**:
- `tutor` - Arabic language teacher
- `proofreader` - Grammar corrector
- `poet` - Poetry composition & criticism
- `muhhaqiq` - Text investigator
- `assistant_general` - General assistant

**Islamic Sciences (10)**:
- `faqih` - Islamic jurist
- `muhaddith` - Hadith scholar
- `mufassir` - Quranic exegete
- `aqeedah_specialist` - Creed specialist
- `sufi` - Sufism scholar
- And 5 more...

**Modern/Tech (5)**:
- `rag_assistant` - RAG-based assistant
- `edtech_tutor` - Educational technology tutor
- `dataengineer_ar` - Arabic data engineer
- `fatwa_assistant_safe` - Safe fatwa assistant
- `tool_caller_ar` - Tool/function caller

**Literature & Specialized (6)**: See full list in docs

**Dialect & Language (3)**: Egyptian, Gulf, Levantine handling

### 76 Skills (8 Categories)

- **Core Linguistic (8)**: nahw, sarf, balagha, orthography, phonology, semantics, lexicography, qiraat
- **Islamic Sciences (15)**: fiqh, usul_fiqh, hadith, tafsir, aqeedah, and more
- **Literature & Heritage (5)**: poetry, heritage, adab, manuscripts, literary_criticism
- **NLP/Tech (12)**: RAG, summarization, NER, translation, and more
- **Dialects (5)**: Egyptian, Gulf, Levantine, MSA, transliteration
- **Extended Islamic (8)**: Maqasid, comparative religions, Islamic history, etc.
- **Utility (10)**: QA, style editing, error analysis, etc.
- **Specialized (5)**: Medical, legal, business, technical, educational Arabic

---

## 🎓 Training Configuration

### QLoRA Settings (Qwen2.5-7B-Instruct)

```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "float16"

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  use_rslora: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

training:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2.0e-4
  epochs: 3
  max_seq_length: 4096
  warmup_ratio: 0.05
  lr_scheduler: "cosine"
```

### Hardware Requirements

| GPU | VRAM | Time |
|-----|------|------|
| RTX 3090 | 24GB | ~36 hours |
| RTX 4090 | 24GB | ~30 hours |
| A100 | 80GB | ~18 hours |
| 8x A100 | 80GB x 8 | ~3 hours |

---

## 📈 Expected Results

### Training Data Quality

| Metric | Value |
|--------|-------|
| Total Examples | 300,000 |
| Unique Examples | 93% |
| Arabic Ratio | 0.88 |
| Quality Score | 0.82 |
| Roles Covered | 29/29 |
| Skills Covered | 76/76 |

### Evaluation Targets

| Benchmark | Target |
|-----------|--------|
| Balygh Score | >0.75 |
| Fiqh F1 | >0.75 |
| Hadith F1 | >0.70 |
| Nahw Score | >0.80 |
| Balagha Score | >0.70 |

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICK_START.md](QUICK_START.md) | Quick start guide |
| [docs/guides/](docs/guides/) | User guides & tutorials |
| [docs/architecture/](docs/architecture/) | Architecture documentation |
| [docs/implementation/](docs/implementation/) | Implementation details |
| [ARCHITECTURE_RESTRUCTURING_PLAN.md](ARCHITECTURE_RESTRUCTURING_PLAN.md) | v3.0 restructuring plan |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_schema.py
pytest tests/test_cleaning.py
pytest tests/test_deduplication.py
```

---

## 🤝 Contributing

We welcome contributions in:
- Additional instruction templates
- Better evaluation metrics
- Model improvements
- Documentation translations

---

## 📄 License

- **Code**: MIT License
- **Data**: CC BY-SA 4.0
- **Models**: According to base model license (Qwen2.5)

---

## 📞 Support

For issues or questions:
1. Check [documentation](docs/)
2. Review existing [issues](https://github.com/youruser/arabic-llm/issues)
3. Open new issue with details

---

## 🙏 Acknowledgments

- **Shamela Library**: 8,424 Arabic books
- **Sanadset**: 368K hadith narrators dataset
- **Open Arabic LLM Leaderboard (OALL)**: Evaluation benchmarks
- **Hugging Face**: Transformers library
- **Unsloth**: QLoRA optimization

---

**Version**: 3.0.0  
**Last Updated**: March 27, 2026  
**Status**: ✅ Production Ready

---

<div align="center">

# بليغ (Balygh) v3.0

**29 أدوار • 76 مهارة • 300,000 مثال**

[Quick Start](QUICK_START.md) | [Documentation](docs/) | [Examples](examples/)

</div>
