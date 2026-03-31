# Balygh (بليغ) v3.0 - Architecture Overview

## نظرة عامة على البنية المعمارية

**Version**: 3.0.0  
**Last Updated**: March 27, 2026

---

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                         │
├─────────────────────────────────────────────────────────────────┤
│  • CLI Scripts (scripts/)                                       │
│  • Python API (arabic_llm.*)                                    │
│  • Gradio Demo (deployment/)                                    │
│  • REST API (deployment/api/)                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│   Processing  │   Generation  │   Training    │    Evaluation   │
│   Scripts     │   Scripts     │   Scripts     │    Scripts      │
│               │               │               │                 │
│ • audit       │ • build_sft   │ • train       │ • prepare       │
│ • process_*   │ • refine_llm  │ • evaluate    │ • metrics       │
│ • integrate   │               │               │                 │
└───────────────┴───────────────┴───────────────┴─────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       CORE PACKAGE                              │
│                    (arabic_llm/)                                │
├───────────┬───────────┬───────────┬───────────┬────────────────┤
│   core/   │processing/│generation/│ training/ │   agents/      │
│           │           │           │           │                │
│ • schema  │ • cleaning│ • dataset │ • qlora   │ • data_        │
│ •templates│ • dedup   │ _generator│ • quantiz │   collector    │
│           │ • books   │           │ • checkpt │ • evaluator    │
└───────────┴───────────┴───────────┴───────────┴────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INTEGRATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  • databases.py (SQLite, PostgreSQL)                            │
│  • system_books.py (Book management)                            │
│  • utils/ (Arabic text processing, I/O, logging)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
├───────────────┬───────────────┬───────────────┬─────────────────┤
│  5 Data       │  Processed    │  JSONL        │  Evaluation     │
│  Sources      │  Data         │  Datasets     │  Sets           │
│               │               │               │                 │
│ • arabic_web  │ • cleaned     │ • training    │ • fiqh_eval    │
│ • books       │ • deduped     │ • validation  │ • nahw_eval    │
│ • metadata    │ • segmented   │ • test        │ • balagha_eval │
│ • sanadset    │               │               │ • scraping_eval│
│ • system_dbs  │               │               │                │
└───────────────┴───────────────┴───────────────┴─────────────────┘
```

---

## 🗂️ Module Breakdown

### 1. **Core Module** (`arabic_llm/core/`)

**Purpose**: Define data schemas and instruction templates

**Files**:
- `schema.py` (866 lines)
  - 29 roles across 5 categories
  - 76 skills across 8 categories
  - TrainingExample dataclass
  - DatasetConfig with role distributions
  - Validation functions

- `templates.py` (1,180+ lines)
  - 200+ instruction templates
  - Role-specific templates
  - Skill-specific templates
  - Level-based variations

**Key Classes**:
```python
class Role(Enum):
    TUTOR, PROOFREADER, POET, MUHHAQIQ, ASSISTANT_GENERAL,
    FAQIH, MUHADDITH, MUFASSIR, AQEEDAH_SPECIALIST, SUFI,
    HISTORIAN, GENEALOGIST, GEOGRAPHER, PHYSICIAN, LOGICIAN,
    RAG_ASSISTANT, EDTECH_TUTOR, DATAENGINEER_AR,
    FATWA_ASSISTANT_SAFE, TOOL_CALLER_AR, ...

class Skill(Enum):
    NAHW, SARF, BALAGHA, ORTHOGRAPHY, PHONOLOGY, SEMANTICS,
    FIQH, USUL_FIQH, HADITH, HADITH_MUSTALAH, TAFSIR, AQEEDAH,
    RAG_RETRIEVAL, RAG_GROUNDED_ANSWERING, SUMMARIZATION, ...

@dataclass
class TrainingExample:
    instruction: str
    input: str
    output: str
    role: Role
    skills: List[Skill]
    level: Level
    domain: Domain
    ...
```

---

### 2. **Processing Module** (`arabic_llm/processing/`)

**Purpose**: Clean and process raw Arabic text

**Files**:
- `cleaning.py` (910 lines)
  - 7-stage cleaning pipeline
  - ArabicTextCleaner class
  - Quality scoring functions

- `deduplication.py` (550 lines)
  - 3-level deduplication
  - MinHash LSH implementation
  - Exact, near-duplicate, sentence-level

- `book_processor.py`
  - Book extraction
  - Metadata integration
  - Content segmentation

**Pipeline Flow**:
```
Raw Text → Encoding Fix → Unicode NFC → Arabic Norm → 
Control Chars → Whitespace → OCR Fix → Punctuation → Clean Text
```

---

### 3. **Generation Module** (`arabic_llm/generation/`)

**Purpose**: Generate SFT training examples

**Files**:
- `dataset_generator.py`
  - Template-based generation
  - Role-balanced sampling
  - Quality filtering

**Generation Process**:
```
Books → Segment → Apply Templates → Generate Examples → 
Filter → Deduplicate → JSONL Output
```

---

### 4. **Training Module** (`arabic_llm/training/`)

**Purpose**: QLoRA training utilities

**Files**:
- `qlora.py`
  - QLoRA configuration
  - Adapter setup
  - Training loop utilities

- `quantization.py`
  - 4-bit quantization
  - BnB configuration
  - Memory optimization

- `checkpoints.py`
  - Checkpoint management
  - Resume training
  - Model export

**Training Configuration**:
```yaml
lora:
  r: 64
  alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj, ...]
  
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  
training:
  batch_size: 4
  gradient_accumulation: 4
  learning_rate: 2e-4
  epochs: 3
```

---

### 5. **Agents Module** (`arabic_llm/agents/`)

**Purpose**: AI agents for data collection and evaluation

**Files**:
- `data_collector.py` (700 lines)
  - Web scraping agent
  - Rate limiting
  - HTML parsing

- `evaluator.py` (800 lines)
  - Balygh score computation
  - OALL benchmarks
  - Custom linguistics tests

**Agent Architecture**:
```
User Request → Planner Agent → Scraper Agent → 
Preprocessing Agent → Formatter Agent → Response
```

---

### 6. **Integration Module** (`arabic_llm/integration/`)

**Purpose**: Database and system integration

**Files**:
- `databases.py`
  - SQLite connections
  - Query utilities
  - ORM helpers

- `system_books.py`
  - Book management
  - Metadata integration
  - Cross-referencing

---

### 7. **Utils Module** (`arabic_llm/utils/`)

**Purpose**: Utility functions

**Files**:
- `arabic.py` - Arabic text utilities
- `io.py` - I/O utilities
- `logging.py` - Logging setup
- `text.py` - Text processing utilities

---

## 📊 Data Flow

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 5 DATA SOURCES (29.4 GB, 495K items)                       │
├──────────────┬──────────────┬──────────────┬───────────────┤
│ arabic_web   │extracted_    │  metadata    │  Sanadset     │
│ (10 GB)      │books         │  (6 files)   │  (2 GB)       │
│              │(16.4 GB)     │              │               │
└──────────────┴──────────────┴──────────────┴───────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PROCESSING SCRIPTS                                         │
│ • complete_data_audit.py                                   │
│ • process_arabic_web.py                                    │
│ • process_books.py                                         │
│ • process_sanadset.py                                      │
│ • integrate_datasets.py                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ CORE PROCESSING                                            │
│ • 7-stage cleaning                                         │
│ • 3-level deduplication                                    │
│ • Quality filtering                                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ GENERATION                                                 │
│ • Template-based example generation                        │
│ • Role-balanced sampling                                   │
│ • Output: 358K raw examples                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ MERGE & DEDUPLICATE                                        │
│ • merge_all_datasets.py                                    │
│ • MinHash LSH (threshold=0.85)                            │
│ • Output: 300K unique examples                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ TRAINING                                                   │
│ • QLoRA fine-tuning                                        │
│ • Qwen2.5-7B-Instruct                                      │
│ • 3 epochs, ~36 hours                                      │
│ • Output: balygh-v3 checkpoint                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ EVALUATION                                                 │
│ • balygh_score computation                                 │
│ • OALL benchmarks                                          │
│ • Custom tests                                             │
│ • Output: Evaluation report                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ DEPLOYMENT                                                 │
│ • Hugging Face publishing                                  │
│ • Gradio demo                                              │
│ • REST API                                                 │
│ • Docker container                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Script Organization

### Processing Scripts (`scripts/processing/`)
- `complete_data_audit.py` - Audit all 5 sources
- `process_arabic_web.py` - Process web corpus
- `process_books.py` - Process extracted books
- `process_sanadset.py` - Process hadith narrators
- `integrate_datasets.py` - Integrate all sources

### Generation Scripts (`scripts/generation/`)
- `build_balygh_sft.py` - Build SFT dataset
- `refine_with_llm.py` - Refine with LLM

### Training Scripts (`scripts/training/`)
- `train.py` - Train model
- `prepare_eval.py` - Prepare evaluation

### Utility Scripts (`scripts/utilities/`)
- `merge_all_datasets.py` - Merge & deduplicate
- `audit_datasets.py` - Dataset auditing

### Master Pipeline (`scripts/run_pipeline.py`)
- Orchestrates complete pipeline

---

## 📁 File Locations Reference

| Component | Path | Lines |
|-----------|------|-------|
| Schema | `arabic_llm/core/schema.py` | 866 |
| Templates | `arabic_llm/core/templates.py` | 1,180+ |
| Cleaning | `arabic_llm/processing/cleaning.py` | 910 |
| Deduplication | `arabic_llm/processing/deduplication.py` | 550 |
| Data Collector | `arabic_llm/agents/data_collector.py` | 700 |
| Evaluator | `arabic_llm/agents/evaluator.py` | 800 |
| QLoRA | `arabic_llm/training/qlora.py` | ~400 |
| Training Config | `configs/training.yaml` | 500 |

---

## 🎯 Key Design Decisions

### 1. **Modular Architecture**
- Separation of concerns (processing, generation, training)
- Clear module boundaries
- Easy to extend and maintain

### 2. **Data-Centric Approach**
- 5 diverse data sources
- Rigorous quality control
- Deduplication at multiple levels

### 3. **Role-Based Training**
- 29 specialized roles
- 76 distinct skills
- Balanced representation

### 4. **Production-Ready**
- Comprehensive error handling
- Logging and monitoring
- Checkpoint management
- Evaluation suite

### 5. **Scalability**
- Parallel processing support
- Memory-efficient deduplication
- Gradient accumulation for large batches

---

## 📈 Performance Characteristics

| Operation | Time | Memory |
|-----------|------|--------|
| Data Audit | 5 min | 1 GB |
| Processing (all sources) | 60 min | 4 GB |
| Merging & Dedup | 10 min | 8 GB |
| Training (300K examples) | 36 hours | 22 GB |
| Evaluation | 30 min | 4 GB |

---

## 🔒 Security Considerations

1. **Data Privacy**: All data stored locally (git-ignored)
2. **API Keys**: Environment variables only
3. **Access Control**: Role-based access for deployment
4. **Audit Logging**: Comprehensive logging of all operations

---

**Status**: ✅ Production Ready  
**Next Steps**: See [QUICK_START.md](../QUICK_START.md) for usage

---

<div align="center">

# بليغ (Balygh) v3.0

**Architecture Overview**

[Quick Start](../QUICK_START.md) | [Implementation](implementation/) | [API](api/)

</div>
