# Arabic LLM Project Structure - Improved

## البنية المحسّنة لمشروع اللغة العربية

**Date**: March 25, 2026  
**Version**: 2.0.0  
**Status**: Production-Ready Structure  

---

## Issues with Current Structure

### ❌ Current Problems

1. **Root Directory Cluttered**: 11 Python/MD files in root
2. **No Package Structure**: `src/` not organized as proper Python package
3. **Mixed Concerns**: Core logic mixed with utilities
4. **No Tests**: Missing `tests/` directory
5. **No Examples**: Missing `examples/` directory
6. **Root Scripts**: `prepare.py`, `train.py`, `agent.py` should be in `scripts/`
7. **No CLI**: No command-line interface entry points
8. **No Version File**: Version not tracked properly
9. **Large Modules**: Some modules >900 lines (should be split)
10. **No Clear API**: No public API definition

---

## ✅ Improved Structure

### Proposed Directory Layout

```
arabic-llm/
│
├── 📁 arabic_llm/              # Main Python package (RENAME src/ → arabic_llm/)
│   ├── __init__.py             # Package init with version, public API
│   ├── version.py              # Version information
│   │
│   ├── 📁 core/                # Core business logic
│   │   ├── __init__.py
│   │   ├── schema.py           # Data models (moved from src/)
│   │   ├── schema_enhanced.py  # Enhanced schema (moved)
│   │   ├── templates.py        # Instruction templates (renamed)
│   │   ├── book_processor.py   # Book processing (moved)
│   │   └── dataset_generator.py # Dataset generation (moved)
│   │
│   ├── 📁 pipeline/            # Data processing pipelines
│   │   ├── __init__.py
│   │   ├── cleaning.py         # Data cleaning pipeline (renamed)
│   │   ├── segmentation.py     # Text segmentation
│   │   └── validation.py       # Quality validation
│   │
│   ├── 📁 integration/         # External integrations
│   │   ├── __init__.py
│   │   ├── system_books.py     # System book integration (renamed)
│   │   ├── databases.py        # Database connections
│   │   └── lucene.py           # Lucene index support
│   │
│   ├── 📁 models/              # Model training
│   │   ├── __init__.py
│   │   ├── qlora.py            # QLoRA training utilities
│   │   ├── quantization.py     # Quantization helpers
│   │   └── checkpoints.py      # Checkpoint management
│   │
│   ├── 📁 utils/               # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py          # Logging configuration
│   │   ├── io.py               # I/O utilities
│   │   ├── text.py             # Text processing utilities
│   │   └── arabic.py           # Arabic-specific utilities
│   │
│   └── 📁 agents/              # Autonomous agents
│       ├── __init__.py
│       ├── researcher.py       # Research agent (renamed from agent.py)
│       ├── proposals.py        # Experiment proposals
│       └── evaluator.py        # Experiment evaluator
│
├── 📁 scripts/                 # Command-line scripts
│   ├── __init__.py
│   ├── prepare.py              # Data preparation (moved from root)
│   ├── train.py                # Training script (moved)
│   ├── agent.py                # Autonomous agent (moved)
│   ├── process_books.py        # Process books (renamed 01_process_books.py)
│   ├── generate_dataset.py     # Generate dataset (renamed 02_generate_dataset.py)
│   ├── train_model.py          # Train model (renamed 03_train_model.py)
│   └── audit_datasets.py       # Dataset audit (moved)
│
├── 📁 tests/                   # Test suite (NEW)
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_schema.py          # Schema tests
│   ├── test_templates.py       # Template tests
│   ├── test_book_processor.py  # Book processor tests
│   ├── test_cleaning.py        # Cleaning pipeline tests
│   └── test_integration.py     # Integration tests
│
├── 📁 examples/                # Usage examples (NEW)
│   ├── __init__.py
│   ├── basic_usage.py          # Basic usage example
│   ├── custom_templates.py     # Custom template example
│   ├── fine_tuning.py          # Fine-tuning example
│   └── autonomous_research.py  # Autonomous research example
│
├── 📁 configs/                 # Configuration files
│   ├── training_config.yaml    # Training hyperparameters
│   ├── data_config.yaml        # Data configuration
│   ├── model_config.yaml       # Model selection (NEW)
│   └── agent_config.yaml       # Agent configuration (NEW)
│
├── 📁 docs/                    # Documentation
│   ├── README.md               # Documentation index
│   ├── architecture.md         # Architecture overview
│   ├── installation.md         # Installation guide
│   ├── quickstart.md           # Quick start guide
│   ├── api/                    # API documentation
│   │   ├── core.md
│   │   ├── pipeline.md
│   │   ├── models.md
│   │   └── agents.md
│   ├── guides/                 # User guides
│   │   ├── data_preparation.md
│   │   ├── fine_tuning.md
│   │   └── autonomous_research.md
│   └── reference/              # API reference
│       ├── schema.md
│       ├── templates.md
│       └── ...
│
├── 📁 notebooks/               # Jupyter notebooks
│   ├── exploration.ipynb       # Data exploration
│   ├── analysis.ipynb          # Dataset analysis
│   └── experiments/            # Experiment notebooks
│
├── 📁 data/                    # Data directories
│   ├── raw/                    # Raw extracted books
│   ├── processed/              # Processed data
│   ├── jsonl/                  # JSONL datasets
│   └── evaluation/             # Evaluation datasets
│
├── 📁 models/                  # Trained models (gitignored)
│   ├── checkpoints/            # Training checkpoints
│   └── final/                  # Final models
│
├── 📁 experiments/             # Experiment logs
│   ├── experiment_log.jsonl
│   └── best_loss.txt
│
├── 📄 Root Files (Clean)
│   ├── README.md               # Project overview
│   ├── QUICK_REFERENCE.md      # Quick reference
│   ├── pyproject.toml          # Project configuration
│   ├── requirements.txt        # Dependencies
│   ├── .gitignore              # Git ignore
│   └── .pre-commit-config.yaml # Pre-commit hooks (NEW)
│
└── 📄 Additional Files (NEW)
    ├── Makefile                # Make commands
    ├── CHANGELOG.md            # Changelog
    ├── LICENSE                 # License file
    └── setup.cfg               # Setup configuration
```

---

## Key Improvements

### 1. Proper Python Package

**Before**: `src/` directory with loose modules  
**After**: `arabic_llm/` package with subpackages

```python
# Before
from src.schema import TrainingExample

# After
from arabic_llm.core import TrainingExample
from arabic_llm.pipeline import DataCleaningPipeline
from arabic_llm.agents import ResearchAgent
```

### 2. Separation of Concerns

| Subpackage | Purpose | Modules |
|------------|---------|---------|
| `core/` | Business logic | schema, templates, processors |
| `pipeline/` | Data processing | cleaning, segmentation, validation |
| `integration/` | External systems | databases, system books, lucene |
| `models/` | ML training | qlora, quantization, checkpoints |
| `utils/` | Utilities | logging, io, text, arabic |
| `agents/` | Autonomous agents | researcher, proposals, evaluator |

### 3. Module Splitting

**Large Modules Split**:

```
data_cleaning_pipeline.py (910 lines)
  ↓
pipeline/
  ├── cleaning.py         # Main pipeline (400 lines)
  ├── segmentation.py     # Text segmentation (250 lines)
  └── validation.py       # Quality validation (260 lines)

instruction_templates.py (619 lines)
  ↓
core/templates.py
  ├── tutor_templates.py    # Tutor templates (200 lines)
  ├── proofreader_templates.py  # Proofreader templates (150 lines)
  ├── poet_templates.py     # Poet templates (150 lines)
  └── muhhaqiq_templates.py # Muhhaqiq templates (119 lines)
```

### 4. Test Suite

**New `tests/` directory**:

```python
# tests/test_schema.py
def test_training_example_validation():
    example = TrainingExample(...)
    errors = validate_example(example)
    assert len(errors) == 0

# tests/test_cleaning.py
def test_seven_stage_cleaning():
    cleaner = TextCleaner()
    cleaned = cleaner.clean("نص تجريبي")
    assert len(cleaned) > 0
```

### 5. Examples Directory

**New `examples/` directory**:

```python
# examples/basic_usage.py
from arabic_llm.core import DatasetGenerator
from arabic_llm.pipeline import DataCleaningPipeline

# Clean books
pipeline = DataCleaningPipeline("datasets/extracted_books")
cleaned = pipeline.run()

# Generate dataset
generator = DatasetGenerator(config)
dataset = generator.generate(cleaned)
```

### 6. CLI Entry Points

**Defined in `pyproject.toml`**:

```toml
[project.scripts]
arabic-llm-prepare = "arabic_llm.scripts.prepare:main"
arabic-llm-train = "arabic_llm.scripts.train:main"
arabic-llm-agent = "arabic_llm.scripts.agent:main"
arabic-llm-audit = "arabic_llm.scripts.audit_datasets:main"
arabic-llm-process = "arabic_llm.scripts.process_books:main"
arabic-llm-generate = "arabic_llm.scripts.generate_dataset:main"
```

### 7. Version Management

**New `arabic_llm/version.py`**:

```python
__version__ = "2.0.0"
__version_info__ = (2, 0, 0)
```

**Exposed in `arabic_llm/__init__.py`**:

```python
from .version import __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
    # Core
    "TrainingExample",
    "Role",
    "Skill",
    # Pipeline
    "DataCleaningPipeline",
    # Agents
    "ResearchAgent",
]
```

---

## Migration Plan

### Phase 1: Package Reorganization (Day 1)

1. ✅ Rename `src/` → `arabic_llm/`
2. ✅ Create subpackages: `core/`, `pipeline/`, `integration/`, `models/`, `utils/`, `agents/`
3. ✅ Move modules to appropriate subpackages
4. ✅ Update all imports
5. ✅ Create `arabic_llm/__init__.py` with public API

### Phase 2: Script Reorganization (Day 1)

1. ✅ Move `prepare.py`, `train.py`, `agent.py` to `scripts/`
2. ✅ Rename numbered scripts: `01_` → descriptive names
3. ✅ Update script imports
4. ✅ Add CLI entry points to `pyproject.toml`

### Phase 3: Module Splitting (Day 2)

1. ✅ Split `data_cleaning_pipeline.py` → `pipeline/` submodules
2. ✅ Split `instruction_templates.py` → `core/templates/` submodules
3. ✅ Update all references

### Phase 4: Testing Infrastructure (Day 3)

1. ✅ Create `tests/` directory
2. ✅ Add `pytest` configuration
3. ✅ Write tests for core modules
4. ✅ Add CI/CD configuration

### Phase 5: Examples & Documentation (Day 4)

1. ✅ Create `examples/` directory
2. ✅ Write usage examples
3. ✅ Update documentation with new structure
4. ✅ Add API documentation

---

## Benefits

### Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **Root Files** | 11 Python/MD files | 6 clean files |
| **Package Structure** | Flat `src/` | Organized subpackages |
| **Module Size** | Up to 910 lines | Max 400 lines |
| **Tests** | None | Comprehensive suite |
| **Examples** | None | 4+ examples |
| **CLI** | Manual execution | Entry points |
| **Version** | Hardcoded | Managed in `version.py` |
| **API** | Implicit | Explicit in `__init__.py` |
| **Documentation** | 9 files | Organized by type |

### Import Comparison

```python
# BEFORE - Confusing imports
from src.schema import TrainingExample
from src.data_cleaning_pipeline import DataCleaningPipeline
from ..agent import agent

# AFTER - Clear imports
from arabic_llm.core import TrainingExample
from arabic_llm.pipeline import DataCleaningPipeline
from arabic_llm.agents import ResearchAgent

# Or even simpler
import arabic_llm
example = arabic_llm.TrainingExample(...)
pipeline = arabic_llm.DataCleaningPipeline(...)
```

---

## Implementation Status

- [ ] Phase 1: Package Reorganization
- [ ] Phase 2: Script Reorganization
- [ ] Phase 3: Module Splitting
- [ ] Phase 4: Testing Infrastructure
- [ ] Phase 5: Examples & Documentation

**Target Completion**: 4 days  
**Risk Level**: Low (backward compatible during migration)  
**Breaking Changes**: Import paths (documented in migration guide)

---

**Version**: 2.0.0  
**Date**: March 25, 2026  
**Status**: Approved for Implementation
