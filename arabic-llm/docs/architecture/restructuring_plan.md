# Balygh (بليغ) - Architecture Restructuring Plan

## خطة إعادة هيكلة البنية المعمارية

**Date**: March 27, 2026  
**Version**: 3.0.0  
**Status**: 🟡 Planning Phase

---

## 📊 Current Architecture Issues

### 1. **Inconsistent Directory Structure**

**Current Issues**:
- ❌ Root directory has 28+ files (too many)
- ❌ Documentation files scattered in root
- ❌ Scripts mixed between `scripts/` and root
- ❌ Multiple overlapping documentation files
- ❌ No clear separation between source code and scripts

**Files in Root (28 items)**:
```
arabic-llm/
├── *.md (11 documentation files - should be in docs/)
├── *.py (3 scripts - should be in scripts/)
├── *.yaml (0 - good)
├── *.toml (2 - good)
├── *.txt (1 - good)
├── Makefile (1 - good)
└── Directories (8 - needs reorganization)
```

### 2. **Module Organization Issues**

**arabic_llm/core/**:
- ✅ `schema.py` - Good
- ✅ `templates.py` - Good
- ⚠️ `schema_enhanced.py` - Redundant (merge with schema.py)
- ⚠️ `book_processor.py` - Should be in `processing/`
- ⚠️ `dataset_generator.py` - Should be in `generation/`

**arabic_llm/agents/**:
- ✅ `data_collector.py` - Good
- ✅ `evaluator.py` - Good
- ❌ `proposals.py` - Unclear purpose
- ❌ `researcher.py` - Unclear purpose
- ❌ `tracker.py` - Unclear purpose

**arabic_llm/pipeline/**:
- ✅ `cleaning.py` - Good
- ✅ `deduplication.py` - Good

**arabic_llm/models/**:
- ⚠️ `checkpoints.py` - Should be in `training/`
- ⚠️ `qlora.py` - Should be in `training/`
- ⚠️ `quantization.py` - Should be in `training/`

**arabic_llm/integration/**:
- ✅ `databases.py` - Good
- ✅ `system_books.py` - Good

**arabic_llm/utils/**:
- ✅ `arabic.py` - Good
- ✅ `io.py` - Good
- ✅ `logging.py` - Good
- ✅ `text.py` - Good

### 3. **Scripts Organization**

**Current (17 scripts)**:
```
scripts/
├── 01_process_books.py          # Keep
├── 02_generate_dataset.py       # Keep
├── 03_train_model.py            # Keep
├── agent.py                     # Rename?
├── analysis.py                  # Too generic
├── audit_datasets.py            # Keep
├── build_balygh_sft_dataset.py  # Keep
├── complete_data_audit.py       # Keep
├── complete_pipeline.py         # Duplicate of run_complete_pipeline.py
├── integrate_datasets.py        # Keep
├── merge_all_datasets.py        # Keep
├── prepare.py                   # Keep
├── process_arabic_web.py        # Keep
├── process_sanadset.py          # Keep
├── refine_balygh_sft_with_llm.py # Keep
├── run_complete_pipeline.py     # Keep
└── train.py                     # Duplicate of 03_train_model.py
```

**Issues**:
- ❌ `complete_pipeline.py` and `run_complete_pipeline.py` - duplicates
- ❌ `train.py` and `03_train_model.py` - duplicates
- ❌ `analysis.py` - too generic
- ❌ `agent.py` - unclear purpose

### 4. **Documentation Overload**

**Current (11 MD files in root)**:
```
├── AUTORESEARCH_README.md           # Move to docs/
├── CLEANUP_PLAN.md                  # Delete or move to docs/archive/
├── COMPLETE_DATA_UTILIZATION_PLAN.md # Move to docs/
├── DATA_UPDATES_IMPROVEMENTS.md     # Move to docs/
├── FINAL_ARCHITECTURE_STATUS.md     # Delete (outdated)
├── FINAL_IMPLEMENTATION_SUMMARY.md  # Move to docs/
├── IMPLEMENTATION_COMPLETE.md       # Move to docs/
├── IMPLEMENTATION_LINES_8000_9866.md # Move to docs/archive/
├── IMPLEMENTATION_LINES_9800_11993.md # Move to docs/archive/
├── QUICK_REFERENCE.md               # Keep in root (OK)
├── QUICK_START.md                   # Keep in root (OK)
└── README.md                        # Keep in root (OK)
```

---

## 🎯 Target Architecture (Version 3.0)

### Proposed Directory Structure

```
arabic-llm/
├── 📁 arabic_llm/                    # Main package
│   ├── __init__.py
│   ├── version.py
│   │
│   ├── 📁 core/                      # Core schemas & templates
│   │   ├── __init__.py
│   │   ├── schema.py                 # 29 roles, 76 skills
│   │   └── templates.py              # Instruction templates
│   │
│   ├── 📁 processing/                # Data processing
│   │   ├── __init__.py
│   │   ├── cleaning.py               # 7-stage cleaning
│   │   ├── deduplication.py          # MinHash LSH
│   │   └── book_processor.py         # Book extraction
│   │
│   ├── 📁 generation/                # Dataset generation
│   │   ├── __init__.py
│   │   └── dataset_generator.py      # SFT example generation
│   │
│   ├── 📁 training/                  # Training utilities
│   │   ├── __init__.py
│   │   ├── qlora.py                  # QLoRA utilities
│   │   ├── quantization.py           # Quantization config
│   │   └── checkpoints.py            # Checkpoint management
│   │
│   ├── 📁 agents/                    # AI agents
│   │   ├── __init__.py
│   │   ├── data_collector.py         # Web scraping
│   │   └── evaluator.py              # Evaluation
│   │
│   ├── 📁 integration/               # Database integration
│   │   ├── __init__.py
│   │   ├── databases.py              # DB connections
│   │   └── system_books.py           # Book system
│   │
│   └── 📁 utils/                     # Utilities
│       ├── __init__.py
│       ├── arabic.py                 # Arabic utilities
│       ├── io.py                     # I/O utilities
│       ├── logging.py                # Logging setup
│       └── text.py                   # Text utilities
│
├── 📁 scripts/                       # All executable scripts
│   ├── 📁 processing/                # Data processing scripts
│   │   ├── complete_data_audit.py
│   │   ├── process_arabic_web.py
│   │   ├── process_books.py
│   │   ├── process_sanadset.py
│   │   └── integrate_datasets.py
│   │
│   ├── 📁 generation/                # Dataset generation
│   │   ├── build_balygh_sft.py
│   │   └── refine_with_llm.py
│   │
│   ├── 📁 training/                  # Training scripts
│   │   ├── train.py
│   │   └── prepare_eval.py
│   │
│   ├── 📁 utilities/                 # Utility scripts
│   │   ├── merge_datasets.py
│   │   └── audit_datasets.py
│   │
│   └── run_pipeline.py               # Master pipeline
│
├── 📁 configs/                       # Configuration files
│   ├── training.yaml
│   ├── data.yaml
│   ├── model.yaml
│   └── evaluation.yaml
│
├── 📁 data/                          # Data directories
│   ├── raw/                          # Raw data (git-ignored)
│   ├── processed/                    # Processed data (git-ignored)
│   ├── jsonl/                        # JSONL datasets (git-ignored)
│   ├── evaluation/                   # Evaluation sets
│   └── README.md                     # Data documentation
│
├── 📁 models/                        # Model outputs (git-ignored)
│   ├── balygh-v1/
│   ├── balygh-v2/
│   └── README.md
│
├── 📁 docs/                          # All documentation
│   ├── 📁 guides/                    # User guides
│   │   ├── quick_start.md
│   │   ├── installation.md
│   │   └── tutorial.md
│   │
│   ├── 📁 architecture/              # Architecture docs
│   │   ├── overview.md
│   │   ├── data_pipeline.md
│   │   └── training_pipeline.md
│   │
│   ├── 📁 api/                       # API documentation
│   │   ├── core.md
│   │   ├── processing.md
│   │   └── agents.md
│   │
│   ├── 📁 implementation/            # Implementation docs
│   │   ├── complete.md
│   │   ├── data_utilization.md
│   │   └── improvements.md
│   │
│   └── 📁 archive/                   # Archived docs
│       ├── implementation_lines_*.md
│       └── old_plans.md
│
├── 📁 tests/                         # Test suite
│   ├── __init__.py
│   ├── test_schema.py
│   ├── test_cleaning.py
│   ├── test_deduplication.py
│   └── test_generation.py
│
├── 📁 examples/                      # Example notebooks & scripts
│   ├── basic_usage.ipynb
│   ├── advanced_rag.ipynb
│   └── fine_tuning_example.py
│
├── 📁 deployment/                    # Deployment configs
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   └── deployment.yaml
│   └── api/
│       └── fastapi_app.py
│
├── README.md                         # Main readme (updated)
├── QUICK_START.md                    # Quick start guide
├── pyproject.toml                    # Python project config
├── requirements.txt                  # Dependencies
├── Makefile                          # Make commands
└── .gitignore                        # Git ignore rules
```

---

## 🔄 Migration Plan

### Phase 1: Clean Root Directory (30 minutes)

**Actions**:
1. Move all `.md` files (except README, QUICK_START) to `docs/`
2. Delete outdated files (CLEANUP_PLAN.md, FINAL_ARCHITECTURE_STATUS.md)
3. Merge duplicate implementation docs
4. Move duplicate scripts to archive

**Files to Move**:
```bash
# Move to docs/implementation/
mv IMPLEMENTATION_COMPLETE.md docs/implementation/
mv COMPLETE_DATA_UTILIZATION_PLAN.md docs/implementation/
mv DATA_UPDATES_IMPROVEMENTS.md docs/implementation/
mv FINAL_IMPLEMENTATION_SUMMARY.md docs/implementation/

# Move to docs/archive/
mv IMPLEMENTATION_LINES_8000_9866.md docs/archive/
mv IMPLEMENTATION_LINES_9800_11993.md docs/archive/
mv FINAL_ARCHITECTURE_STATUS.md docs/archive/
mv CLEANUP_PLAN.md docs/archive/

# Move to docs/guides/
mv QUICK_START.md docs/guides/quick_start.md
mv QUICK_REFERENCE.md docs/guides/quick_reference.md

# Delete duplicates
rm complete_pipeline.py  # Duplicate of run_complete_pipeline.py
rm train.py  # Duplicate of 03_train_model.py
```

### Phase 2: Reorganize arabic_llm Package (1 hour)

**Actions**:
1. Create new subdirectories: `processing/`, `generation/`, `training/`
2. Move files to appropriate directories
3. Update imports
4. Merge redundant files

**File Movements**:
```bash
# Create new directories
mkdir -p arabic_llm/processing
mkdir -p arabic_llm/generation
mkdir -p arabic_llm/training

# Move processing files
mv arabic_llm/pipeline/cleaning.py arabic_llm/processing/
mv arabic_llm/pipeline/deduplication.py arabic_llm/processing/
mv arabic_llm/core/book_processor.py arabic_llm/processing/

# Move generation files
mv arabic_llm/core/dataset_generator.py arabic_llm/generation/

# Move training files
mv arabic_llm/models/qlora.py arabic_llm/training/
mv arabic_llm/models/quantization.py arabic_llm/training/
mv arabic_llm/models/checkpoints.py arabic_llm/training/

# Remove empty directories
rmdir arabic_llm/pipeline
rmdir arabic_llm/models
```

**Merge Redundant Files**:
```bash
# Merge schema files
cat arabic_llm/core/schema_enhanced.py >> arabic_llm/core/schema.py
rm arabic_llm/core/schema_enhanced.py

# Merge template files
cat arabic_llm/core/templates_extended.py >> arabic_llm/core/templates.py
rm arabic_llm/core/templates_extended.py
```

### Phase 3: Reorganize Scripts (1 hour)

**Actions**:
1. Create subdirectories in `scripts/`
2. Group scripts by function
3. Remove duplicates
4. Rename for clarity

**File Movements**:
```bash
# Create subdirectories
mkdir -p scripts/processing
mkdir -p scripts/generation
mkdir -p scripts/training
mkdir -p scripts/utilities

# Move processing scripts
mv scripts/complete_data_audit.py scripts/processing/
mv scripts/process_arabic_web.py scripts/processing/
mv scripts/01_process_books.py scripts/processing/process_books.py
mv scripts/process_sanadset.py scripts/processing/
mv scripts/integrate_datasets.py scripts/processing/

# Move generation scripts
mv scripts/build_balygh_sft_dataset.py scripts/generation/build_balygh_sft.py
mv scripts/02_generate_dataset.py scripts/generation/generate_dataset.py
mv scripts/refine_balygh_sft_with_llm.py scripts/generation/refine_with_llm.py

# Move training scripts
mv scripts/03_train_model.py scripts/training/train.py
mv scripts/prepare.py scripts/training/prepare_eval.py
mv scripts/train.py scripts/training/  # If not already deleted

# Move utility scripts
mv scripts/audit_datasets.py scripts/utilities/
mv scripts/merge_all_datasets.py scripts/utilities/

# Keep master pipeline in root
mv scripts/run_complete_pipeline.py scripts/run_pipeline.py

# Remove duplicates
rm scripts/complete_pipeline.py
rm scripts/train.py  # If duplicate
```

### Phase 4: Update Configuration (30 minutes)

**Actions**:
1. Rename config files for clarity
2. Create evaluation config
3. Update paths in configs

**File Changes**:
```bash
# Rename for clarity
mv configs/training_config.yaml configs/training.yaml
mv configs/data_config.yaml configs/data.yaml

# Create new configs
touch configs/model.yaml
touch configs/evaluation.yaml
```

### Phase 5: Update Documentation (1 hour)

**Actions**:
1. Create consolidated README.md
2. Update QUICK_START.md with new paths
3. Create architecture documentation
4. Update all internal references

**New Files**:
```bash
# Create docs/architecture/overview.md
# Create docs/guides/installation.md
# Create docs/guides/tutorial.md
# Update README.md with new structure
```

### Phase 6: Update Imports & Tests (2 hours)

**Actions**:
1. Update all imports in codebase
2. Update test imports
3. Run tests to verify
4. Update pyproject.toml scripts

**Import Updates**:
```python
# Old
from arabic_llm.pipeline.cleaning import ArabicTextCleaner
from arabic_llm.core.schema import TrainingExample
from arabic_llm.models.qlora import QLoRAConfig

# New
from arabic_llm.processing.cleaning import ArabicTextCleaner
from arabic_llm.core.schema import TrainingExample
from arabic_llm.training.qlora import QLoRAConfig
```

---

## ✅ Post-Migration Checklist

### Structure Verification
- [ ] Root directory has < 15 items
- [ ] All `.md` files in `docs/`
- [ ] All scripts in `scripts/` subdirectories
- [ ] No duplicate files
- [ ] No empty directories

### Code Verification
- [ ] All imports updated
- [ ] All tests pass
- [ ] No broken imports
- [ ] CLI commands work

### Documentation Verification
- [ ] README.md updated with new structure
- [ ] QUICK_START.md paths updated
- [ ] Architecture docs created
- [ ] API docs created

### Testing Verification
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] CLI commands work
- [ ] Pipeline runs successfully

---

## 📊 Expected Benefits

### Before (Current)
- ❌ 28 items in root directory
- ❌ 11 documentation files scattered
- ❌ 17 unorganized scripts
- ❌ Unclear module boundaries
- ❌ Duplicate files
- ⚠️ Hard to navigate

### After (Target)
- ✅ < 15 items in root directory
- ✅ All docs organized in `docs/`
- ✅ Scripts organized by function
- ✅ Clear module boundaries
- ✅ No duplicates
- ✅ Easy to navigate

---

## 🚀 Implementation Timeline

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| 1 | Clean root directory | 30 min | 🔴 High |
| 2 | Reorganize package | 1 hour | 🔴 High |
| 3 | Reorganize scripts | 1 hour | 🔴 High |
| 4 | Update configs | 30 min | 🟡 Medium |
| 5 | Update documentation | 1 hour | 🟡 Medium |
| 6 | Update imports & tests | 2 hours | 🔴 High |
| **TOTAL** | | **6 hours** | |

---

## 📝 Notes

1. **Backwards Compatibility**: Create alias modules for old imports during transition period
2. **Git History**: This is a breaking change - consider creating a new branch
3. **Version Bump**: Bump to v3.0.0 due to breaking changes
4. **Communication**: Notify all users of the restructuring

---

**Status**: 🟡 Ready for Implementation  
**Next Step**: Get approval and schedule migration window

---

<div align="center">

# بليغ (Balygh) v3.0

**Restructuring Plan**

[Approve](#) | [Implement](#) | [Review](#)

</div>
