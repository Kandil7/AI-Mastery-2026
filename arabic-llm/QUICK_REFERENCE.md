# Arabic LLM Quick Reference Guide

## Quick Start Commands

### 1. Run Audit
```bash
python arabic-llm/scripts/audit_datasets.py
```

### 2. Clean Books
```bash
python arabic-llm/src/data_cleaning_pipeline.py \
    --books-dir datasets/extracted_books \
    --metadata-dir datasets/metadata \
    --output-dir data/processed \
    --workers 8
```

### 3. Generate System Examples
```bash
python arabic-llm/src/system_book_integration.py \
    --base-dir datasets/system_book_datasets \
    --output-dir data/system_examples \
    --limit 11500
```

### 4. Generate Dataset
```bash
python arabic-llm/scripts/02_generate_dataset.py \
    --books-dir datasets/extracted_books \
    --metadata-dir datasets/metadata \
    --input-dir data/processed \
    --output-dir data/jsonl \
    --target-examples 61500
```

### 5. Train Model
```bash
python arabic-llm/scripts/03_train_model.py \
    --dataset data/jsonl/train.jsonl \
    --output-dir models/arabic-linguist-v1 \
    --config configs/training_config.yaml
```

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Books | 8,424 |
| Total Size | 16.4 GB |
| Categories | 40 |
| Authors | 3,146 |
| Training Examples | 61,500+ |
| Roles | 19 |
| Skills | 48+ |

---

## Role Distribution

| Role | % | Examples |
|------|---|----------|
| tutor | 35% | 21,525 |
| proofreader | 25% | 15,375 |
| poet | 20% | 12,300 |
| muhhaqiq | 15% | 9,225 |
| assistant_general | 5% | 3,075 |

---

## File Structure

```
arabic-llm/
├── src/
│   ├── schema.py
│   ├── instruction_templates.py
│   ├── book_processor.py
│   ├── dataset_generator.py
│   ├── data_cleaning_pipeline.py
│   ├── system_book_integration.py
│   └── schema_enhanced.py
├── scripts/
│   ├── 01_process_books.py
│   ├── 02_generate_dataset.py
│   ├── 03_train_model.py
│   └── audit_datasets.py
├── configs/
│   ├── training_config.yaml
│   └── data_config.yaml
├── docs/
│   ├── COMPLETE_DOCUMENTATION.md
│   ├── complete_data_preparation.md
│   ├── data_cleaning_pipeline.md
│   ├── enhanced_roles_skills.md
│   └── system_book_integration.md
└── README.md
```

---

## Common Issues

### Out of Memory
```bash
# Reduce batch size
--batch-size 2

# Reduce workers
--workers 2
```

### Slow Processing
```bash
# Increase workers
--workers 16

# Process in batches
--max-books 500
```

### Validation Errors
```bash
# Check audit report
cat datasets/audit_report.json

# Review quality report
cat data/processed/quality_report.json
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/schema.py` | Data models |
| `src/instruction_templates.py` | 50+ templates |
| `src/data_cleaning_pipeline.py` | 7-stage cleaning |
| `src/system_book_integration.py` | DB integration |
| `scripts/audit_datasets.py` | Dataset audit |
| `configs/training_config.yaml` | QLoRA config |

---

## Performance

| Operation | Speed |
|-----------|-------|
| Cleaning (8 workers) | 3,500 books/hour |
| Full pipeline | 2.4 hours (8,424 books) |
| Training (7B) | ~12 hours |
| Memory (8 workers) | 1.3 GB |

---

## Quality Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Arabic ratio | >70% | 89% ✅ |
| Diacritics | 5-20% | 12% ✅ |
| Empty files | 0 | 0 ✅ |
| Encoding errors | 0 | 0 ✅ |

---

**Version**: 1.0.0  
**Status**: Production Ready
