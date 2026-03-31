# Balygh (بليغ) - Quick Reference Card

## بطاقة المرجع السريع

**Version**: 3.0.0  
**Last Updated**: March 27, 2026

---

## 🚀 Quick Commands

### Installation
```bash
git clone https://github.com/youruser/arabic-llm.git
cd arabic-llm
pip install -e .
python test_all_modules.py  # Verify
```

### Data Processing
```bash
python scripts/complete_data_audit.py           # Audit data
python scripts/processing/process_books.py      # Process books
python scripts/generation/build_balygh_sft.py   # Generate dataset
```

### Training
```bash
python scripts/training/train.py                # Train model
python scripts/training/prepare_eval.py         # Evaluate
```

### Testing
```bash
python test_all_modules.py                      # Test all modules
python -m compileall arabic_llm/ scripts/       # Compile all
```

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Roles** | 29 |
| **Skills** | 76 |
| **Data Sources** | 5 |
| **Total Data Size** | 29.4 GB |
| **Training Examples** | 300,000 |
| **Python Files** | 40+ |
| **Documentation** | 30+ files |

---

## 🏗️ Architecture

```
arabic-llm/
├── arabic_llm/          # Source code
│   ├── core/           # Schema, templates
│   ├── processing/     # Cleaning, dedup
│   ├── generation/     # Dataset gen
│   ├── training/       # QLoRA
│   ├── agents/         # AI agents
│   ├── integration/    # Databases
│   └── utils/          # Utilities
│
├── scripts/            # Scripts
├── configs/            # Configs
├── docs/               # Docs
├── data/               # Data
└── models/             # Models
```

---

## 🎯 29 Roles (5 Categories)

### Core Linguistic (5)
`tutor`, `proofreader`, `poet`, `muhhaqiq`, `assistant_general`

### Islamic Sciences (10)
`faqih`, `muhaddith`, `mufassir`, `aqeedah_specialist`, `sufi`, `historian`, `genealogist`, `geographer`, `physician`, `logician`

### Modern/Tech (5)
`rag_assistant`, `edtech_tutor`, `dataengineer_ar`, `fatwa_assistant_safe`, `tool_caller_ar`

### Literature (3)
`adab_specialist`, `quran_reciter`, `legal_arabic_drafting`

### Dialect (3)
`dialect_handling_egy`, `summarizer_ar`, `translator_ar`

---

## 🎓 76 Skills (8 Categories)

### Core Linguistic (8)
`nahw`, `sarf`, `balagha`, `orthography`, `phonology`, `semantics`, `lexicography`, `qiraat`

### Islamic Sciences (15)
`fiqh`, `usul_fiqh`, `hadith`, `hadith_mustalah`, `tafsir`, `aqeedah`, `sects`, `tasawwuf`, `zakat`, `inheritance`, `fatwa`, `judicial`, `seerah`, `quran_sciences`, `comparative_fiqh`

### Literature (5)
`poetry`, `heritage`, `adab`, `manuscripts`, `literary_criticism`

### NLP/Tech (12)
`rag_retrieval`, `rag_grounded_answering`, `function_calling_ar`, `summarization`, `text_classification`, `named_entity_ar`, `sentiment_ar`, `translation_ar_en`, `assessment_design`, `curriculum_aligned_ar`, `structured_output_ar`, `data_structuring`

### Dialects (5)
`dialect_egy`, `dialect_glf`, `dialect_lev`, `dialect_msa`, `transliteration`

### Extended Islamic (8)
`maqasid_shariah`, `comparative_religions`, `islamic_history`, `islamic_civilization`, `arabic_geography`, `islamic_medicine`, `islamic_philosophy`, `islamic_economics`

### Utility (10)
`qa`, `style_editing`, `error_analysis_ar`, `citation_extraction`, `document_parsing`, `qa_generation`, `consistency_check`, `simplification_ar`, `explanation`, `analysis`

### Specialized (5)
`medical_arabic`, `legal_arabic`, `business_arabic`, `technical_arabic`, `educational_arabic`

---

## 🔧 Configuration

### Training Config (training.yaml)
```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  
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

### Data Distribution
- Fiqh: 30%
- Language: 35%
- Hadith: 20%
- Other: 15%

---

## 📝 Common Tasks

### Generate Small Dataset (1K examples)
```bash
python scripts/generation/build_balygh_sft.py --target-examples 1000
```

### Generate Full Dataset (100K examples)
```bash
python scripts/generation/build_balygh_sft.py --target-examples 100000
```

### Process Specific Books
```bash
python scripts/processing/process_books.py --max-books 500
```

### Run Data Audit
```bash
python scripts/complete_data_audit.py
```

### Test Specific Module
```bash
python -c "from arabic_llm.core.schema import Role; print(len(Role))"
```

---

## 🐛 Troubleshooting

### Import Errors
```bash
pip install -e .
python -m pip install --upgrade pip
```

### Module Not Found
```bash
# Check if module exists
ls arabic_llm/*/
python -c "import arabic_llm; print(arabic_llm.__file__)"
```

### Compilation Errors
```bash
python -m compileall arabic_llm/ scripts/
```

### Data Not Found
```bash
python scripts/complete_data_audit.py
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](../README.md) | Project overview |
| [QUICK_START.md](../QUICK_START.md) | Quick start |
| [COMPLETE_DOCUMENTATION_INDEX.md](../COMPLETE_DOCUMENTATION_INDEX.md) | All docs |
| [LEARNING_PATH.md](education/LEARNING_PATH.md) | Learning guide |
| [BEGINNER_GUIDE.md](education/BEGINNER_GUIDE.md) | Beginner guide |

---

## 🔗 External Resources

- **Hugging Face**: https://huggingface.co
- **OALL Leaderboard**: https://huggingface.co/OALL
- **Qwen2.5**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **Unsloth**: https://github.com/unslothai/unsloth

---

## 📞 Support

- **GitHub Issues**: For bugs and feature requests
- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` and `scripts/` directories

---

**Status**: ✅ **PRODUCTION READY**  
**Version**: 3.0.0  
**Date**: March 27, 2026

---

<div align="center">

# بليغ (Balygh) v3.0

**بطاقة المرجع السريع**

**Quick Reference Card**

[Full Documentation](../COMPLETE_DOCUMENTATION_INDEX.md) | [Learning Path](education/LEARNING_PATH.md)

**29 دور • 76 مهارة • 300,000 مثال**

**29 Roles • 76 Skills • 300K Examples**

</div>
