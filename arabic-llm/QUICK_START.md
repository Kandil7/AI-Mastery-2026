# Balygh (بليغ) - Quick Start Guide

## دليل البدء السريع

**Last Updated**: March 27, 2026  
**Status**: ✅ Ready for Production

---

## 🚀 Option 1: One-Command Pipeline (Recommended)

```bash
cd K:\learning\technical\ai-ml\AI-Mastery-2026\arabic-llm

# Run complete pipeline (audit + process + merge)
python scripts/run_complete_pipeline.py --all

# Or run specific steps
python scripts/run_complete_pipeline.py --audit --process --merge
```

---

## 📋 Option 2: Step-by-Step

### Step 1: Audit Data Sources (5 minutes)

```bash
python scripts/complete_data_audit.py
```

**Output**: `data/complete_audit_report.json`

**Expected**:
```
✅ Arabic Web: found (Files: 1, Size: 10.0 GB)
✅ Extracted Books: found (Files: 8,425, Size: 16.4 GB)
✅ Metadata: found (Files: 6, Books: 8,424)
✅ Sanadset Hadith: found (Files: 1, Narrators: 368,000)
✅ System Books: found (Files: 5, Records: 100,000)

Overall Quality: 0.85
Readiness Score: 1.00
```

---

### Step 2: Process Each Source (30-60 minutes)

```bash
# Process Arabic Web Corpus (10 min)
python scripts/process_arabic_web.py

# Process Extracted Books (30 min)
python scripts/build_balygh_sft_dataset.py --target-examples 113000

# Process Sanadset Hadith (20 min)
python scripts/process_sanadset.py

# Process System Books (10 min)
python scripts/integrate_datasets.py
```

**Expected Output**:
```
✅ Arabic Web: 50,000 examples
✅ Extracted Books: 113,000 examples
✅ Sanadset: 130,000 examples
✅ System Books: 65,000 examples
Total: 358,000 raw examples
```

---

### Step 3: Refine with LLM (Optional, 1-2 hours)

```bash
# Set API key
$env:DEEPSEEK_API_KEY="sk-..."

# Refine outputs
python scripts/refine_balygh_sft_with_llm.py --max-examples 100000
```

**Note**: Skip if you don't have API key. Can use pre-refined data.

---

### Step 4: Merge & Deduplicate (10 minutes)

```bash
python scripts/merge_all_datasets.py
```

**Expected Output**:
```
Total Read: 358,000
Duplicates Removed: 50,000 (14%)
Quality Filtered: 8,000
Final Count: 300,000 examples
```

---

### Step 5: Train Model (36 hours)

```bash
python scripts/03_train_model.py \
  --config configs/training_config.yaml \
  --dataset data/jsonl/balygh_final_sft.jsonl \
  --output-dir models/balygh-complete-v1
```

**Training Time**:
- RTX 3090 (24GB): ~36 hours
- RTX 4090 (24GB): ~30 hours
- A100 (80GB): ~18 hours
- 8x A100: ~3 hours

---

### Step 6: Evaluate (30 minutes)

```bash
# Set model path
$env:BALYGH_MODEL_DIR="models/balygh-complete-v1"

# Run evaluation
python scripts/prepare.py
```

**Expected Output**:
```
balygh_score=0.78
  - fiqh_f1: 0.76
  - hadith_f1: 0.72
  - nahw_score: 0.82
  - balagha_score: 0.74
  - json_acc: 0.87
  - field_f1: 0.84
```

---

### Step 7: Deploy (10 minutes)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "YourUser/Balygh-7B-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "اشرح الفرق بين الفاعل ونائب الفاعل"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```

---

## 📊 Expected Timeline

| Step | Time | Output |
|------|------|--------|
| 1. Audit | 5 min | Audit report |
| 2. Process | 60 min | 358K examples |
| 3. Refine | 120 min | Refined examples |
| 4. Merge | 10 min | 300K unique |
| 5. Train | 36 hours | Model checkpoint |
| 6. Evaluate | 30 min | balygh_score |
| 7. Deploy | 10 min | HF model + demo |
| **TOTAL** | **~38 hours** | **Production model** |

---

## 🎯 Quick Commands Reference

```bash
# Audit
python scripts/complete_data_audit.py

# Process individual sources
python scripts/process_arabic_web.py
python scripts/build_balygh_sft_dataset.py
python scripts/process_sanadset.py
python scripts/integrate_datasets.py

# Merge
python scripts/merge_all_datasets.py

# Train
python scripts/03_train_model.py --config configs/training_config.yaml

# Evaluate
python scripts/prepare.py

# Full pipeline
python scripts/run_complete_pipeline.py --all
```

---

## 📁 File Locations

```
K:\learning\technical\ai-ml\AI-Mastery-2026\arabic-llm\
├── scripts/
│   ├── complete_data_audit.py          # Audit all 5 sources
│   ├── process_arabic_web.py           # Process web corpus
│   ├── build_balygh_sft_dataset.py     # Process books
│   ├── process_sanadset.py             # Process hadith
│   ├── integrate_datasets.py           # Process system DBs
│   ├── merge_all_datasets.py           # Merge all
│   ├── refine_balygh_sft_with_llm.py   # LLM refinement
│   ├── prepare.py                      # Evaluation
│   ├── 03_train_model.py               # Training
│   └── run_complete_pipeline.py        # Master pipeline
│
├── data/
│   ├── jsonl/
│   │   ├── arabic_web_*.jsonl          # Web examples
│   │   ├── balygh_sft_*.jsonl          # Book examples
│   │   ├── sanadset_*.jsonl            # Hadith examples
│   │   ├── balygh_final_sft.jsonl      # Merged dataset
│   │   └── ...
│   └── complete_audit_report.json      # Audit report
│
├── models/
│   └── balygh-complete-v1/             # Trained model
│
└── configs/
    └── training_config.yaml            # Training config
```

---

## ✅ Pre-Flight Checklist

Before running the pipeline:

- [ ] Python 3.10+ installed
- [ ] Dependencies installed: `pip install -e .`
- [ ] GPU with 24GB+ VRAM available (for training)
- [ ] Disk space: 100GB+ free
- [ ] API key set (optional, for refinement): `$env:DEEPSEEK_API_KEY="sk-..."`

---

## 🆘 Troubleshooting

### "Script not found"
```bash
# Make sure you're in the right directory
cd K:\learning\technical\ai-ml\AI-Mastery-2026\arabic-llm
```

### "Out of memory"
```bash
# Reduce batch size in training_config.yaml
# Change: per_device_train_batch_size: 4 → 2
```

### "Dataset not found"
```bash
# Run processing steps first
python scripts/run_complete_pipeline.py --process --merge
```

### "API key required"
```bash
# Skip refinement step or set key
$env:DEEPSEEK_API_KEY="sk-..."
```

---

## 📞 Next Steps After Training

1. **Push to Hugging Face**
   ```bash
   huggingface-cli login
   python -c "from transformers import *; AutoModelForCausalLM.from_pretrained('models/balygh-complete-v1').push_to_hub('YourUser/Balygh-7B-v1')"
   ```

2. **Create Gradio Demo**
   - See `COMPLETE_DATA_UTILIZATION_PLAN.md` for demo code

3. **Write Documentation**
   - Model card, usage examples, limitations

4. **Public Release**
   - Blog post, social media, community outreach

---

**Ready to start? Run this command:**

```bash
python scripts/run_complete_pipeline.py --all
```

---

<div align="center">

# بليغ (Balygh) - Quick Start

**5 مصادر • 300,000 مثال • 29 دور • 76 مهارة**

[Run Pipeline](scripts/run_complete_pipeline.py) | [Documentation](../docs/) | [Models](models/)

</div>
