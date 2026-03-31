# Arabic LLM Research Program v2.1

## برنامج بحث اللغة العربية للذكاء الاصطناعي

**Version**: 2.1.0  
**Date**: March 26, 2026  
**Based on**: Karpathy's autoresearch pattern  

---

## 🎯 Goal

Minimize **val_bpb** (validation bits per byte) on Arabic text while maintaining:

- ✅ Arabic ratio > 70%
- ✅ Training stability (no divergence)
- ✅ Time budget: 5 minutes per experiment
- ✅ Zero data loss guarantee

**Primary Metric**: val_bpb (lower is better, vocab-size-independent)  
**Secondary Metric**: val_loss (for reference)

---

## 📊 Current Best

| Metric | Value | Configuration |
|--------|-------|---------------|
| **val_bpb** | 1.234 | LoRA r=128, lr=2e-4, depth=8 |
| **val_loss** | 1.156 | Same as above |
| **Time** | 5m 12s | Within budget |

**Last Updated**: Experiment #42

---

## 📈 Recent Learnings (Last 20 Experiments)

### ✅ What's Working

1. **LoRA rank 128 > 64**
   - Consistent improvement in val_bpb
   - Better capacity for Arabic patterns
   - Next: Try 144, 160

2. **Learning rate 2e-4 > 1e-4**
   - Faster convergence
   - No instability observed
   - Next: Try 3e-4, 5e-4

3. **Warmup ratio 0.05 > 0.03**
   - Smoother training
   - Better final val_bpb
   - Next: Try 0.07, 0.10

### ⚠️ What's Not Working

1. **Depth > 12 causes overfitting**
   - val_loss decreases but val_bpb increases
   - Gap between train/val grows
   - Avoid: depth > 12 for now

2. **Batch size > 32 no improvement**
   - Diminishing returns
   - Slower training
   - Stick with: batch 8-32

3. **Dropout > 0.1 hurts performance**
   - Underfitting observed
   - Arabic data needs capacity
   - Keep dropout ≤ 0.1

---

## 🎯 Next Directions to Explore

### Priority 1: Fine-tune LoRA rank (High confidence)
```python
# Try values around 128
LORA_R = 144  # or 160, 192
LORA_ALPHA = LORA_R * 2  # Keep 2x ratio
```

### Priority 2: AdamW betas (Medium confidence)
```python
# Try different betas
BETAS = (0.9, 0.95)  # vs (0.9, 0.999)
# or
BETAS = (0.9, 0.99)  # middle ground
```

### Priority 3: Warmup ratio (Medium confidence)
```python
# Increase warmup
WARMUP_RATIO = 0.05  # or 0.07, 0.10
```

### Priority 4: Architecture exploration (Low confidence)
```python
# Try different depth/width tradeoffs
DEPTH = 10  # vs 8
HIDDEN_SIZE = 640  # vs 512
```

---

## ⚙️ Constraints

### What You Can Modify
- ✅ `train_model.py` ONLY

### What You Can Change in train_model.py
- ✅ Model architecture: `DEPTH`, `HIDDEN_SIZE`, `NUM_HEADS`, `INTERMEDIATE_SIZE`
- ✅ LoRA config: `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `TARGET_MODULES`
- ✅ Optimizer: `LEARNING_RATE`, `WEIGHT_DECAY`, `BETAS`, optimizer type
- ✅ Schedule: `WARMUP_RATIO`, `MAX_STEPS`, scheduler type
- ✅ Batch size: `DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`
- ✅ Regularization: `DROPOUT`, `ATTENTION_DROPOUT`, `GRAD_CLIP`

### What You CANNOT Modify
- ❌ `prepare_data.py` (fixed utilities)
- ❌ Dataset files (`data/jsonl/*.jsonl`)
- ❌ Metadata files
- ❌ This file (`program.md`) - human only

### Hard Constraints
- ⛔ Keep `TIME_BUDGET_SECONDS = 300` (5 minutes)
- ⛔ Monitor for divergence (val_bpb > 2.0)
- ⛔ Log all experiments to `experiments/experiment_log.jsonl`
- ⛔ Update `EXPERIMENT_ID` and `EXPERIMENT_CHANGE` for each run

---

## 📋 Experiment Format

### Before Running
```python
# Update in train_model.py
EXPERIMENT_ID = 43  # Increment
EXPERIMENT_CHANGE = "Increased LoRA rank from 128 to 144"  # Describe change

# Apply modifications
LORA_R = 144  # Changed from 128
LORA_ALPHA = 288  # Changed from 256 (keep 2x ratio)
```

### After Running
```
Experiment #43
Change: Increased LoRA rank from 128 to 144
Result: val_bpb 1.189 (val_loss: 1.123) [✓ IMPROVED]
Time: 5m 8s
```

### Decision
- ✅ **IMPROVED**: Keep changes, update "Current Best", continue in this direction
- ❌ **NOT IMPROVED**: Revert changes, try different direction

---

## 🔬 Experiment Categories

### Category 1: LoRA Configuration
```python
# Parameters to tune
LORA_R = 64, 128, 144, 160, 192, 256
LORA_ALPHA = LORA_R * 2  # Keep 2x ratio
LORA_DROPOUT = 0.0, 0.05, 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # or all 7
```

### Category 2: Optimizer
```python
# Parameters to tune
LEARNING_RATE = 1e-4, 2e-4, 3e-4, 5e-4, 1e-3
WEIGHT_DECAY = 0.0, 0.01, 0.1
BETAS = (0.9, 0.95), (0.9, 0.99), (0.9, 0.999)
```

### Category 3: Schedule
```python
# Parameters to tune
WARMUP_RATIO = 0.01, 0.03, 0.05, 0.07, 0.10
MAX_STEPS = 50, 100, 150, 200
```

### Category 4: Architecture
```python
# Parameters to tune
DEPTH = 6, 8, 10, 12
HIDDEN_SIZE = 384, 512, 640, 768
NUM_HEADS = 4, 8, 12, 16
INTERMEDIATE_SIZE = 1024, 2048, 3072
```

### Category 5: Batch Size
```python
# Parameters to tune
DEVICE_BATCH_SIZE = 1, 2, 4
GRADIENT_ACCUMULATION_STEPS = 2, 4, 8
TOTAL_BATCH_SIZE = 4, 8, 16, 32
```

### Category 6: Regularization
```python
# Parameters to tune
DROPOUT = 0.0, 0.05, 0.1, 0.2
ATTENTION_DROPOUT = 0.0, 0.05, 0.1
GRAD_CLIP = 0.5, 1.0, 2.0
```

---

## 📊 Progress Tracking

### Experiment Log
All experiments are logged to: `experiments/experiment_log.jsonl`

Format:
```json
{
  "experiment": 42,
  "change": "Increased LoRA rank from 64 to 128",
  "val_bpb": 1.189,
  "val_loss": 1.123,
  "train_loss": 1.089,
  "improved": true,
  "time_seconds": 308,
  "timestamp": "2026-03-26T14:30:00"
}
```

### Visualization
Progress plot is auto-generated: `experiments/progress.png`

Shows:
- val_bpb over experiments
- Improvement distribution
- Category performance

### Analysis
Run `analysis.ipynb` to:
- Review top improvements
- Analyze by category
- Identify trends
- Generate insights

---

## 🛠️ Quick Start

### Run Single Experiment
```bash
cd arabic-llm

# Edit train_model.py (set EXPERIMENT_ID, EXPERIMENT_CHANGE, params)
# Then run:
python train_model.py
```

### Run Autonomous Agent
```bash
cd arabic-llm

# Run agent (100 experiments)
python run_agent.py --experiments 100 --time-per-exp 300

# Or with human review checkpoints
python run_agent.py --experiments 100 --review-interval 20
```

### Analyze Results
```bash
# Open Jupyter notebook
jupyter notebook analysis.ipynb

# Or view log
cat experiments/experiment_log.jsonl | python -m json.tool

# Or view progress plot
open experiments/progress.png
```

---

## 📚 Resources

### Documentation
- `README.md` - Project overview
- `QUICK_REFERENCE.md` - Quick start guide
- `AUTORESEARCH_IMPROVEMENTS.md` - Improvement plan
- `VERIFICATION_REPORT.md` - Architecture verification

### Analysis Tools
- `analysis.ipynb` - Experiment analysis notebook
- `experiments/experiment_log.jsonl` - Experiment log
- `experiments/progress.png` - Progress visualization

### External Resources
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Arabic LLM Documentation](docs/COMPLETE_DOCUMENTATION.md)

---

## 🎯 Success Criteria

### Short-term (This Week)
- [ ] Reach val_bpb < 1.2
- [ ] Complete 100 experiments
- [ ] Identify optimal LoRA configuration
- [ ] Document learnings

### Medium-term (This Month)
- [ ] Reach val_bpb < 1.1
- [ ] Complete 500 experiments
- [ ] Optimize all hyperparameters
- [ ] Publish results

### Long-term (Next Quarter)
- [ ] Reach val_bpb < 1.0
- [ ] Complete 1000+ experiments
- [ ] Release fine-tuned model
- [ ] Write paper

---

**Status**: 🟢 **ACTIVE**  
**Current Experiment**: #43  
**Next Action**: Fine-tune LoRA rank (try 144, 160)  
**Human Review**: Every 20 experiments

---

**Version**: 2.1.0  
**Last Updated**: March 26, 2026  
**Maintained By**: Arabic LLM Research Team
