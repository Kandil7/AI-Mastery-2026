# Arabic LLM Research Agent - Skill Definition

## Agent Instructions

You are an autonomous AI research agent optimizing Arabic language model fine-tuning.

---

## Your Goal

**Minimize `val_loss`** (validation loss) on Arabic text while maintaining:

- ✅ High Arabic ratio (>70%)
- ✅ Balanced role distribution
- ✅ Zero data loss guarantee
- ✅ Stable training (no divergence)

---

## What You Can Modify

**ONLY MODIFY: `train.py`**

You can change:

| Category | Parameters |
|----------|------------|
| **Architecture** | `DEPTH`, `HIDDEN_SIZE`, `NUM_HEADS`, `INTERMEDIATE_SIZE` |
| **LoRA** | `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `TARGET_MODULES` |
| **Optimizer** | `LEARNING_RATE`, `WEIGHT_DECAY`, `BETAS`, optimizer type |
| **Schedule** | `WARMUP_RATIO`, scheduler type, `MAX_STEPS` |
| **Batch Size** | `DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS` |
| **Regularization** | `DROPOUT`, `ATTENTION_DROPOUT`, `GRAD_CLIP` |
| **Quantization** | `USE_4BIT`, `USE_DOUBLE_QUANT` |

**You CANNOT modify:**
- ❌ `prepare.py` (fixed utilities)
- ❌ Dataset files (`data/jsonl/*.jsonl`)
- ❌ Metadata files
- ❌ This file (`program.md`)

---

## Your Workflow

### Autonomous Research Cycle

```
┌─────────────────────────────────────────────────────────┐
│  AUTONOMOUS EXPERIMENT CYCLE (~5 min)                  │
├─────────────────────────────────────────────────────────┤
│  1. Read current train.py                              │
│  2. Propose ONE specific modification                  │
│  3. Update EXPERIMENT_ID and EXPERIMENT_CHANGE         │
│  4. Run: python train.py                               │
│  5. Wait for training to complete (5 min budget)       │
│  6. Check val_loss metric (lower = better)             │
│  7. Compare to baseline                                │
│  8. If IMPROVED: keep change, update baseline          │
│  9. If WORSE: discard change, revert train.py          │
│ 10. Log experiment to experiments/experiment_log.jsonl │
│ 11. Repeat from step 1                                 │
└─────────────────────────────────────────────────────────┘
```

### Example Experiment

**Experiment #1: Baseline**
```python
EXPERIMENT_ID = 1
EXPERIMENT_CHANGE = "Baseline: LoRA r=64, alpha=128, lr=2e-4"
```

Run training, get `val_loss = 1.234`

**Experiment #2: Increase LoRA rank**
```python
EXPERIMENT_ID = 2
EXPERIMENT_CHANGE = "Increased LoRA rank from 64 to 128"
LORA_R = 128  # Changed from 64
LORA_ALPHA = 256  # Changed from 128 (keep 2x ratio)
```

Run training, get `val_loss = 1.189`

**Result:** ✓ IMPROVED (1.234 → 1.189, -3.6%)

Keep this change and continue experimenting.

---

## Experiment Ideas

### Architecture Experiments

1. **Depth vs Width**
   - Increase `DEPTH` from 8 to 12
   - Decrease `HIDDEN_SIZE` from 512 to 384
   - Test if deeper > wider for Arabic

2. **Attention Heads**
   - Try `NUM_HEADS = 16` (more fine-grained attention)
   - Try `NUM_HEADS = 4` (more efficient)

3. **FFN Size**
   - Try `INTERMEDIATE_SIZE = 4096` (larger capacity)
   - Try `INTERMEDIATE_SIZE = 1024` (more efficient)

### LoRA Experiments

4. **LoRA Rank**
   - `LORA_R = 32` (faster, less capacity)
   - `LORA_R = 64` (baseline)
   - `LORA_R = 128` (more capacity)
   - `LORA_R = 256` (maximum capacity)

5. **LoRA Alpha**
   - Keep `alpha = 2 * rank` (standard scaling)
   - Try `alpha = rank` (more conservative)
   - Try `alpha = 4 * rank` (more aggressive)

6. **Target Modules**
   - Only attention: `["q_proj", "k_proj", "v_proj", "o_proj"]`
   - Only FFN: `["gate_proj", "up_proj", "down_proj"]`
   - All linear (baseline)

### Optimizer Experiments

7. **Learning Rate**
   - `LEARNING_RATE = 1e-4` (conservative)
   - `LEARNING_RATE = 2e-4` (baseline)
   - `LEARNING_RATE = 5e-4` (aggressive)
   - `LEARNING_RATE = 1e-3` (very aggressive)

8. **Warmup Ratio**
   - `WARMUP_RATIO = 0.01` (minimal warmup)
   - `WARMUP_RATIO = 0.03` (baseline)
   - `WARMUP_RATIO = 0.10` (extended warmup)

9. **Weight Decay**
   - `WEIGHT_DECAY = 0.0` (no regularization)
   - `WEIGHT_DECAY = 0.01` (baseline)
   - `WEIGHT_DECAY = 0.1` (strong regularization)

### Batch Size Experiments

10. **Effective Batch Size**
    - `TOTAL_BATCH_SIZE = 4` (small, unstable)
    - `TOTAL_BATCH_SIZE = 8` (baseline)
    - `TOTAL_BATCH_SIZE = 16` (larger, more stable)
    - `TOTAL_BATCH_SIZE = 32` (very large)

### Regularization Experiments

11. **Dropout**
    - `DROPOUT = 0.0` (no dropout, baseline)
    - `DROPOUT = 0.1` (light dropout)
    - `DROPOUT = 0.2` (moderate dropout)

12. **Gradient Clipping**
    - `GRAD_CLIP = 0.5` (aggressive clipping)
    - `GRAD_CLIP = 1.0` (baseline)
    - `GRAD_CLIP = 2.0` (gentle clipping)

---

## Success Metrics

### Primary Metric

**`val_loss`** (validation loss)
- Lower is better
- Compare to previous best
- Target: < 1.0 for Arabic

### Secondary Metrics

**`train_loss`** (training loss)
- Should decrease over training
- Gap between train/val indicates overfitting

**Training Speed**
- Steps per minute
- Target: > 20 steps/min

**Memory Usage**
- GPU memory consumption
- Target: < 20 GB (leave room for larger batches)

### Quality Metrics

**Arabic Ratio**
- Must stay > 70%
- Verified by `prepare.py`

**Role Distribution**
- Should match target distribution
- tutor: 35%, proofreader: 25%, poet: 20%, muhhaqiq: 15%

---

## Logging Format

Every experiment is logged to `experiments/experiment_log.jsonl`:

```json
{
  "experiment": 1,
  "change": "Baseline: LoRA r=64, alpha=128, lr=2e-4",
  "val_loss": 1.234,
  "train_loss": 1.156,
  "improved": false,
  "time_seconds": 312,
  "timestamp": "2026-03-25T14:30:00"
}
```

---

## Constraints

### Hardware

- **GPU:** NVIDIA RTX 3090/4090 (24GB VRAM)
- **CPU:** AMD Ryzen 9 / Intel i9
- **RAM:** 32 GB
- **Storage:** 100 GB free space

### Time

- **Per experiment:** 5 minutes (wall clock)
- **Per night:** ~100 experiments (12 hours)
- **Total project:** 1 week

### Model

- **Base:** Qwen2.5-7B-Instruct
- **Quantization:** 4-bit (QLoRA)
- **Max sequence:** 2048 tokens
- **Vocab size:** 151,936

### Dataset

- **Training:** 55,350 examples (90%)
- **Validation:** 3,075 examples (5%)
- **Test:** 3,075 examples (5%)
- **Total:** 61,500 examples

---

## Best Practices

### 1. Change One Thing at a Time

Isolate variables to understand impact:

```python
# ✓ GOOD: Single change
LORA_R = 128  # Only changed this

# ✗ BAD: Multiple changes
LORA_R = 128
LEARNING_RATE = 5e-4
DEPTH = 12
```

### 2. Start with High-Impact Changes

Priority order:
1. LoRA rank (biggest impact on capacity)
2. Learning rate (affects convergence)
3. Batch size (affects stability)
4. Architecture (affects everything)

### 3. Keep Detailed Notes

Update `EXPERIMENT_CHANGE` with clear description:

```python
# ✓ GOOD: Clear description
EXPERIMENT_CHANGE = "Increased LoRA rank from 64 to 128"

# ✗ BAD: Vague description
EXPERIMENT_CHANGE = "Modified LoRA config"
```

### 4. Monitor for Divergence

Watch for:
- `val_loss` increasing over training
- `train_loss` NaN or Inf
- GPU out of memory errors

If divergence occurs:
- Reduce learning rate
- Increase warmup ratio
- Decrease batch size
- Add gradient clipping

### 5. Save Best Checkpoints

Best model is automatically saved to:
```
checkpoints/exp{EXPERIMENT_ID}/
```

---

## Quick Reference

### Run Data Preparation
```bash
cd arabic-llm
python prepare.py
```

### Run Training Experiment
```bash
python train.py
```

### View Experiment Log
```bash
cat experiments/experiment_log.jsonl | python -m json.tool
```

### Compare Experiments
```python
import json

with open('experiments/experiment_log.jsonl') as f:
    experiments = [json.loads(line) for line in f]

# Sort by val_loss
best = min(experiments, key=lambda x: x['val_loss'])
print(f"Best experiment: #{best['experiment']}")
print(f"Change: {best['change']}")
print(f"val_loss: {best['val_loss']:.4f}")
```

---

## Troubleshooting

### Issue: GPU Out of Memory

**Solution:**
```python
# Reduce batch size
DEVICE_BATCH_SIZE = 1  # From 2

# Or reduce model size
DEPTH = 6  # From 8
HIDDEN_SIZE = 384  # From 512
```

### Issue: Training Divergence

**Solution:**
```python
# Reduce learning rate
LEARNING_RATE = 1e-4  # From 2e-4

# Increase warmup
WARMUP_RATIO = 0.10  # From 0.03

# Add gradient clipping
GRAD_CLIP = 0.5  # From 1.0
```

### Issue: Slow Training

**Solution:**
```python
# Enable 4-bit quantization
USE_4BIT = True

# Reduce sequence length
MAX_SEQ_LEN = 1024  # From 2048

# Reduce LoRA rank
LORA_R = 32  # From 64
```

---

## Contact

For questions about this system, see:
- `docs/COMPLETE_DOCUMENTATION.md`
- `docs/implementation.md`
- `QUICK_REFERENCE.md`

---

**Version:** 1.0.0  
**Last Updated:** March 25, 2026  
**Status:** Autonomous Research Ready
