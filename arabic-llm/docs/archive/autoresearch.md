# Arabic LLM Autonomous Research Agent

## النظام الآلي لبحث اللغة العربية

Based on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern, adapted for Arabic LLM fine-tuning.

---

## Quick Start

### 1. Prepare Data (One-time)
```bash
cd arabic-llm
python prepare.py
```

### 2. Run Autonomous Agent
```bash
# Run 100 experiments (5 minutes each = ~8 hours)
python agent.py --experiments 100 --time-per-exp 300

# Run overnight (100 experiments)
python agent.py --experiments 100 --time-per-exp 300

# Run single experiment for testing
python agent.py --experiments 1 --time-per-exp 300
```

### 3. Run Single Training Experiment
```bash
# Run training with current configuration
python train.py
```

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  AUTONOMOUS RESEARCH CYCLE (~5 min per experiment)     │
├─────────────────────────────────────────────────────────┤
│  1. Agent reads program.md (instructions)              │
│  2. Proposes train.py modification                     │
│  3. Runs training (5-min time budget)                  │
│  4. Evaluates val_loss (lower = better)                │
│  5. If IMPROVED: keep change, update baseline          │
│  6. If WORSE: discard change, revert                   │
│  7. Log to experiments/experiment_log.jsonl            │
│  8. Repeat (~12 experiments/hour)                      │
└─────────────────────────────────────────────────────────┘
```

### File Structure

```
arabic-llm/
├── prepare.py          # Fixed: Data prep, constants (DO NOT MODIFY)
├── train.py            # Agent modifies: Model, optimizer, training
├── program.md          # Human edits: Agent instructions
├── agent.py            # Autonomous loop controller
├── pyproject.toml      # Dependencies
└── experiments/        # Experiment logs
    └── experiment_log.jsonl
```

---

## What the Agent Can Modify

**ONLY `train.py`** - The agent can change:

| Category | Parameters | Range |
|----------|------------|-------|
| **LoRA** | `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT` | r: 32-256 |
| **Optimizer** | `LEARNING_RATE`, `WEIGHT_DECAY`, `BETAS` | lr: 1e-4 to 1e-3 |
| **Architecture** | `DEPTH`, `HIDDEN_SIZE`, `NUM_HEADS` | depth: 6-16 |
| **Batch Size** | `DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS` | batch: 4-32 |
| **Schedule** | `WARMUP_RATIO`, `MAX_STEPS` | warmup: 0.01-0.10 |
| **Regularization** | `DROPOUT`, `GRAD_CLIP` | dropout: 0.0-0.2 |

**Cannot modify:**
- ❌ `prepare.py` (fixed utilities)
- ❌ Dataset files
- ❌ `program.md` (human instructions)

---

## Experiment Proposals

The agent automatically tests these configurations:

### LoRA Experiments
- r=32, alpha=64 (small capacity)
- r=64, alpha=128 (baseline)
- r=128, alpha=256 (large capacity)
- r=256, alpha=512 (maximum capacity)

### Learning Rate Experiments
- lr=1e-4 (conservative)
- lr=2e-4 (baseline)
- lr=5e-4 (aggressive)
- lr=1e-3 (very aggressive)

### Architecture Experiments
- depth=6, hidden=384 (small)
- depth=8, hidden=512 (baseline)
- depth=12, hidden=768 (large)
- depth=16, hidden=1024 (very large)

### Batch Size Experiments
- batch=4 (small, unstable)
- batch=8 (baseline)
- batch=16 (stable)
- batch=32 (very stable)

---

## Success Metrics

### Primary Metric
**`val_loss`** (validation loss)
- Lower is better
- Target: < 1.0 for Arabic
- Compared to previous best

### Secondary Metrics
- **`train_loss`**: Should decrease over training
- **Training speed**: Steps per minute
- **Memory usage**: GPU VRAM consumption

### Quality Metrics
- **Arabic ratio**: Must stay > 70%
- **Role distribution**: Balanced across tutor, proofreader, poet, muhhaqiq

---

## Experiment Log

Results are logged to `experiments/experiment_log.jsonl`:

```json
{
  "experiment": 42,
  "change": "Increased LoRA rank from 64 to 128",
  "val_loss": 1.189,
  "train_loss": 1.156,
  "improved": true,
  "time_seconds": 312,
  "timestamp": "2026-03-25T14:30:00"
}
```

### View Experiment Log
```bash
# View all experiments
cat experiments/experiment_log.jsonl | python -m json.tool

# Find best experiment
python -c "
import json
with open('experiments/experiment_log.jsonl') as f:
    exps = [json.loads(line) for line in f]
best = min(exps, key=lambda x: x['val_loss'])
print(f'Best: #{best[\"experiment\"]}')
print(f'Change: {best[\"change\"]}')
print(f'val_loss: {best[\"val_loss\"]:.4f}')
"
```

---

## Expected Performance

### Throughput
- **Per experiment:** 5 minutes
- **Per hour:** ~12 experiments
- **Per night (12h):** ~100 experiments
- **Per week:** ~700 experiments

### Hardware Requirements
- **GPU:** NVIDIA RTX 3090/4090 (24GB VRAM)
- **RAM:** 32 GB
- **Storage:** 50 GB free space

### Baseline Performance
- **Initial val_loss:** ~1.5
- **Target val_loss:** < 1.0
- **Expected improvement:** 20-40% over 100 experiments

---

## Troubleshooting

### Issue: GPU Out of Memory

**Solution:**
```python
# In train.py, reduce:
DEVICE_BATCH_SIZE = 1  # From 2
DEPTH = 6  # From 8
```

### Issue: Training Divergence

**Solution:**
```python
# In train.py, reduce:
LEARNING_RATE = 1e-4  # From 2e-4
WARMUP_RATIO = 0.10  # From 0.03
```

### Issue: Slow Training

**Solution:**
```python
# In train.py:
USE_4BIT = True  # Enable quantization
LORA_R = 32  # Reduce rank
```

---

## Advanced Usage

### Resume Interrupted Run
```bash
# Agent automatically resumes from last experiment
python agent.py --resume --experiments 100
```

### Custom Time Budget
```bash
# Run longer experiments for better convergence
python agent.py --experiments 50 --time-per-exp 600
```

### Quick Test
```bash
# Run single 1-minute experiment for testing
python agent.py --experiments 1 --time-per-exp 60
```

---

## Comparison with Karpathy's autoresearch

| Feature | Karpathy's Version | Arabic LLM Version |
|---------|-------------------|-------------------|
| **Base Model** | GPT from scratch | Qwen2.5-7B-Instruct |
| **Task** | Text completion | Instruction tuning |
| **Dataset** | TinyShakespeare | 61,500 Arabic examples |
| **Metric** | val_bpb | val_loss |
| **Method** | Full fine-tuning | QLoRA |
| **Time/Exp** | 5 minutes | 5 minutes |
| **GPU** | H100 | RTX 3090/4090 |

---

## References

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Arabic LLM Documentation](docs/COMPLETE_DOCUMENTATION.md)

---

**Version**: 1.0.0  
**Last Updated**: March 25, 2026  
**Status**: Autonomous Research Ready  
**Expected Throughput**: 100 experiments/night
