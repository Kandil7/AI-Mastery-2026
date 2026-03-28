# Arabic LLM Autonomous Research System

## النظام الآلي لبحث اللغة العربية

# Based on Karpathy's autoresearch pattern, adapted for Arabic LLM fine-tuning.

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime



## System Architecture

"""
arabic-llm/
├── prepare.py          # Data preparation, constants, utilities (FIXED)
├── train.py            # Model, optimizer, training loop (AGENT MODIFIES)
├── program.md          # Agent instructions/skill definition (HUMAN MODIFIES)
├── pyproject.toml      # Dependencies
├── analysis.ipynb      # Analysis notebook
└── experiments/        # Experiment logs and results
```

---

## Core Components

### 1. `program.md` - Agent Instructions

Defines the "skill" of the Arabic LLM research agent:

```markdown
# Arabic LLM Research Agent

You are an autonomous AI research agent optimizing Arabic language model fine-tuning.

## Your Goal
Minimize `val_loss` (validation loss) on Arabic text while maintaining:
- High Arabic ratio (>70%)
- Balanced role distribution (tutor 35%, proofreader 25%, poet 20%, muhhaqiq 15%)
- Zero data loss guarantee

## What You Can Modify
- `train.py` ONLY
- You can change: architecture, hyperparameters, optimizer, learning rate, batch size
- You CANNOT modify: `prepare.py`, dataset files, metadata

## Your Workflow
1. Read current `train.py` to understand architecture
2. Propose ONE specific modification
3. Run training for 5 minutes (time budget)
4. Check `val_loss` metric (lower = better)
5. If improved: keep change, log success
6. If worse: discard change, try different approach
7. Repeat autonomously

## Constraints
- Single GPU (NVIDIA RTX 3090/4090, 24GB VRAM)
- Base model: Qwen2.5-7B-Instruct
- Dataset: 61,500 Arabic training examples
- Maximum sequence length: 2048 tokens
- Time budget per experiment: 5 minutes

## Success Metrics
- Primary: `val_loss` (validation loss)
- Secondary: `train_loss` (training loss)
- Quality: Arabic ratio, role balance, skill coverage

## Experiment Log Format
```
Experiment #42
Change: Increased LoRA rank from 64 to 128
Result: val_loss 1.234 → 1.189 (✓ IMPROVED)
Time: 5m 12s
```
```

---

### 2. `prepare.py` - Fixed Data Preparation

Data preparation and utilities for Arabic LLM training.
DO NOT MODIFY - This file is fixed.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# =============================================================================
# CONSTANTS - DO NOT MODIFY
# =============================================================================

# Dataset paths
DATASETS_DIR = Path("datasets")
EXTRACTED_BOOKS = DATASETS_DIR / "extracted_books"
METADATA_DIR = DATASETS_DIR / "metadata"
SYSTEM_BOOKS = DATASETS_DIR / "system_book_datasets"

# Training data
TRAINING_DATA = Path("data/jsonl/train.jsonl")
VAL_DATA = Path("data/jsonl/val.jsonl")
TEST_DATA = Path("data/jsonl/test.jsonl")

# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936  # Qwen2.5 vocab size

# Evaluation
EVAL_TOKENS = 100000  # Tokens for validation
EVAL_STEPS = 100

# Quality thresholds
MIN_ARABIC_RATIO = 0.70
MAX_DIACRITICS_RATIO = 0.30

# Role distribution targets
ROLE_DISTRIBUTION = {
    "tutor": 0.35,
    "proofreader": 0.25,
    "poet": 0.20,
    "muhhaqiq": 0.15,
    "assistant_general": 0.05,
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file into list of dicts"""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def load_training_data() -> Tuple[List[Dict], List[Dict]]:
    """Load training and validation datasets"""
    train_data = load_jsonl(TRAINING_DATA)
    val_data = load_jsonl(VAL_DATA)
    
    print(f"Loaded {len(train_data):,} training examples")
    print(f"Loaded {len(val_data):,} validation examples")
    
    return train_data, val_data


def verify_data_quality(examples: List[Dict]) -> Dict:
    """Verify dataset quality metrics"""
    stats = {
        "total": len(examples),
        "arabic_ratio": [],
        "role_distribution": {},
        "skill_distribution": {},
    }
    
    for ex in examples:
        # Arabic ratio
        text = ex.get('output', '') + ex.get('input', '')
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        stats["arabic_ratio"].append(arabic_chars / len(text) if text else 0)
        
        # Role distribution
        role = ex.get('role', 'unknown')
        stats["role_distribution"][role] = stats["role_distribution"].get(role, 0) + 1
        
        # Skill distribution
        for skill in ex.get('skills', []):
            stats["skill_distribution"][skill] = stats["skill_distribution"].get(skill, 0) + 1
    
    # Compute averages
    stats["avg_arabic_ratio"] = sum(stats["arabic_ratio"]) / len(stats["arabic_ratio"])
    
    # Normalize distributions
    total = stats["total"]
    stats["role_distribution"] = {k: v/total for k, v in stats["role_distribution"].items()}
    stats["skill_distribution"] = {k: v/total for k, v in stats["skill_distribution"].items()}
    
    return stats


# =============================================================================
# TOKENIZATION
# =============================================================================

def get_tokenizer():
    """Get tokenizer for Qwen2.5-7B-Instruct"""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =============================================================================
# EVALUATION
# =============================================================================

def compute_val_loss(model, val_data, tokenizer, max_tokens=EVAL_TOKENS) -> float:
    """Compute validation loss on subset of validation data"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    model.eval()
    
    # Tokenize validation data
    texts = [ex['output'] for ex in val_data[:1000]]
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors='pt'
    )
    
    # Compute loss
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), 4):
            batch_start = i
            batch_end = min(i + 4, len(texts))
            
            input_ids = encodings.input_ids[batch_start:batch_end].cuda()
            attention_mask = encodings.attention_mask[batch_start:batch_end].cuda()
            labels = input_ids.clone()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Stop if we've processed enough tokens
            if num_batches * 4 * MAX_SEQ_LEN >= max_tokens:
                break
    
    avg_loss = total_loss / num_batches
    return avg_loss


# =============================================================================
# LOGGING
# =============================================================================

def log_experiment(experiment_num: int, change: str, val_loss: float, 
                   train_loss: float, improved: bool, time_seconds: float):
    """Log experiment results to file"""
    log_dir = Path("experiments")
    log_dir.mkdir(exist_ok=True)
    
    log_entry = {
        "experiment": experiment_num,
        "change": change,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "improved": improved,
        "time_seconds": time_seconds,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Append to log file
    log_file = log_dir / "experiment_log.jsonl"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    # Print summary
    status = "✓ IMPROVED" if improved else "✗ WORSE"
    print(f"\nExperiment #{experiment_num}")
    print(f"Change: {change}")
    print(f"Result: val_loss {val_loss:.4f} (train_loss: {train_loss:.4f}) [{status}]")
    print(f"Time: {time_seconds/60:.1f}m")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Arabic LLM - Data Preparation")
    print("=" * 70)
    
    # Load data
    train_data, val_data = load_training_data()
    
    # Verify quality
    stats = verify_data_quality(train_data)
    
    print(f"\nDataset Quality:")
    print(f"  Total examples: {stats['total']:,}")
    print(f"  Avg Arabic ratio: {stats['avg_arabic_ratio']:.1%}")
    print(f"  Role distribution: {stats['role_distribution']}")
    print(f"  Skills covered: {len(stats['skill_distribution'])}")
    
    # Check thresholds
    if stats['avg_arabic_ratio'] < MIN_ARABIC_RATIO:
        print(f"\n⚠️  WARNING: Arabic ratio below threshold!")
    
    print("\n✓ Data preparation complete")
