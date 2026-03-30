"""
Arabic LLM Training Script

AGENT: You can modify this file to experiment with:
- Model architecture (DEPTH, hidden size, attention heads)
- Optimizer (AdamW, Muon, learning rate, betas)
- Learning rate schedule (warmup, decay, scheduler type)
- Batch size and gradient accumulation
- LoRA configuration (rank, alpha, dropout)
- Regularization (dropout, weight decay)

DO NOT MODIFY: prepare.py, dataset files, or metadata

Goal: Minimize val_loss while maintaining quality metrics
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

# Optional PEFT import (install with: pip install peft)
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not installed. Install with: pip install peft")

# Import fixed utilities
from prepare import (
    load_training_data,
    load_jsonl,
    verify_data_quality,
    get_tokenizer,
    compute_val_loss,
    log_experiment,
    BASE_MODEL,
    MAX_SEQ_LEN,
    ROLE_DISTRIBUTION,
)


# =============================================================================
# HYPERPARAMETERS - AGENT CAN MODIFY THESE
# =============================================================================

# Time budget (5 minutes for training, excludes startup)
TIME_BUDGET_SECONDS = 300

# Model architecture
DEPTH = 8  # Number of transformer layers
HIDDEN_SIZE = 512  # Hidden dimension
NUM_HEADS = 8  # Attention heads
INTERMEDIATE_SIZE = 2048  # FFN intermediate size

# LoRA configuration
LORA_R = 64  # LoRA rank
LORA_ALPHA = 128  # LoRA alpha
LORA_DROPOUT = 0.05  # LoRA dropout

# Target modules for LoRA
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Batch size and gradient accumulation
DEVICE_BATCH_SIZE = 2  # Batch size per device
GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation steps
TOTAL_BATCH_SIZE = DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # Effective batch size

# Learning rate and optimizer
LEARNING_RATE = 2.0e-4  # Peak learning rate
WEIGHT_DECAY = 0.01  # Weight decay
WARMUP_RATIO = 0.03  # Warmup fraction
BETAS = (0.9, 0.999)  # Adam betas

# Training
MAX_STEPS = 100  # Maximum training steps (will be cut by time budget)
GRAD_CLIP = 1.0  # Gradient clipping

# Regularization
DROPOUT = 0.0  # Model dropout
ATTENTION_DROPOUT = 0.0  # Attention dropout

# Quantization
USE_4BIT = True  # Use 4-bit quantization
USE_DOUBLE_QUANT = True  # Use double quantization

# Logging
LOG_INTERVAL = 10  # Log every N steps
EVAL_INTERVAL = 50  # Evaluate every N steps

# Experiment tracking
EXPERIMENT_ID = 1  # Increment for each experiment
EXPERIMENT_CHANGE = "Baseline configuration"  # Describe your change here


# =============================================================================
# DATASET
# =============================================================================

class ArabicInstructionDataset(Dataset):
    """Dataset for Arabic instruction tuning"""
    
    def __init__(self, examples, tokenizer, max_length=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as instruction prompt
        prompt = f"""### Instruction:
{example.get('instruction', '')}

### Input:
{example.get('input', '')}

### Output:
{example.get('output', '')}"""
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding.input_ids.squeeze(0),
            'attention_mask': encoding.attention_mask.squeeze(0),
            'labels': encoding.input_ids.squeeze(0).clone()
        }


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer():
    """Load base model with quantization and LoRA"""
    
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()
    
    print("Loading base model...")
    
    # Quantization config
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        )
    else:
        quantization_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not USE_4BIT else None,
    )
    
    # Prepare for training
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# TRAINING
# =============================================================================

def train():
    """Main training loop"""
    
    print("=" * 70)
    print("Arabic LLM - Training")
    print("=" * 70)
    print(f"Experiment #{EXPERIMENT_ID}: {EXPERIMENT_CHANGE}")
    print("=" * 70)
    
    # Load data
    print("\nLoading datasets...")
    train_examples = load_jsonl(Path("data/jsonl/train.jsonl"))
    val_examples = load_jsonl(Path("data/jsonl/val.jsonl"))
    
    print(f"  Train: {len(train_examples):,} examples")
    print(f"  Val: {len(val_examples):,} examples")
    
    # Verify quality
    stats = verify_data_quality(train_examples)
    print(f"  Arabic ratio: {stats['avg_arabic_ratio']:.1%}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ArabicInstructionDataset(train_examples, get_tokenizer())
    val_dataset = ArabicInstructionDataset(val_examples, get_tokenizer())
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=DEVICE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=DEVICE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Learning rate scheduler
    num_warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=MAX_STEPS,
    )
    
    # Training loop
    print("\nStarting training...")
    print(f"  Time budget: {TIME_BUDGET_SECONDS/60:.1f} minutes")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Batch size: {TOTAL_BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA rank: {LORA_R}")
    print()
    
    model.train()
    start_time = time.time()

    step = 0
    total_loss = 0.0
    best_val_loss = float('inf')
    elapsed = 0.0  # Initialize to avoid unbound variable warning

    for epoch in range(1):  # Single epoch, cut by time budget
        for batch in train_loader:
            step_start = time.time()
            
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > TIME_BUDGET_SECONDS:
                print(f"\n⏰ Time budget reached ({elapsed/60:.1f}m)")
                break
            
            # Move batch to GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            total_loss += outputs.loss.item()
            
            # Gradient step
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if (step + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / LOG_INTERVAL
                elapsed = time.time() - start_time
                steps_per_min = step / elapsed if elapsed > 0 else 0
                
                print(f"Step {step+1}/{MAX_STEPS} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                      f"Speed: {steps_per_min*60:.1f} steps/min")
                
                total_loss = 0.0
            
            # Evaluation
            if (step + 1) % EVAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = compute_val_loss(model, val_examples, tokenizer)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  ✓ New best val_loss: {val_loss:.4f}")
                    
                    # Save checkpoint
                    checkpoint_dir = Path(f"checkpoints/exp{EXPERIMENT_ID}")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                
                model.train()
            
            step += 1
        
        if elapsed > TIME_BUDGET_SECONDS:
            break
    
    # Final evaluation
    total_time = time.time() - start_time
    model.eval()
    
    with torch.no_grad():
        final_val_loss = compute_val_loss(model, val_examples, tokenizer)
    
    # Compute average training loss
    avg_train_loss = total_loss / max(step, 1)
    
    # Determine if improved (compare to baseline)
    baseline_val_loss = 1.5  # Placeholder - load from previous experiment log
    improved = final_val_loss < baseline_val_loss
    
    # Log experiment
    log_experiment(
        experiment_num=EXPERIMENT_ID,
        change=EXPERIMENT_CHANGE,
        val_loss=final_val_loss,
        train_loss=avg_train_loss,
        improved=improved,
        time_seconds=total_time,
    )
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Final val_loss: {final_val_loss:.4f}")
    print(f"Final train_loss: {avg_train_loss:.4f}")
    print(f"Total time: {total_time/60:.1f}m")
    print(f"Steps completed: {step}")
    print(f"Status: {'✓ IMPROVED' if improved else '✗ NOT IMPROVED'}")
    print("=" * 70)
    
    return final_val_loss


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    val_loss = train()
    
    # Exit with appropriate code
    # 0 = success, 1 = failure
    exit(0 if val_loss > 0 else 1)
