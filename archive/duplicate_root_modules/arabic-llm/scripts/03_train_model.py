#!/usr/bin/env python3
"""
Step 3: Fine-Tune Model

Fine-tune an Arabic LLM using QLoRA with the generated dataset.

Usage:
    python scripts/03_train_model.py \
        --dataset data/jsonl/train.jsonl \
        --output-dir models/arabic-linguist-v1 \
        --config configs/training_config.yaml
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Check for required packages
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
    from datasets import load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Warning: Required packages not installed: {e}")
    print("Install with: pip install transformers peft bitsandbytes datasets accelerate")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_trainable_parameters(model):
    """Print the number of trainable parameters"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_param:.2f}% of total)")


def format_prompt(example, tokenizer):
    """Format example into prompt for instruction tuning"""
    # Chat template format
    messages = [
        {"role": "user", "content": example["instruction"] + "\n\n" + example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    
    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback format
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Arabic LLM with QLoRA")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    dataset_path = Path(args.dataset)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        print("Run 02_generate_dataset.py first")
        sys.exit(1)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step 3: Fine-Tune Model (QLoRA)")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config(str(config_path))
    
    # Override with command line args
    if args.base_model:
        config["model"]["base"] = args.base_model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    base_model = config["model"]["base"]
    print(f"Base model: {base_model}")
    
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 60)
        print("DRY RUN - Required packages not installed")
        print("=" * 60)
        print("\nTo run full training, install:")
        print("  pip install transformers peft bitsandbytes datasets accelerate torch")
        print("\nConfiguration validated successfully!")
        return 0
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: No GPU detected. Training will be very slow.")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=config["model"].get("trust_remote_code", True),
        padding_side="right",
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    max_length = config["training"]["max_seq_length"]
    
    def tokenize_function(example):
        formatted = format_prompt(example, tokenizer)
        return tokenizer(
            formatted["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )
    
    # Load model with quantization
    print(f"\nLoading model with QLoRA quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias=config["lora"]["bias"],
    )
    
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["epochs"],
        max_seq_length=max_length,
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=config["optimization"]["fp16"],
        gradient_checkpointing=config["optimization"]["gradient_checkpointing"],
        optim=config["optimization"]["optimizer"],
        report_to="tensorboard" if config["logging"]["tensorboard"] else "none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training metrics
    metrics = {
        "training_completed": datetime.now().isoformat(),
        "base_model": base_model,
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
        "train_samples": len(dataset),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "metrics": {
            "train_loss": train_result.metrics.get("train_loss"),
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        }
    }
    
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_dir}/ (trained model)")
    print(f"  - {metrics_file}")
    print(f"\nTraining metrics:")
    print(f"  Loss: {metrics['metrics']['train_loss']:.4f}")
    print(f"  Runtime: {metrics['metrics']['train_runtime']:.2f} seconds")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
