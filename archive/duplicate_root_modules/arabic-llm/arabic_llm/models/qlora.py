"""
Arabic LLM - QLoRA Training Utilities

Utilities for QLoRA (Quantized Low-Rank Adaptation) training.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QLoRAConfig:
    """QLoRA configuration"""
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list = None
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


def create_qlora_config(
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> QLoRAConfig:
    """
    Create QLoRA configuration.
    
    Args:
        r: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
        target_modules: Target modules for LoRA
        
    Returns:
        QLoRAConfig object
    """
    return QLoRAConfig(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )


def load_qlora_model(
    model_name: str,
    qlora_config: QLoRAConfig,
    use_4bit: bool = True,
):
    """
    Load model with QLoRA configuration.
    
    Args:
        model_name: Base model name
        qlora_config: QLoRA configuration
        use_4bit: Use 4-bit quantization
        
    Returns:
        Model and tokenizer
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from transformers import BitsAndBytesConfig
        import torch
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Quantization config
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # LoRA config
        lora_config = LoraConfig(
            r=qlora_config.r,
            lora_alpha=qlora_config.alpha,
            target_modules=qlora_config.target_modules,
            lora_dropout=qlora_config.dropout,
            bias="none",
            task_type=qlora_config.task_type,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        return model, tokenizer
        
    except ImportError as e:
        raise ImportError(
            "Required packages not installed. "
            "Install with: pip install transformers peft bitsandbytes torch"
        ) from e


def train_qlora(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    config: Dict[str, Any],
):
    """
    Train model with QLoRA.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        
    Returns:
        Trained model
    """
    try:
        from transformers import TrainingArguments, Trainer
        from peft import prepare_model_for_kbit_training
        
        # Prepare model
        model = prepare_model_for_kbit_training(model)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.get("output_dir", "output"),
            per_device_train_batch_size=config.get("batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation", 4),
            learning_rate=config.get("learning_rate", 2e-4),
            num_train_epochs=config.get("epochs", 3),
            max_seq_length=config.get("max_seq_length", 2048),
            warmup_ratio=config.get("warmup_ratio", 0.03),
            lr_scheduler_type=config.get("lr_scheduler", "cosine"),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 500),
            evaluation_strategy="steps",
            eval_steps=config.get("eval_steps", 500),
            gradient_checkpointing=True,
            optim="adamw_torch",
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )
        
        # Train
        trainer.train()
        
        return model
        
    except ImportError as e:
        raise ImportError(
            "Required packages not installed. "
            "Install with: pip install transformers accelerate"
        ) from e
