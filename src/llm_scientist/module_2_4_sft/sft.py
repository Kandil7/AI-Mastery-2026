"""
Supervised Fine-Tuning - Module 2.4.1

Production-ready SFT implementation:
- Configuration
- Dataset preparation
- Training loop
- Checkpointing
- Evaluation

References:
- "Fine-Tuning Language Models from Human Preferences" (Ziegler et al., 2019)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""
    # Model settings
    model_name_or_path: str = ""
    trust_remote_code: bool = False
    
    # Data settings
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    max_seq_length: int = 2048
    padding_side: str = "right"
    
    # Training settings
    output_dir: str = "./sft_output"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Optimization settings
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    
    # Logging settings
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Other settings
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = True
    report_to: str = "none"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class SFTDataCollator:
    """Data collator for SFT."""
    tokenizer: Any
    max_length: int = 2048
    padding: str = "longest"
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        import transformers
        
        # Use transformers default_data_collator if available
        try:
            from transformers import default_data_collator
            return default_data_collator(features)
        except ImportError:
            pass
        
        # Manual collation
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }
        
        for feature in features:
            input_ids = feature.get('input_ids', [])
            attention_mask = feature.get('attention_mask', [1] * len(input_ids))
            labels = feature.get('labels', input_ids.copy())
            
            # Truncate if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                labels = labels[:self.max_length]
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)
        
        # Pad and convert to tensors
        result = {}
        for key, values in batch.items():
            max_len = max(len(v) for v in values)
            
            if self.pad_to_multiple_of:
                max_len = ((max_len + self.pad_to_multiple_of - 1) // 
                          self.pad_to_multiple_of) * self.pad_to_multiple_of
            
            padded = []
            for v in values:
                pad_length = max_len - len(v)
                if self.padding == "right":
                    if key == 'labels':
                        padded_v = v + [self.label_pad_token_id] * pad_length
                    else:
                        padded_v = v + [0] * pad_length
                else:
                    if key == 'labels':
                        padded_v = [self.label_pad_token_id] * pad_length + v
                    else:
                        padded_v = [0] * pad_length + v
                padded.append(padded_v)
            
            result[key] = torch.tensor(padded, dtype=torch.long)
        
        return result


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    
    Args:
        data: List of training examples
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        format_func: Optional function to format examples
        
    Example:
        >>> dataset = SFTDataset(data, tokenizer, max_length=2048)
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 2048,
        format_func: Optional[Callable] = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_func = format_func or self._default_format
    
    def _default_format(self, example: Dict[str, Any]) -> str:
        """Default formatting function."""
        if 'text' in example:
            return example['text']
        
        # Format as instruction-response
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        if input_text:
            return f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
        else:
            return f"Instruction: {instruction}\nResponse: {output}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Format text
        text = self.format_func(example)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels (copy of input_ids for causal LM)
        input_ids = encoded['input_ids']
        labels = input_ids.copy()
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
        }


class SFTTrainer:
    """
    Trainer for Supervised Fine-Tuning.
    
    Args:
        model: Model to train
        config: SFT configuration
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        
    Example:
        >>> trainer = SFTTrainer(model, config, tokenizer, train_dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SFTConfig,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        optimizers: Optional[Tuple[Optimizer, LRScheduler]] = None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Data collator
        self.data_collator = data_collator or SFTDataCollator(
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
        )
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimizer and scheduler
        if optimizers:
            self.optimizer, self.scheduler = optimizers
        else:
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Mixed precision
        self.use_amp = config.fp16 or config.bf16
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        # Move model to device
        self.model.to(self.device)
        
        # Gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer."""
        # Separate weight decay parameters
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self) -> LRScheduler:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        num_training_steps = int(
            len(self.train_dataset) / 
            (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps) *
            self.config.num_train_epochs
        )
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=0,
            )
        elif self.config.lr_scheduler_type == "linear":
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=num_training_steps,
            )
        else:
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
            )
        
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=num_warmup_steps,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[num_warmup_steps],
            )
        
        return scheduler
    
    def _get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create dataloader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
        )
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Perform a single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.detach()
    
    def optimizer_step(self) -> None:
        """Perform optimizer step."""
        # Gradient clipping
        if self.use_amp and self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        # Optimizer step
        if self.use_amp and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Scheduler step
        self.scheduler.step()
        
        # Zero gradients
        self.model.zero_grad()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataset is None:
            return {}
        
        self.model.eval()
        
        eval_dataloader = self._get_dataloader(
            self.eval_dataset,
            self.config.per_device_eval_batch_size,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        self.model.train()
        
        return {'eval_loss': avg_loss}
    
    def save_checkpoint(self, output_dir: Optional[str] = None) -> None:
        """Save checkpoint."""
        output_dir = output_dir or os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}",
        )
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
        }
        
        if self.scaler:
            training_state['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(training_state, os.path.join(output_dir, 'training_state.pt'))
        
        # Save config
        with open(os.path.join(output_dir, 'sft_config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Checkpoint saved to {output_dir}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Training metrics
        """
        logger.info("Starting SFT training...")
        
        # Create dataloader
        train_dataloader = self._get_dataloader(
            self.train_dataset,
            self.config.per_device_train_batch_size,
            shuffle=True,
        )
        
        # Calculate training steps
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = int(num_update_steps_per_epoch * self.config.num_train_epochs)
        
        # Training loop
        progress_bar = tqdm(total=num_training_steps, desc="Training")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        training_metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
        }
        
        accumulation_step = 0
        
        for epoch in range(int(self.config.num_train_epochs)):
            self.epoch = epoch
            
            for batch in train_dataloader:
                # Training step
                loss = self.training_step(batch)
                accumulation_step += 1
                
                # Optimizer step
                if accumulation_step % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Log metrics
                    training_metrics['train_loss'].append(loss.item() * self.config.gradient_accumulation_steps)
                    training_metrics['learning_rate'].append(self.scheduler.get_last_lr()[0])
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(
                            f"Step {self.global_step}: "
                            f"loss={loss.item() * self.config.gradient_accumulation_steps:.4f}, "
                            f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                        )
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        training_metrics['eval_loss'].append(eval_metrics.get('eval_loss', 0))
                        
                        logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 0):.4f}")
                        
                        # Save best model
                        if eval_metrics.get('eval_loss', float('inf')) < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['eval_loss']
                            self.save_checkpoint(os.path.join(self.config.output_dir, 'best'))
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                
                # Early stopping check
                if self.global_step >= num_training_steps:
                    break
            
            # Save epoch checkpoint
            self.save_checkpoint(os.path.join(self.config.output_dir, f'epoch-{epoch}'))
        
        progress_bar.close()
        
        # Save final checkpoint
        self.save_checkpoint()
        
        logger.info("Training complete!")
        
        return training_metrics


class SFTPipeline:
    """
    Complete SFT Pipeline.
    
    Orchestrates the full fine-tuning workflow:
    1. Data loading
    2. Model preparation
    3. Training
    4. Evaluation
    5. Export
    
    Example:
        >>> pipeline = SFTPipeline(config)
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model(self) -> nn.Module:
        """Load model for fine-tuning."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model from {self.config.model_name_or_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                padding_side=self.config.padding_side,
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else None),
            )
            
            # Resize embeddings if needed
            if len(self.tokenizer) > self.model.get_input_embeddings().weight.shape[0]:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            return self.model
        
        except ImportError:
            logger.error("Transformers library not installed. Install with: pip install transformers")
            raise
    
    def load_data(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Load training and evaluation data."""
        # Load training data
        train_data = self._load_json_data(self.config.train_data_path)
        
        # Load eval data
        eval_data = None
        if self.config.eval_data_path:
            eval_data = self._load_json_data(self.config.eval_data_path)
        
        # Create datasets
        train_dataset = SFTDataset(
            train_data,
            self.tokenizer,
            max_length=self.config.max_seq_length,
        )
        
        eval_dataset = None
        if eval_data:
            eval_dataset = SFTDataset(
                eval_data,
                self.tokenizer,
                max_length=self.config.max_seq_length,
            )
        
        logger.info(f"Loaded {len(train_dataset)} training examples")
        if eval_dataset:
            logger.info(f"Loaded {len(eval_dataset)} evaluation examples")
        
        return train_dataset, eval_dataset
    
    def _load_json_data(self, path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        path = Path(path)
        
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the full SFT pipeline.
        
        Returns:
            Training results
        """
        logger.info("Starting SFT pipeline...")
        
        # Load model and tokenizer
        self.load_model()
        
        # Load data
        train_dataset, eval_dataset = self.load_data()
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        training_metrics = self.trainer.train()
        
        # Final evaluation
        final_eval = self.trainer.evaluate()
        
        results = {
            'training_metrics': training_metrics,
            'final_eval': final_eval,
            'best_eval_loss': self.trainer.best_eval_loss,
            'output_dir': self.config.output_dir,
        }
        
        # Save results
        results_path = Path(self.config.output_dir) / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline complete. Results saved to {results_path}")
        
        return results
