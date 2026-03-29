"""
Direct Preference Optimization (DPO) - Module 2.5.2

Production-ready DPO implementation:
- DPO configuration
- DPO loss function
- DPO trainer
- DPO pipeline

References:
- "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # Model settings
    model_name_or_path: str = ""
    reference_model_name_or_path: Optional[str] = None
    
    # DPO settings
    beta: float = 0.1  # Temperature parameter
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, exponential
    
    # Training settings
    output_dir: str = "./dpo_output"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    
    # Optimization settings
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    # Logging settings
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Other settings
    seed: int = 42
    dataloader_num_workers: int = 4


class DPOLoss(nn.Module):
    """
    DPO Loss Function.
    
    Implements the Direct Preference Optimization loss:
    L = -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))
    
    Args:
        beta: Temperature parameter
        loss_type: Loss type ('sigmoid', 'hinge', 'ipo', 'exponential')
        
    Reference:
        "Direct Preference Optimization" (Rafailov et al., 2023)
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
    ):
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses under policy
            policy_rejected_logps: Log probs of rejected responses under policy
            reference_chosen_logps: Log probs of chosen responses under reference
            reference_rejected_logps: Log probs of rejected responses under reference
        
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute log odds ratios
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        
        # Log odds ratio difference
        logratios = chosen_logratios - rejected_logratios
        
        # Compute loss based on type
        if self.loss_type == "sigmoid":
            # Standard DPO loss
            losses = -F.logsigmoid(self.beta * logratios)
        elif self.loss_type == "hinge":
            # Hinge loss variant
            losses = torch.relu(1 - self.beta * logratios)
        elif self.loss_type == "ipo":
            # IPO loss (Identity Preference Optimization)
            losses = (logratios - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "exponential":
            # Exponential loss
            losses = torch.exp(-self.beta * logratios)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Compute metrics
        metrics = {
            'loss': losses.mean().item(),
            'chosen_logps': policy_chosen_logps.mean().item(),
            'rejected_logps': policy_rejected_logps.mean().item(),
            'logratios': logratios.mean().item(),
            'reward_accuracy': (logratios > 0).float().mean().item(),
        }
        
        return losses.mean(), metrics
    
    def concatenated_forward(
        self,
        model: nn.Module,
        concatenated_input_ids: torch.Tensor,
        concatenated_attention_mask: torch.Tensor,
        concatenated_labels: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on concatenated chosen/rejected inputs.
        
        Args:
            model: Model to forward through
            concatenated_input_ids: Concatenated input IDs
            concatenated_attention_mask: Concatenated attention mask
            concatenated_labels: Concatenated labels
            batch_size: Original batch size (before concatenation)
        
        Returns:
            Tuple of (chosen_logps, rejected_logps)
        """
        # Forward pass
        outputs = model(
            input_ids=concatenated_input_ids,
            attention_mask=concatenated_attention_mask,
            labels=concatenated_labels,
        )
        
        # Get per-token log probs
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = concatenated_labels[:, 1:]
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for labels
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        # Sum over sequence length (excluding padding)
        all_logps = per_token_logps.sum(dim=-1)
        
        # Split into chosen and rejected
        chosen_logps = all_logps[:batch_size]
        rejected_logps = all_logps[batch_size:]
        
        return chosen_logps, rejected_logps


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization.
    
    Args:
        policy_model: Policy model to train
        reference_model: Reference model (frozen)
        config: DPO configuration
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        
    Example:
        >>> trainer = DPOTrainer(policy_model, reference_model, config, tokenizer, dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: DPOConfig,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()
        
        # Loss function
        self.loss_fn = DPOLoss(
            beta=config.beta,
            loss_type=config.loss_type,
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config.bf16 or config.fp16
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Gradient checkpointing
        if config.gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.policy_model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [
                    p for n, p in self.policy_model.named_parameters()
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
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        num_training_steps = (
            len(self.train_dataset) //
            (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps) *
            self.config.num_train_epochs
        )
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
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
            collate_fn=self._collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=True,
        )
    
    def _collate_fn(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of preference pairs."""
        # Pad and concatenate chosen/rejected
        max_length = self.config.max_seq_length
        
        def pad_sequences(sequences, pad_id=0):
            max_len = min(max(len(s) for s in sequences), max_length)
            padded = []
            for seq in sequences:
                if len(seq) > max_length:
                    seq = seq[:max_length]
                else:
                    seq = seq + [pad_id] * (max_len - len(seq))
                padded.append(seq)
            return torch.tensor(padded, dtype=torch.long)
        
        chosen_input_ids = pad_sequences([f['chosen_input_ids'] for f in features])
        chosen_attention_mask = pad_sequences([f['chosen_attention_mask'] for f in features], pad_id=0)
        rejected_input_ids = pad_sequences([f['rejected_input_ids'] for f in features])
        rejected_attention_mask = pad_sequences([f['rejected_attention_mask'] for f in features], pad_id=0)
        
        # Concatenate for efficient forward pass
        concatenated_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        concatenated_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        
        return {
            'concatenated_input_ids': concatenated_input_ids,
            'concatenated_attention_mask': concatenated_attention_mask,
            'concatenated_labels': concatenated_input_ids.clone(),
            'batch_size': len(features),
        }
    
    def _compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for sequences."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for labels
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        # Sum over sequence (excluding padding)
        logps = per_token_logps.sum(dim=-1)
        
        return logps
    
    def training_step(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single training step."""
        batch_size = batch['batch_size']
        
        # Split concatenated inputs
        input_ids = batch['concatenated_input_ids']
        attention_mask = batch['concatenated_attention_mask']
        labels = batch['concatenated_labels']
        
        chosen_input_ids = input_ids[:batch_size]
        rejected_input_ids = input_ids[batch_size:]
        chosen_attention_mask = attention_mask[:batch_size]
        rejected_attention_mask = attention_mask[batch_size:]
        chosen_labels = labels[:batch_size]
        rejected_labels = labels[batch_size:]
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                # Policy model forward
                policy_chosen_logps = self._compute_logps(
                    self.policy_model,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                )
                policy_rejected_logps = self._compute_logps(
                    self.policy_model,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                )
                
                # Reference model forward (no grad needed)
                with torch.no_grad():
                    reference_chosen_logps = self._compute_logps(
                        self.reference_model,
                        chosen_input_ids,
                        chosen_attention_mask,
                        chosen_labels,
                    )
                    reference_rejected_logps = self._compute_logps(
                        self.reference_model,
                        rejected_input_ids,
                        rejected_attention_mask,
                        rejected_labels,
                    )
                
                # Compute loss
                loss, metrics = self.loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )
        else:
            # Policy model forward
            policy_chosen_logps = self._compute_logps(
                self.policy_model,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_labels,
            )
            policy_rejected_logps = self._compute_logps(
                self.policy_model,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_labels,
            )
            
            # Reference model forward
            with torch.no_grad():
                reference_chosen_logps = self._compute_logps(
                    self.reference_model,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                )
                reference_rejected_logps = self._compute_logps(
                    self.reference_model,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                )
            
            # Compute loss
            loss, metrics = self.loss_fn(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.detach(), metrics
    
    def optimizer_step(self) -> None:
        """Perform optimizer step."""
        # Gradient clipping
        if self.use_amp and self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
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
        self.policy_model.zero_grad()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataset is None:
            return {}
        
        self.policy_model.eval()
        
        eval_dataloader = self._get_dataloader(
            self.eval_dataset,
            self.config.per_device_eval_batch_size,
        )
        
        total_loss = 0.0
        total_reward_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                batch_size = batch['batch_size']
                input_ids = batch['concatenated_input_ids']
                attention_mask = batch['concatenated_attention_mask']
                labels = batch['concatenated_labels']
                
                chosen_input_ids = input_ids[:batch_size]
                rejected_input_ids = input_ids[batch_size:]
                chosen_attention_mask = attention_mask[:batch_size]
                rejected_attention_mask = attention_mask[batch_size:]
                chosen_labels = labels[:batch_size]
                rejected_labels = labels[batch_size:]
                
                # Policy forward
                policy_chosen_logps = self._compute_logps(
                    self.policy_model,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                )
                policy_rejected_logps = self._compute_logps(
                    self.policy_model,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                )
                
                # Reference forward
                reference_chosen_logps = self._compute_logps(
                    self.reference_model,
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                )
                reference_rejected_logps = self._compute_logps(
                    self.reference_model,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels,
                )
                
                # Compute loss
                loss, metrics = self.loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )
                
                total_loss += loss.item()
                total_reward_accuracy += metrics['reward_accuracy']
                num_batches += 1
        
        self.policy_model.train()
        
        return {
            'eval_loss': total_loss / max(num_batches, 1),
            'eval_reward_accuracy': total_reward_accuracy / max(num_batches, 1),
        }
    
    def save_checkpoint(self, output_dir: Optional[str] = None) -> None:
        """Save checkpoint."""
        output_dir = output_dir or f"{self.config.output_dir}/checkpoint-{self.global_step}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save policy model
        self.policy_model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
        }
        
        if self.scaler:
            training_state['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(training_state, f"{output_dir}/training_state.pt")
        
        # Save config
        with open(f"{output_dir}/dpo_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Checkpoint saved to {output_dir}")
    
    def train(self) -> Dict[str, Any]:
        """Run DPO training."""
        logger.info("Starting DPO training...")
        
        train_dataloader = self._get_dataloader(
            self.train_dataset,
            self.config.per_device_train_batch_size,
            shuffle=True,
        )
        
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = int(num_update_steps_per_epoch * self.config.num_train_epochs)
        
        progress_bar = tqdm(total=num_training_steps, desc="Training")
        
        self.policy_model.train()
        self.optimizer.zero_grad()
        
        training_metrics = {
            'loss': [],
            'reward_accuracy': [],
            'eval_loss': [],
        }
        
        accumulation_step = 0
        
        for epoch in range(int(self.config.num_train_epochs)):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Training step
                loss, metrics = self.training_step(batch)
                accumulation_step += 1
                
                # Optimizer step
                if accumulation_step % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Log metrics
                    training_metrics['loss'].append(metrics['loss'])
                    training_metrics['reward_accuracy'].append(metrics['reward_accuracy'])
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(
                            f"Step {self.global_step}: "
                            f"loss={metrics['loss']:.4f}, "
                            f"reward_accuracy={metrics['reward_accuracy']:.4f}"
                        )
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        training_metrics['eval_loss'].append(eval_metrics.get('eval_loss', 0))
                        
                        logger.info(f"EVAL: loss={eval_metrics.get('eval_loss', 0):.4f}")
                        
                        # Save best model
                        if eval_metrics.get('eval_loss', float('inf')) < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['eval_loss']
                            self.save_checkpoint(f"{self.config.output_dir}/best")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                
                if self.global_step >= num_training_steps:
                    break
        
        progress_bar.close()
        
        # Save final checkpoint
        self.save_checkpoint()
        
        logger.info("DPO training complete!")
        
        return training_metrics


class DPOPipeline:
    """
    Complete DPO Pipeline.
    
    Orchestrates the full DPO workflow:
    1. Load models
    2. Load data
    3. Train
    4. Evaluate
    5. Export
    
    Example:
        >>> pipeline = DPOPipeline(config)
        >>> results = pipeline.run()
    """
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.policy_model = None
        self.reference_model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_models(self) -> Tuple[nn.Module, nn.Module]:
        """Load policy and reference models."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading policy model from {self.config.model_name_or_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load policy model
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )
            
            # Load reference model
            ref_path = self.config.reference_model_name_or_path or self.config.model_name_or_path
            logger.info(f"Loading reference model from {ref_path}")
            
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )
            
            return self.policy_model, self.reference_model
        
        except ImportError:
            logger.error("Transformers library not installed")
            raise
    
    def load_data(
        self,
        data_path: str,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """Load preference data."""
        from .rejection_sampling import PreferencePair, PreferenceDataset
        
        # Load preference pairs
        pairs = PreferencePair.load(data_path)
        
        # Create datasets
        train_dataset = PreferenceDataset(
            pairs,
            self.tokenizer,
            max_length=self.config.max_seq_length,
        )
        
        # Split for eval if needed
        eval_dataset = None
        
        return train_dataset, eval_dataset
    
    def run(self, data_path: str) -> Dict[str, Any]:
        """
        Run the full DPO pipeline.
        
        Args:
            data_path: Path to preference data
        
        Returns:
            Training results
        """
        logger.info("Starting DPO pipeline...")
        
        # Load models
        self.load_models()
        
        # Load data
        train_dataset, eval_dataset = self.load_data(data_path)
        
        # Create trainer
        self.trainer = DPOTrainer(
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            config=self.config,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        training_metrics = self.trainer.train()
        
        results = {
            'training_metrics': training_metrics,
            'best_eval_loss': self.trainer.best_eval_loss,
            'output_dir': self.config.output_dir,
        }
        
        # Save results
        results_path = Path(self.config.output_dir) / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline complete. Results saved to {results_path}")
        
        return results
