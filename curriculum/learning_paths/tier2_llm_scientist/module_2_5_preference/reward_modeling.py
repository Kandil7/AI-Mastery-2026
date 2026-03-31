"""
Reward Modeling - Module 2.5.4

Production-ready reward modeling implementation:
- Reward model configuration
- Bradley-Terry model
- Pairwise dataset
- Reward model trainer

References:
- "Learning to Rank from Pairwise Preferences" (Joachims, 2002)
- "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)
"""

import json
import logging
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
class RewardModelConfig:
    """Configuration for reward model training."""
    # Model settings
    model_name_or_path: str = ""
    hidden_size: Optional[int] = None
    
    # Training settings
    output_dir: str = "./reward_model_output"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    # Optimization settings
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 1.0
    max_seq_length: int = 1024
    
    # Logging settings
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Other settings
    seed: int = 42


class BradleyTerryModel(nn.Module):
    """
    Bradley-Terry Model for pairwise preferences.
    
    Models the probability that item i is preferred over item j:
    P(i > j) = exp(r_i) / (exp(r_i) + exp(r_j)) = sigmoid(r_i - r_j)
    
    Args:
        base_model: Base transformer model
        hidden_size: Hidden size for reward head
        dropout: Dropout probability
        
    Example:
        >>> model = BradleyTerryModel(base_model)
        >>> rewards = model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.base_model = base_model
        
        # Get hidden size
        if hidden_size is None:
            if hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = 768
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward scores.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Reward scores (one per sequence)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Get last non-padded token for each sequence
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        # Gather last token hidden states
        last_hidden = hidden_states[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths
        ]
        
        # Compute reward
        rewards = self.reward_head(last_hidden).squeeze(-1)
        
        return rewards
    
    def compute_pairwise_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Bradley-Terry pairwise loss.
        
        Args:
            chosen_input_ids: Chosen response token IDs
            chosen_attention_mask: Chosen attention mask
            rejected_input_ids: Rejected response token IDs
            rejected_attention_mask: Rejected attention mask
        
        Returns:
            Tuple of (loss, metrics)
        """
        # Compute rewards for chosen and rejected
        chosen_rewards = self.forward(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.forward(rejected_input_ids, rejected_attention_mask)
        
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        logits = chosen_rewards - rejected_rewards
        loss = F.softplus(-logits).mean()
        
        # Metrics
        metrics = {
            'loss': loss.item(),
            'chosen_rewards_mean': chosen_rewards.mean().item(),
            'rejected_rewards_mean': rejected_rewards.mean().item(),
            'reward_diff_mean': (chosen_rewards - rejected_rewards).mean().item(),
            'accuracy': (chosen_rewards > rejected_rewards).float().mean().item(),
        }
        
        return loss, metrics


class PairwiseDataset(Dataset):
    """
    Dataset for pairwise preference data.
    
    Args:
        pairs: List of preference pairs
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        
    Example:
        >>> dataset = PairwiseDataset(pairs, tokenizer)
    """
    
    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 1024,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self._preprocess()
    
    def _preprocess(self) -> None:
        """Preprocess pairs for training."""
        self.processed = []
        
        for pair in self.pairs:
            # Get prompt and responses
            prompt = pair.get('prompt', '')
            chosen = pair.get('chosen', '')
            rejected = pair.get('rejected', '')
            
            # Format as prompt + response
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            # Tokenize chosen
            chosen_encoded = self.tokenizer(
                chosen_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            
            # Tokenize rejected
            rejected_encoded = self.tokenizer(
                rejected_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            
            self.processed.append({
                'chosen_input_ids': chosen_encoded['input_ids'],
                'chosen_attention_mask': chosen_encoded['attention_mask'],
                'rejected_input_ids': rejected_encoded['input_ids'],
                'rejected_attention_mask': rejected_encoded['attention_mask'],
            })
    
    def __len__(self) -> int:
        return len(self.processed)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed[idx]
        return {
            'chosen_input_ids': torch.tensor(item['chosen_input_ids'], dtype=torch.long),
            'chosen_attention_mask': torch.tensor(item['chosen_attention_mask'], dtype=torch.long),
            'rejected_input_ids': torch.tensor(item['rejected_input_ids'], dtype=torch.long),
            'rejected_attention_mask': torch.tensor(item['rejected_attention_mask'], dtype=torch.long),
        }


class RewardModelTrainer:
    """
    Trainer for Reward Model.
    
    Trains a reward model using pairwise preference data
    and the Bradley-Terry loss.
    
    Args:
        model: Bradley-Terry model
        config: Reward model configuration
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        
    Example:
        >>> trainer = RewardModelTrainer(model, config, tokenizer, dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: BradleyTerryModel,
        config: RewardModelConfig,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config.bf16 or config.fp16
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        # Training state
        self.global_step = 0
        self.best_eval_accuracy = 0.0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
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
            num_workers=4,
            pin_memory=True,
        )
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform a single training step."""
        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                loss, metrics = self.model.compute_pairwise_loss(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                )
        else:
            loss, metrics = self.model.compute_pairwise_loss(
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
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
        
        total_accuracy = 0.0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                _, metrics = self.model.compute_pairwise_loss(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask'],
                )
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                num_batches += 1
        
        self.model.train()
        
        return {
            'eval_loss': total_loss / max(num_batches, 1),
            'eval_accuracy': total_accuracy / max(num_batches, 1),
        }
    
    def save_checkpoint(self, output_dir: Optional[str] = None) -> None:
        """Save checkpoint."""
        output_dir = output_dir or f"{self.config.output_dir}/checkpoint-{self.global_step}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_accuracy': self.best_eval_accuracy,
        }
        
        if self.scaler:
            training_state['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(training_state, f"{output_dir}/training_state.pt")
        
        # Save config
        with open(f"{output_dir}/reward_model_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Checkpoint saved to {output_dir}")
    
    def train(self) -> Dict[str, Any]:
        """Run reward model training."""
        logger.info("Starting reward model training...")
        
        train_dataloader = self._get_dataloader(
            self.train_dataset,
            self.config.per_device_train_batch_size,
            shuffle=True,
        )
        
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = int(num_update_steps_per_epoch * self.config.num_train_epochs)
        
        progress_bar = tqdm(total=num_training_steps, desc="Training")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        training_metrics = {
            'loss': [],
            'accuracy': [],
            'eval_accuracy': [],
        }
        
        accumulation_step = 0
        
        for epoch in range(int(self.config.num_train_epochs)):
            for batch in train_dataloader:
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
                    training_metrics['accuracy'].append(metrics['accuracy'])
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(
                            f"Step {self.global_step}: "
                            f"loss={metrics['loss']:.4f}, "
                            f"accuracy={metrics['accuracy']:.4f}"
                        )
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        training_metrics['eval_accuracy'].append(eval_metrics.get('eval_accuracy', 0))
                        
                        logger.info(f"EVAL: accuracy={eval_metrics.get('eval_accuracy', 0):.4f}")
                        
                        # Save best model
                        if eval_metrics.get('eval_accuracy', 0) > self.best_eval_accuracy:
                            self.best_eval_accuracy = eval_metrics['eval_accuracy']
                            self.save_checkpoint(f"{self.config.output_dir}/best")
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
                
                if self.global_step >= num_training_steps:
                    break
        
        progress_bar.close()
        
        # Save final checkpoint
        self.save_checkpoint()
        
        logger.info("Reward model training complete!")
        
        return training_metrics
