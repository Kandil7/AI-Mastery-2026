"""
RLHF with PPO - Module 2.5.3

Production-ready RLHF implementation using PPO:
- PPO configuration
- PPO trainer
- Reward model
- Value model
- RLHF pipeline

References:
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
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
class PPOConfig:
    """Configuration for PPO training."""
    # Model settings
    model_name_or_path: str = ""
    reward_model_path: Optional[str] = None
    value_model_path: Optional[str] = None
    
    # PPO settings
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    batch_size: int = 16
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.1
    ent_coef: float = 0.01  # Entropy coefficient
    
    # Training settings
    output_dir: str = "./ppo_output"
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 1024
    max_new_tokens: int = 256
    
    # Optimization settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 0.5
    bf16: bool = True
    
    # Logging settings
    logging_steps: int = 10
    save_steps: int = 500
    
    # Other settings
    seed: int = 42


class RewardModel(nn.Module):
    """
    Reward Model for RLHF.
    
    Scores text sequences based on quality/preferences.
    
    Args:
        base_model: Base transformer model
        hidden_size: Hidden size for reward head
        dropout: Dropout probability
        
    Example:
        >>> reward_model = RewardModel(base_model)
        >>> rewards = reward_model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.base_model = base_model
        
        # Get hidden size from base model
        if hidden_size is None:
            if hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = 768
        
        # Reward head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_rewards: bool = False,
    ) -> torch.Tensor:
        """
        Compute reward scores.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_all_rewards: Return per-token rewards
        
        Returns:
            Reward scores
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Get last non-padded token for each sequence
        if return_all_rewards:
            # Per-token rewards
            rewards = self.value_head(hidden_states).squeeze(-1)
        else:
            # Single reward per sequence (from last token)
            batch_size = input_ids.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            
            # Gather last token hidden states
            last_hidden = hidden_states[
                torch.arange(batch_size, device=input_ids.device),
                sequence_lengths
            ]
            
            rewards = self.value_head(last_hidden).squeeze(-1)
        
        return rewards


class ValueModel(nn.Module):
    """
    Value Model for PPO.
    
    Estimates the value of states for advantage computation.
    
    Args:
        base_model: Base transformer model
        hidden_size: Hidden size for value head
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        
        self.base_model = base_model
        
        if hidden_size is None:
            if hasattr(base_model.config, 'hidden_size'):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = 768
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.value_head.modules():
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
        Compute value estimates.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
        
        Returns:
            Value estimates
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.hidden_states[-1]
        
        # Per-token values
        values = self.value_head(hidden_states).squeeze(-1)
        
        return values


class PPOTrainer:
    """
    PPO Trainer for RLHF.
    
    Implements Proximal Policy Optimization for fine-tuning
    language models with reward feedback.
    
    Args:
        policy_model: Policy model (actor)
        value_model: Value model (critic)
        reward_model: Reward model
        config: PPO configuration
        tokenizer: Tokenizer
        
    Example:
        >>> trainer = PPOTrainer(policy, value, reward, config, tokenizer)
        >>> trainer.train(prompts)
    """
    
    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        reward_model: nn.Module,
        config: PPOConfig,
        tokenizer: Any,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.config = config
        self.tokenizer = tokenizer
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.policy_model.to(self.device)
        self.value_model.to(self.device)
        self.reward_model.to(self.device)
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        
        # Optimizers
        self.policy_optimizer = self._create_optimizer(self.policy_model)
        self.value_optimizer = self._create_optimizer(self.value_model)
        
        # Training state
        self.global_step = 0
    
    def _create_optimizer(
        self,
        model: nn.Module,
    ) -> torch.optim.Optimizer:
        """Create optimizer."""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
        )
    
    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate responses for prompts.
        
        Args:
            prompts: List of prompts
        
        Returns:
            Tuple of (texts, token_ids, log_probs)
        """
        self.policy_model.eval()
        
        texts = []
        all_token_ids = []
        all_log_probs = []
        
        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                add_special_tokens=False,
            ).to(self.device)
            
            # Generate
            outputs = self.policy_model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
            # Extract generated tokens
            prompt_len = inputs.shape[-1]
            generated_ids = outputs.sequences[0, prompt_len:]
            
            # Compute log probs
            log_probs = []
            for score in outputs.scores:
                log_prob = F.log_softmax(score[0], dim=-1)
                token_id = generated_ids[len(log_probs)]
                log_probs.append(log_prob[token_id].item())
            
            # Decode
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            texts.append(text)
            all_token_ids.append(generated_ids)
            all_log_probs.append(torch.tensor(log_probs))
        
        return texts, all_token_ids, all_log_probs
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE.
        
        Args:
            rewards: Rewards
            values: Value estimates
            masks: Padding masks
        
        Returns:
            Tuple of (advantages, returns)
        """
        gamma = self.config.gamma
        lam = self.config.lam
        
        # Compute advantages using Generalized Advantage Estimation
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            gae = delta + gamma * lam * masks[t] * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=rewards.device)
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def ppo_update(
        self,
        query_input_ids: torch.Tensor,
        response_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            query_input_ids: Prompt token IDs
            response_input_ids: Response token IDs
            attention_mask: Attention mask
            old_log_probs: Old log probabilities
            rewards: Rewards
        
        Returns:
            Metrics dictionary
        """
        # Concatenate query and response
        input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
        
        # Get sequence lengths
        query_lengths = query_input_ids.shape[-1]
        
        # PPO epochs
        metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        for _ in range(self.config.ppo_epochs):
            # Forward pass through policy
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            logits = outputs.logits
            
            # Get logits for response tokens only
            response_logits = logits[:, query_lengths - 1:-1, :]
            
            # Compute new log probs
            new_log_probs = F.log_softmax(response_logits, dim=-1)
            
            # Gather log probs for generated tokens
            response_len = response_input_ids.shape[-1]
            generated_tokens = response_input_ids
            
            # Compute per-token log probs
            new_log_probs_gathered = torch.gather(
                new_log_probs,
                dim=-1,
                index=generated_tokens.unsqueeze(-1),
            ).squeeze(-1)
            
            # Compute importance weights
            ratio = torch.exp(new_log_probs_gathered - old_log_probs)
            
            # Compute advantages
            values = self.value_model(input_ids, attention_mask)
            response_values = values[:, query_lengths - 1:-1]
            
            # Simplified advantage computation
            advantages = rewards.unsqueeze(1) - response_values.mean(dim=1, keepdim=True)
            advantages = advantages.detach()
            
            # Policy loss (clipped surrogate objective)
            clip_range = self.config.clip_range
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy = -(new_log_probs_gathered.exp() * new_log_probs_gathered).sum(dim=-1).mean()
            
            # Total policy loss
            total_policy_loss = policy_loss - self.config.ent_coef * entropy
            
            # Value loss
            returns = rewards.unsqueeze(1).expand_as(response_values)
            value_loss = F.mse_loss(response_values, returns)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm,
            )
            self.policy_optimizer.step()
            
            # Update value model
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_model.parameters(),
                self.config.max_grad_norm,
            )
            self.value_optimizer.step()
            
            # Accumulate metrics
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
            metrics['entropy'] += entropy.item()
        
        # Average metrics
        for key in metrics:
            metrics[key] /= self.config.ppo_epochs
        
        return metrics
    
    def train(
        self,
        prompts: List[str],
    ) -> Dict[str, Any]:
        """
        Run PPO training.
        
        Args:
            prompts: List of prompts
        
        Returns:
            Training metrics
        """
        logger.info(f"Starting PPO training with {len(prompts)} prompts...")
        
        all_metrics = {
            'rewards': [],
            'policy_loss': [],
            'value_loss': [],
        }
        
        # Process in batches
        batch_size = self.config.batch_size
        
        for epoch in range(int(self.config.num_train_epochs)):
            for i in tqdm(range(0, len(prompts), batch_size), desc=f"Epoch {epoch}"):
                batch_prompts = prompts[i:i + batch_size]
                
                # Generate responses
                responses, response_ids, old_log_probs = self.generate_responses(batch_prompts)
                
                # Tokenize full sequences
                full_texts = [p + r for p, r in zip(batch_prompts, responses)]
                
                encoded = self.tokenizer(
                    full_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                ).to(self.device)
                
                # Compute rewards
                rewards = self.reward_model(
                    encoded.input_ids,
                    encoded.attention_mask,
                )
                
                # Tokenize queries
                query_encoded = self.tokenizer(
                    batch_prompts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                ).to(self.device)
                
                # Pad response IDs to same length
                max_response_len = max(len(rid) for rid in response_ids)
                padded_response_ids = []
                for rid in response_ids:
                    padded = torch.cat([
                        rid,
                        torch.zeros(max_response_len - len(rid), dtype=torch.long, device=self.device),
                    ])
                    padded_response_ids.append(padded)
                response_ids_tensor = torch.stack(padded_response_ids)
                
                # Pad old log probs
                max_log_prob_len = max(len(olp) for olp in old_log_probs)
                padded_log_probs = []
                for olp in old_log_probs:
                    padded = torch.cat([
                        olp,
                        torch.zeros(max_log_prob_len - len(olp), device=self.device),
                    ])
                    padded_log_probs.append(padded)
                old_log_probs_tensor = torch.stack(padded_log_probs)
                
                # PPO update
                metrics = self.ppo_update(
                    query_encoded.input_ids,
                    response_ids_tensor,
                    encoded.attention_mask,
                    old_log_probs_tensor,
                    rewards,
                )
                
                # Record metrics
                all_metrics['rewards'].append(rewards.mean().item())
                all_metrics['policy_loss'].append(metrics['policy_loss'])
                all_metrics['value_loss'].append(metrics['value_loss'])
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"reward={rewards.mean().item():.4f}, "
                        f"policy_loss={metrics['policy_loss']:.4f}"
                    )
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
        
        logger.info("PPO training complete!")
        
        return all_metrics
    
    def save_checkpoint(self, output_dir: Optional[str] = None) -> None:
        """Save checkpoint."""
        output_dir = output_dir or f"{self.config.output_dir}/checkpoint-{self.global_step}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.policy_model.save_pretrained(f"{output_dir}/policy")
        self.value_model.save_pretrained(f"{output_dir}/value")
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }
        
        torch.save(training_state, f"{output_dir}/training_state.pt")
        
        logger.info(f"Checkpoint saved to {output_dir}")


class RLHFPipeline:
    """
    Complete RLHF Pipeline.
    
    Orchestrates the full RLHF workflow:
    1. Load models
    2. Load reward model
    3. PPO training
    4. Export
    
    Example:
        >>> pipeline = RLHFPipeline(config)
        >>> results = pipeline.run(prompts)
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.policy_model = None
        self.value_model = None
        self.reward_model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_models(self) -> None:
        """Load all models."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading policy model from {self.config.model_name_or_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model for policy
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )
            
            # Policy model
            self.policy_model = base_model
            
            # Value model
            self.value_model = ValueModel(base_model)
            
            # Reward model
            if self.config.reward_model_path:
                reward_base = AutoModelForCausalLM.from_pretrained(
                    self.config.reward_model_path,
                    trust_remote_code=True,
                )
                self.reward_model = RewardModel(reward_base)
            else:
                self.reward_model = RewardModel(base_model)
            
            logger.info("Models loaded successfully")
        
        except ImportError:
            logger.error("Transformers library not installed")
            raise
    
    def run(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Run the full RLHF pipeline.
        
        Args:
            prompts: List of prompts for training
        
        Returns:
            Training results
        """
        logger.info("Starting RLHF pipeline...")
        
        # Load models
        self.load_models()
        
        # Create trainer
        self.trainer = PPOTrainer(
            policy_model=self.policy_model,
            value_model=self.value_model,
            reward_model=self.reward_model,
            config=self.config,
            tokenizer=self.tokenizer,
        )
        
        # Train
        training_metrics = self.trainer.train(prompts)
        
        results = {
            'training_metrics': training_metrics,
            'output_dir': self.config.output_dir,
        }
        
        # Save results
        results_path = Path(self.config.output_dir) / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline complete. Results saved to {results_path}")
        
        return results
