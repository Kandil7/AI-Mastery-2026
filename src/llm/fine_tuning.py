"""
Fine-Tuning Module

This module implements various fine-tuning techniques for LLMs,
including LoRA (Low-Rank Adaptation), P-Tuning, and full fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math
import copy
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuningMethod(Enum):
    """Enumeration of fine-tuning methods."""
    FULL_FINE_TUNING = "full_fine_tuning"
    LORA = "lora"
    P_TUNING = "p_tuning"
    PROMPT_TUNING = "prompt_tuning"
    ADAPTER = "adapter"


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    method: FineTuningMethod
    model_name: str
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    lora_rank: int = 8  # For LoRA
    lora_alpha: int = 32  # For LoRA
    lora_dropout: float = 0.1  # For LoRA
    adapter_reduction_factor: int = 16  # For adapters
    p_tuning_num_virtual_tokens: int = 20  # For P-Tuning
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer.
    This layer adds a low-rank decomposition to the original weight matrix.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Initialize A and B matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * math.sqrt(1 / rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Mark the layer as LoRA
        self.lora_enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Standard linear transformation
        standard_output = F.linear(x, self.weight, self.bias)
        
        # LoRA transformation: x @ A.T @ B.T
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return standard_output + lora_output


def apply_lora_to_model(model: nn.Module, config: FineTuningConfig) -> nn.Module:
    """
    Apply LoRA to a model by replacing linear layers with LoRA layers.
    
    Args:
        model: The model to apply LoRA to
        config: Fine-tuning configuration
        
    Returns:
        Model with LoRA applied
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create a new LoRA layer with the same dimensions
            lora_layer = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
            
            # Copy the original weights to the new layer
            lora_layer.weight = module.weight
            lora_layer.bias = module.bias
            
            # Replace the original layer with the LoRA layer
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, lora_layer)
    
    return model


class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning.
    """
    
    def __init__(self, input_dim: int, reduction_factor: int = 16, non_linearity: str = 'relu'):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = input_dim // reduction_factor
        
        # Down-projector
        self.down_projector = nn.Linear(input_dim, self.bottleneck_dim)
        
        # Non-linearity
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'gelu':
            self.non_linearity = nn.GELU()
        else:
            raise ValueError(f"Unsupported non-linearity: {non_linearity}")
        
        # Up-projector
        self.up_projector = nn.Linear(self.bottleneck_dim, input_dim)
        
        # Initialize weights
        nn.init.zeros_(self.up_projector.weight)
        nn.init.zeros_(self.up_projector.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter."""
        # Residual connection
        residual = x
        
        # Down-project
        x = self.down_projector(x)
        
        # Apply non-linearity
        x = self.non_linearity(x)
        
        # Up-project
        x = self.up_projector(x)
        
        # Add residual
        return x + residual


def apply_adapters_to_model(model: nn.Module, config: FineTuningConfig) -> nn.Module:
    """
    Apply adapter layers to a model.
    
    Args:
        model: The model to apply adapters to
        config: Fine-tuning configuration
        
    Returns:
        Model with adapters applied
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'output' in name:  # Only add adapters after feed-forward layers
            # Create an adapter layer
            adapter = AdapterLayer(
                input_dim=module.out_features,
                reduction_factor=config.adapter_reduction_factor
            )
            
            # Create a sequential with the original layer and the adapter
            new_module = nn.Sequential(module, adapter)
            
            # Replace the original module
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent_name]
            setattr(parent_module, child_name, new_module)
    
    return model


class PVector(nn.Module):
    """
    P-Vector for P-Tuning - learnable virtual tokens.
    """
    
    def __init__(self, num_tokens: int, model_dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        
        # Learnable virtual tokens
        self.p_tokens = nn.Parameter(torch.randn(num_tokens, model_dim))
        
        # Initialize the tokens
        nn.init.normal_(self.p_tokens, mean=0.0, std=0.02)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """Forward pass to get virtual tokens for a batch."""
        # Expand the tokens to match the batch size
        expanded_tokens = self.p_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return expanded_tokens


class P_TUNING_MODEL(nn.Module):
    """
    Model wrapper for P-Tuning.
    """
    
    def __init__(self, base_model: nn.Module, config: FineTuningConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Get the model's embedding dimension
        if hasattr(base_model, 'config'):
            model_dim = base_model.config.hidden_size
        else:
            # Fallback: assume a common dimension
            model_dim = 768
        
        # Create P-Vectors
        self.p_vectors = PVector(config.p_tuning_num_virtual_tokens, model_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Forward pass with P-Tuning."""
        batch_size = input_ids.size(0)
        
        # Get virtual tokens
        virtual_tokens = self.p_vectors(batch_size)
        
        # Get embeddings for real tokens
        real_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Concatenate virtual and real tokens
        embeddings = torch.cat([virtual_tokens, real_embeddings], dim=1)
        
        # Update attention mask to account for virtual tokens
        if attention_mask is not None:
            virtual_mask = torch.ones(batch_size, self.config.p_tuning_num_virtual_tokens, 
                                    dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([virtual_mask, attention_mask], dim=1)
        
        # Forward pass through the base model with concatenated embeddings
        return self.base_model(inputs_embeds=embeddings, attention_mask=attention_mask, **kwargs)


class FineTuner:
    """
    Main class for fine-tuning LLMs with various methods.
    """
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        
        # Load the base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply the selected fine-tuning method
        if config.method == FineTuningMethod.LORA:
            self.model = apply_lora_to_model(self.model, config)
        elif config.method == FineTuningMethod.ADAPTER:
            self.model = apply_adapters_to_model(self.model, config)
        elif config.method == FineTuningMethod.P_TUNING:
            self.model = P_TUNING_MODEL(self.model, config)
        elif config.method == FineTuningMethod.FULL_FINE_TUNING:
            # No modification needed for full fine-tuning
            pass
        else:
            raise ValueError(f"Unsupported fine-tuning method: {config.method}")
        
        # Move model to device
        self.model.to(config.device)
    
    def prepare_dataset(self, texts: List[str], labels: Optional[List[Any]] = None) -> Dataset:
        """
        Prepare a dataset for fine-tuning.
        
        Args:
            texts: List of input texts
            labels: Optional list of labels (for supervised tasks)
            
        Returns:
            Dataset ready for training
        """
        class TextDataset(Dataset):
            def __init__(self, texts: List[str], labels: Optional[List[Any]], tokenizer, max_length: int):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
                
                # Tokenize all texts
                self.encoded = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encoded.items()}
                if self.labels is not None:
                    item['labels'] = torch.tensor(self.labels[idx]) if isinstance(self.labels[idx], int) else self.labels[idx]
                return item
        
        return TextDataset(texts, labels, self.tokenizer, self.config.max_length)
    
    def train(
        self, 
        train_dataset: Dataset, 
        eval_dataset: Optional[Dataset] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Train the model with the specified fine-tuning method.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            save_path: Optional path to save the fine-tuned model
            
        Returns:
            Training metrics
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Set up optimizer
        # For LoRA and adapters, only optimize the new parameters
        if self.config.method in [FineTuningMethod.LORA, FineTuningMethod.ADAPTER]:
            # Only optimize LoRA or adapter parameters
            params_to_optimize = [p for n, p in self.model.named_parameters() if 'lora' in n or 'adapter' in n or 'p_tokens' in n]
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            # Optimize all parameters for full fine-tuning
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.config.num_epochs
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        step_count = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Extract loss (assuming the model returns it)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    # For models that don't return loss directly, we need to compute it
                    # This is a simplified example - actual implementation depends on the task
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    labels = batch.get('labels')
                    
                    if labels is not None:
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        # For language modeling, use the standard loss calculation
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = batch['input_ids'][..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                step_count += 1
                
                # Update weights
                if (step_count + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Average Loss: {epoch_loss / len(train_loader):.4f}")
        
        # Save the model if path is provided
        if save_path:
            self.save_model(save_path)
        
        return {
            "average_loss": total_loss / (len(train_loader) * self.config.num_epochs),
            "total_steps": step_count
        }
    
    def save_model(self, path: str):
        """
        Save the fine-tuned model.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save the model
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save the config
        import json
        with open(f"{path}/config.json", 'w') as f:
            json.dump(self.config.__dict__, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a fine-tuned model.
        
        Args:
            path: Path to load the model from
        """
        # Load the model state dict
        state_dict = torch.load(f"{path}/model.pt", map_location=self.config.device)
        self.model.load_state_dict(state_dict)
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        logger.info(f"Model loaded from {path}")
    
    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output to return only the generated part
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        return generated_text.strip()


class QLoRA(nn.Module):
    """
    Quantized Low-Rank Adaptation (QLoRA) implementation.
    This is a simplified version focusing on the core concepts.
    """
    
    def __init__(self, base_model: nn.Module, config: FineTuningConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # For QLoRA, we would typically:
        # 1. Quantize the base model weights
        # 2. Apply LoRA to the quantized model
        # 3. Only train the LoRA parameters
        
        # Apply LoRA to the base model
        self.lora_model = apply_lora_to_model(copy.deepcopy(base_model), config)
        
        # Freeze the base model parameters
        for param in self.lora_model.parameters():
            param.requires_grad = False
        
        # Only LoRA parameters should be trainable
        for name, param in self.lora_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        return self.lora_model(*args, **kwargs)


def create_fine_tuner(config: FineTuningConfig) -> FineTuner:
    """
    Factory function to create a fine-tuner.
    
    Args:
        config: Fine-tuning configuration
        
    Returns:
        FineTuner instance
    """
    return FineTuner(config)


def evaluate_model(fine_tuner: FineTuner, eval_dataset: Dataset, batch_size: int = 8) -> Dict[str, float]:
    """
    Evaluate a fine-tuned model.
    
    Args:
        fine_tuner: FineTuner instance
        eval_dataset: Evaluation dataset
        batch_size: Batch size for evaluation
        
    Returns:
        Evaluation metrics
    """
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    fine_tuner.model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            batch = {k: v.to(fine_tuner.config.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = fine_tuner.model(**batch)
            
            # Calculate loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # Calculate loss manually if not provided
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                labels = batch.get('labels')
                
                if labels is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    # For language modeling
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch['input_ids'][..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
            
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)
    
    return {
        "eval_loss": total_loss / total_samples,
        "total_samples": total_samples
    }


# Example usage
if __name__ == "__main__":
    # Example configuration for LoRA fine-tuning
    config = FineTuningConfig(
        method=FineTuningMethod.LORA,
        model_name="bert-base-uncased",
        learning_rate=1e-4,
        batch_size=4,
        num_epochs=3,
        max_length=128,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    # Create fine-tuner
    fine_tuner = create_fine_tuner(config)
    
    # Example texts for training (in practice, you'd load your actual dataset)
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science."
    ]
    
    # Prepare dataset
    train_dataset = fine_tuner.prepare_dataset(train_texts)
    
    # Train the model
    metrics = fine_tuner.train(train_dataset)
    print(f"Training metrics: {metrics}")
    
    # Example generation
    generated = fine_tuner.generate("Machine learning is")
    print(f"Generated text: {generated}")