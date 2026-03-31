"""
Model Merging - Module 2.8.1

Model merging techniques: SLERP, DARE, TIES, Task Arithmetic, Model Soups.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    """Configuration for model merging."""
    method: str = "linear"
    weights: Optional[List[float]] = None
    density: float = 1.0
    prune_method: str = "magnitude"


class ModelMerger(ABC):
    """Abstract base class for model mergers."""
    
    @abstractmethod
    def merge(
        self,
        models: List[nn.Module],
        config: MergeConfig,
    ) -> nn.Module:
        """Merge multiple models."""
        pass


class LinearMerger(ModelMerger):
    """Linear interpolation of model weights."""
    
    def merge(
        self,
        models: List[nn.Module],
        config: MergeConfig,
    ) -> nn.Module:
        """Merge models using linear interpolation."""
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        # Get state dicts
        state_dicts = [m.state_dict() for m in models]
        
        # Merge
        merged_state = {}
        for key in state_dicts[0].keys():
            merged_state[key] = sum(
                w * sd[key].float()
                for w, sd in zip(weights, state_dicts)
            )
        
        # Load into first model
        models[0].load_state_dict(merged_state)
        
        return models[0]


class SLERPMerger(ModelMerger):
    """
    Spherical Linear Interpolation (SLERP) Merger.
    
    Interpolates on the unit sphere for better weight interpolation.
    
    Reference: "Spherical Linear Interpolation" (Shoemake, 1985)
    """
    
    def merge(
        self,
        models: List[nn.Module],
        config: MergeConfig,
    ) -> nn.Module:
        """Merge models using SLERP."""
        if len(models) != 2:
            raise ValueError("SLERP requires exactly 2 models")
        
        t = config.weights[0] if config.weights else 0.5
        
        state1 = models[0].state_dict()
        state2 = models[1].state_dict()
        
        merged_state = {}
        
        for key in state1.keys():
            w1 = state1[key].float()
            w2 = state2[key].float()
            
            # Normalize
            w1_norm = w1 / w1.norm()
            w2_norm = w2 / w2.norm()
            
            # Compute angle
            dot = (w1_norm * w2_norm).sum()
            dot = dot.clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(dot)
            
            # SLERP
            if theta < 1e-7:
                merged = (1 - t) * w1 + t * w2
            else:
                sin_theta = torch.sin(theta)
                merged = (
                    torch.sin((1 - t) * theta) / sin_theta * w1 +
                    torch.sin(t * theta) / sin_theta * w2
                )
            
            merged_state[key] = merged
        
        models[0].load_state_dict(merged_state)
        return models[0]


class DAREMerger(ModelMerger):
    """
    Drop And REscale (DARE) Merger.
    
    Drops random weights and rescales to maintain performance.
    
    Reference: "DARE: Drop And REscale for Model Merging" (Yu et al., 2024)
    """
    
    def merge(
        self,
        models: List[nn.Module],
        config: MergeConfig,
    ) -> nn.Module:
        """Merge models using DARE."""
        density = config.density
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        state_dicts = [m.state_dict() for m in models]
        
        merged_state = {}
        
        for key in state_dicts[0].keys():
            # Stack all model weights
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            
            # Random drop
            mask = torch.rand_like(stacked[0]) < density
            
            # Average with mask
            delta = stacked - stacked[0:1]
            delta = delta * mask.float()
            
            # Rescale
            delta = delta / density
            
            # Merge
            merged = stacked[0] + sum(
                w * d for w, d in zip(weights, delta)
            )
            
            merged_state[key] = merged
        
        models[0].load_state_dict(merged_state)
        return models[0]


class TIESMerger(ModelMerger):
    """
    TIES (Task Interpolation with Elect Sign) Merger.
    
    Resolves interference by electing dominant signs.
    
    Reference: "TIES-Merging: Resolving Interference When Merging Models" (Yadav et al., 2023)
    """
    
    def merge(
        self,
        models: List[nn.Module],
        config: MergeConfig,
    ) -> nn.Module:
        """Merge models using TIES."""
        density = config.density
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        # Reference model (first model)
        ref_state = models[0].state_dict()
        state_dicts = [m.state_dict() for m in models]
        
        merged_state = {}
        
        for key in ref_state.keys():
            # Compute deltas from reference
            deltas = [
                (sd[key] - ref_state[key]).float()
                for sd in state_dicts
            ]
            
            # Stack deltas
            stacked = torch.stack(deltas)
            
            # Trim: keep top-k by magnitude
            k = int(density * stacked.numel())
            flat = stacked.abs().flatten()
            threshold = flat.topk(k).values[-1]
            
            trimmed = stacked * (stacked.abs() >= threshold).float()
            
            # Elect sign: majority vote
            signs = torch.sign(trimmed)
            signs[signs == 0] = 0
            elected_sign = torch.sign(signs.sum(dim=0))
            
            # Disjoint merge: keep only weights with elected sign
            merged = torch.zeros_like(ref_state[key])
            
            for i, (delta, w) in enumerate(zip(trimmed, weights)):
                mask = torch.sign(delta) == elected_sign
                merged = merged + w * delta * mask.float()
            
            merged_state[key] = ref_state[key] + merged
        
        models[0].load_state_dict(merged_state)
        return models[0]


class TaskArithmetic(ModelMerger):
    """
    Task Arithmetic for Model Merging.
    
    Adds task vectors to base model.
    
    Reference: "Task Arithmetic in the Tangent Space" (Ilharco et al., 2022)
    """
    
    def merge(
        self,
        models: List[nn.Module],
        config: MergeConfig,
    ) -> nn.Module:
        """Merge using task arithmetic."""
        weights = config.weights or [1.0] * len(models)
        
        # First model is base
        base_state = models[0].state_dict()
        
        merged_state = {k: v.clone() for k, v in base_state.items()}
        
        # Add task vectors
        for i, model in enumerate(models[1:], 1):
            task_state = model.state_dict()
            w = weights[i] if i < len(weights) else 1.0
            
            for key in base_state.keys():
                task_vector = task_state[key] - base_state[key]
                merged_state[key] = merged_state[key] + w * task_vector
        
        models[0].load_state_dict(merged_state)
        return models[0]


class ModelSoups(ModelMerger):
    """
    Model Soups: Averaging Fine-tuned Models.
    
    Averages weights of multiple fine-tuned models.
    
    Reference: "Model Soups: Averaging Weights of Multiple Fine-tuned Models" (Wortsman et al., 2022)
    """
    
    def merge(
        self,
        models: List[nn.Module],
        config: MergeConfig,
    ) -> nn.Module:
        """Merge models using uniform averaging."""
        weights = config.weights or [1.0 / len(models)] * len(models)
        
        state_dicts = [m.state_dict() for m in models]
        
        merged_state = {}
        
        for key in state_dicts[0].keys():
            merged_state[key] = sum(
                w * sd[key].float()
                for w, sd in zip(weights, state_dicts)
            )
        
        models[0].load_state_dict(merged_state)
        return models[0]
    
    def greedy_soup(
        self,
        models: List[nn.Module],
        validation_fn,
    ) -> nn.Module:
        """Greedy soup: iteratively add models that improve validation."""
        best_model = models[0]
        best_score = validation_fn(best_model)
        
        soup_state = {k: v.clone() for k, v in best_model.state_dict().items()}
        soup_models = [best_model]
        
        for model in models[1:]:
            # Try adding this model
            trial_state = {}
            n = len(soup_models) + 1
            
            for key in soup_state.keys():
                trial_state[key] = (
                    sum(m.state_dict()[key] for m in soup_models) +
                    model.state_dict()[key]
                ) / n
            
            # Create trial model
            trial_model = type(best_model)()
            trial_model.load_state_dict(trial_state)
            
            # Evaluate
            score = validation_fn(trial_model)
            
            if score > best_score:
                best_score = score
                soup_state = trial_state
                soup_models.append(model)
                logger.info(f"Added model to soup, new score: {score}")
        
        best_model.load_state_dict(soup_state)
        return best_model


def merge_models(
    models: List[nn.Module],
    method: str = "linear",
    weights: Optional[List[float]] = None,
    **kwargs,
) -> nn.Module:
    """
    Convenience function to merge models.
    
    Args:
        models: Models to merge
        method: Merging method
        weights: Optional weights for each model
        **kwargs: Additional config options
    
    Returns:
        Merged model
    """
    config = MergeConfig(method=method, weights=weights, **kwargs)
    
    mergers = {
        'linear': LinearMerger(),
        'slerp': SLERPMerger(),
        'dare': DAREMerger(),
        'ties': TIESMerger(),
        'task_arithmetic': TaskArithmetic(),
        'soup': ModelSoups(),
    }
    
    if method not in mergers:
        raise ValueError(f"Unknown method: {method}")
    
    return mergers[method].merge(models, config)
