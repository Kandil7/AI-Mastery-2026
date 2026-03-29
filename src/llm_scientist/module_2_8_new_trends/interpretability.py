"""
Interpretability - Module 2.8.3

Model interpretability: Sparse Autoencoders, Feature Visualization, Activation Patching.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability analysis."""
    sparse_autoencoder_dim: int = 32768
    sparsity_penalty: float = 1e-4
    num_features_to_visualize: int = 100


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for Feature Discovery.
    
    Learns sparse features from model activations.
    
    Reference: "Sparse Autoencoders Find Highly Interpretable Features" (Cunningham et al., 2023)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32768,
        sparsity_penalty: float = 1e-4,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_penalty = sparsity_penalty
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        # Initialize decoder as transpose of encoder
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder as transpose
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features."""
        return F.relu(self.encoder(x))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features."""
        return self.decoder(features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss with sparsity penalty."""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x)
        
        # Sparsity loss (L1 on features)
        sparsity_loss = features.abs().mean()
        
        # Total loss
        return recon_loss + self.sparsity_penalty * sparsity_loss
    
    def train_step(
        self,
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Single training step."""
        optimizer.zero_grad()
        
        reconstruction, features = self(x)
        loss = self.compute_loss(x, reconstruction, features)
        
        loss.backward()
        optimizer.step()
        
        # Tie weights
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        
        return {
            'loss': loss.item(),
            'recon_loss': F.mse_loss(reconstruction, x).item(),
            'sparsity_loss': features.abs().mean().item(),
            'active_features': (features > 0).float().mean().item(),
        }


class FeatureVisualizer:
    """
    Feature Visualizer for SAE features.
    
    Visualizes what activates specific features.
    """
    
    def __init__(self, autoencoder: SparseAutoencoder):
        self.autoencoder = autoencoder
    
    def get_top_activations(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top activating examples for a feature."""
        feature_activations = activations[:, feature_idx]
        
        top_values, top_indices = feature_activations.topk(top_k)
        
        return top_values, top_indices
    
    def decode_feature(
        self,
        feature_idx: int,
    ) -> torch.Tensor:
        """Decode a single feature to input space."""
        # Create one-hot feature vector
        feature_vec = torch.zeros(1, self.autoencoder.hidden_dim)
        feature_vec[0, feature_idx] = 1.0
        
        # Decode
        return self.autoencoder.decode(feature_vec)
    
    def analyze_feature(
        self,
        data: torch.Tensor,
        feature_idx: int,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Analyze a specific feature."""
        # Get activations
        with torch.no_grad():
            _, features = self.autoencoder(data)
        
        # Get top activations
        top_values, top_indices = self.get_top_activations(features, feature_idx)
        
        # Get decoder direction
        decoder_direction = self.decode_feature(feature_idx)
        
        result = {
            'feature_idx': feature_idx,
            'top_activation_values': top_values.tolist(),
            'top_activation_indices': top_indices.tolist(),
            'decoder_direction_norm': decoder_direction.norm().item(),
        }
        
        if tokenizer:
            # Decode top examples to text
            top_texts = []
            for idx in top_indices:
                text = tokenizer.decode(data[idx].long())
                top_texts.append(text)
            result['top_texts'] = top_texts
        
        return result


class ActivationPatcher:
    """
    Activation Patching for Causal Analysis.
    
    Patches activations to test causal relationships.
    
    Reference: "A Framework for the Quantitative Visualization of Semantics in Deep NLP Models" (Vig et al., 2020)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
    
    def register_hook(
        self,
        module_name: str,
        hook_fn: Optional[callable] = None,
    ) -> None:
        """Register hook on module."""
        module = dict(self.model.named_modules())[module_name]
        
        def hook(module, input, output):
            self.activations[module_name] = output.detach()
            if hook_fn:
                hook_fn(module, input, output)
        
        self.hooks.append(module.register_forward_hook(hook))
    
    def patch_activation(
        self,
        module_name: str,
        patch_value: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Patch activation for a module."""
        module = dict(self.model.named_modules())[module_name]
        
        def patch_hook(module, input, output):
            if indices is not None:
                output = output.clone()
                output[indices] = patch_value
            else:
                output = patch_value
            return output
        
        hook = module.register_forward_hook(patch_hook)
        self.hooks.append(hook)
    
    def run_with_patching(
        self,
        inputs: Dict[str, torch.Tensor],
        patches: Dict[str, torch.Tensor],
    ) -> Any:
        """Run model with activation patches."""
        # Register patches
        for module_name, patch_value in patches.items():
            self.patch_activation(module_name, patch_value)
        
        # Run model
        with torch.no_grad():
            output = self.model(**inputs)
        
        # Clear hooks
        self.clear_hooks()
        
        return output
    
    def causal_tracing(
        self,
        clean_inputs: Dict[str, torch.Tensor],
        corrupted_inputs: Dict[str, torch.Tensor],
        target_module: str,
    ) -> torch.Tensor:
        """
        Perform causal tracing.
        
        Tests if restoring clean activation at a module
        restores clean output.
        """
        # Get clean output
        clean_output = self.model(**clean_inputs)
        
        # Get corrupted output
        corrupted_output = self.model(**corrupted_inputs)
        
        # Get clean activations
        with torch.no_grad():
            self.model(**clean_inputs)
        clean_activation = self.activations.get(target_module)
        
        # Run corrupted with clean activation patch
        restored_output = self.run_with_patching(
            corrupted_inputs,
            {target_module: clean_activation},
        )
        
        # Compute restoration score
        if isinstance(clean_output, tuple):
            clean_output = clean_output[0]
            corrupted_output = corrupted_output[0]
            restored_output = restored_output[0]
        
        clean_diff = (clean_output - corrupted_output).abs().mean()
        restored_diff = (restored_output - corrupted_output).abs().mean()
        
        restoration_score = 1 - (restored_diff / (clean_diff + 1e-8))
        
        return restoration_score
    
    def clear_hooks(self) -> None:
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


class InterpretabilityAnalyzer:
    """
    Complete Interpretability Analysis System.
    
    Combines SAE, feature visualization, and activation patching.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: InterpretabilityConfig,
    ):
        self.model = model
        self.config = config
        
        # Initialize components
        self.autoencoders: Dict[str, SparseAutoencoder] = {}
        self.patcher = ActivationPatcher(model)
    
    def train_autoencoder(
        self,
        module_name: str,
        activations: torch.Tensor,
        num_epochs: int = 100,
    ) -> SparseAutoencoder:
        """Train SAE on module activations."""
        input_dim = activations.shape[-1]
        
        autoencoder = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.config.sparse_autoencoder_dim,
            sparsity_penalty=self.config.sparsity_penalty,
        )
        
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            # Shuffle
            indices = torch.randperm(len(activations))
            activations = activations[indices]
            
            # Train in batches
            for i in range(0, len(activations), 256):
                batch = activations[i:i+256]
                metrics = autoencoder.train_step(batch, optimizer)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={metrics['loss']:.4f}")
        
        self.autoencoders[module_name] = autoencoder
        
        return autoencoder
    
    def analyze_module(
        self,
        module_name: str,
        data: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Analyze a module's activations."""
        if module_name not in self.autoencoders:
            raise ValueError(f"No autoencoder for {module_name}")
        
        autoencoder = self.autoencoders[module_name]
        visualizer = FeatureVisualizer(autoencoder)
        
        # Get activations
        with torch.no_grad():
            _, features = autoencoder(data)
        
        # Analyze top features
        results = {}
        
        for feature_idx in range(min(self.config.num_features_to_visualize, features.shape[1])):
            results[feature_idx] = visualizer.analyze_feature(
                data,
                feature_idx,
                tokenizer,
            )
        
        return results
    
    def causal_analysis(
        self,
        module_name: str,
        clean_inputs: Dict[str, torch.Tensor],
        corrupted_inputs: Dict[str, torch.Tensor],
    ) -> float:
        """Perform causal analysis on a module."""
        return self.patcher.causal_tracing(
            clean_inputs,
            corrupted_inputs,
            module_name,
        )
