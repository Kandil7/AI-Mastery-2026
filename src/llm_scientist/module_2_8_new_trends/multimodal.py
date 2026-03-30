"""
Multimodal Models - Module 2.8.2

Vision-language models: CLIP, LLaVA.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VisionLanguageConfig:
    """Configuration for vision-language models."""
    vision_model: str = "vit-base"
    language_model: str = "llama-7b"
    projection_type: str = "linear"
    hidden_size: int = 4096
    vision_hidden_size: int = 768
    image_size: int = 224
    patch_size: int = 16


class VisionEncoder(nn.Module):
    """Vision encoder (ViT-style)."""
    
    def __init__(self, config: VisionLanguageConfig):
        super().__init__()
        
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3,
            config.vision_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        
        num_patches = (config.image_size // config.patch_size) ** 2
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.vision_hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_hidden_size))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.vision_hidden_size,
                nhead=12,
                dim_feedforward=config.vision_hidden_size * 4,
            )
            for _ in range(12)
        ])
        
        self.norm = nn.LayerNorm(config.vision_hidden_size)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images."""
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


class CLIPModel(nn.Module):
    """
    CLIP-style Vision-Language Model.
    
    Contrastive language-image pre-training.
    
    Reference: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
    """
    
    def __init__(self, config: VisionLanguageConfig):
        super().__init__()
        
        self.config = config
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(config)
        
        # Text encoder (simplified)
        self.text_encoder = nn.Embedding(50257, config.hidden_size)
        
        # Projections
        self.vision_projection = nn.Linear(config.vision_hidden_size, config.hidden_size)
        self.text_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images."""
        image_features = self.vision_encoder(images)[:, 0]  # CLS token
        return self.vision_projection(image_features)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text."""
        text_features = self.text_encoder(text)
        text_features = text_features.mean(dim=1)
        return self.text_projection(text_features)
    
    def forward(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for CLIP.
        
        Returns:
            Tuple of (image_features, text_features, logit_scale)
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features, self.logit_scale.exp()
    
    def compute_loss(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP contrastive loss."""
        image_features, text_features, logit_scale = self.forward(images, text)
        
        # Compute similarity matrix
        logits = logit_scale * image_features @ text_features.T
        
        # Contrastive loss
        labels = torch.arange(len(logits), device=logits.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        
        return (loss_i + loss_t) / 2


class LLaVAModel(nn.Module):
    """
    LLaVA-style Vision-Language Model.
    
    Large Language and Vision Assistant.
    
    Reference: "Visual Instruction Tuning" (Liu et al., 2023)
    """
    
    def __init__(self, config: VisionLanguageConfig, language_model: nn.Module):
        super().__init__()
        
        self.config = config
        self.language_model = language_model
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(config)
        
        # Multi-modal projector
        if config.projection_type == "linear":
            self.projector = nn.Linear(config.vision_hidden_size, config.hidden_size)
        elif config.projection_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(config.vision_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projection type: {config.projection_type}")
    
    def encode_multimodal(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images for language model."""
        # Get vision features
        vision_features = self.vision_encoder(images)
        
        # Project to language model dimension
        return self.projector(vision_features)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        image_positions: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for LLaVA.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Optional images
            image_positions: Positions to insert image features
            labels: Optional labels for loss
        
        Returns:
            Logits or loss
        """
        if images is not None:
            # Encode images
            image_features = self.encode_multimodal(images)
            
            # Insert image features into input embeddings
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            if image_positions:
                for i, pos in enumerate(image_positions):
                    input_embeds[:, pos:pos + image_features.shape[1]] = image_features[i:i+1]
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs


class MultimodalProcessor:
    """
    Multimodal Data Processor.
    
    Processes images and text for multimodal models.
    """
    
    def __init__(self, config: VisionLanguageConfig):
        self.config = config
    
    def process_image(
        self,
        image: Any,
    ) -> torch.Tensor:
        """Process image for model input."""
        # This would use torchvision transforms in practice
        # Simplified version:
        if isinstance(image, torch.Tensor):
            return image
        
        # Convert PIL/numpy to tensor
        import numpy as np
        
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        # Resize
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(self.config.image_size, self.config.image_size),
        ).squeeze(0)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor
    
    def process_text(
        self,
        text: str,
        tokenizer: Any,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """Process text for model input."""
        return tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
    
    def process_multimodal(
        self,
        image: Any,
        text: str,
        tokenizer: Any,
    ) -> Dict[str, torch.Tensor]:
        """Process image-text pair."""
        image_tensor = self.process_image(image)
        text_inputs = self.process_text(text, tokenizer)
        
        return {
            'images': image_tensor.unsqueeze(0),
            **text_inputs,
        }
