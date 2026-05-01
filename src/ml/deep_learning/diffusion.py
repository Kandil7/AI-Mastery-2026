"""
Diffusion Models Implementation from Scratch

This module implements various diffusion models including:
- DDPM (Denoising Diffusion Probabilistic Models)
- DDIM (Denoising Implicit Diffusion Models)
- Stable Diffusion components
- Various noise schedules and sampling strategies

All implementations follow the "white-box" philosophy - understanding
every component from mathematical foundations to implementation details.
"""

import math
from typing import Optional, Tuple, List, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np

# Try to import PyTorch - fall back to pure numpy if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model training."""

    image_size: int = 32
    channels: int = 3
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    noise_schedule: str = "linear"  # linear, cosine, exponential
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    latent_dim: int = 4  # For VAE in latent diffusion
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (16, 8)
    num_heads: int = 4
    dropout: float = 0.1


class NoiseSchedule:
    """
    Noise schedule for diffusion process.

    The noise schedule defines how noise is added at each timestep.
    Different schedules offer different trade-offs between quality and speed.
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.betas = self._create_schedule()

        # Precompute diffusion constants
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Compute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Compute values for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Log variance for posterior
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.clip(self.posterior_variance, 1e-20, None)
        )

        # Compute mean coefficients for posterior
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _create_schedule(self) -> np.ndarray:
        """Create the noise schedule based on configuration."""
        if self.config.noise_schedule == "linear":
            return self._linear_schedule()
        elif self.config.noise_schedule == "cosine":
            return self._cosine_schedule()
        elif self.config.noise_schedule == "exponential":
            return self._exponential_schedule()
        else:
            raise ValueError(f"Unknown schedule: {self.config.noise_schedule}")

    def _linear_schedule(self) -> np.ndarray:
        """Linear schedule from beta_start to beta_end."""
        return np.linspace(
            self.config.beta_start, self.config.beta_end, self.config.num_timesteps
        )

    def _cosine_schedule(self) -> np.ndarray:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672.

        This schedule is designed to keep the signal-to-noise ratio higher
        for longer, which often leads to better sample quality.
        """
        steps = self.config.num_timesteps + 1
        x = np.linspace(0, self.config.num_timesteps, steps)
        alphas_cumprod = (
            np.cos(((x / self.config.num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)

    def _exponential_schedule(self) -> np.ndarray:
        """Exponential schedule for faster noise accumulation."""
        return (
            self.config.beta_end
            * np.linspace(1, self.config.num_timesteps, self.config.num_timesteps)
            / self.config.num_timesteps
        )

    def add_noise(
        self, x0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to clean images at timestep t.

        Args:
            x0: Clean images of shape (batch_size, channels, height, width)
            t: Timesteps of shape (batch_size,)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy images, added noise)
        """
        if noise is None:
            noise = np.random.randn(*x0.shape)

        # Get the appropriate values for each timestep
        sqrt_alpha_prod = self._get_values_at_timesteps(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha_prod = self._get_values_at_timesteps(
            self.sqrt_one_minus_alphas_cumprod, t
        )

        # Reshape for broadcasting
        for _ in range(len(x0.shape) - 1):
            sqrt_alpha_prod = sqrt_alpha_prod[..., None]
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., None]

        # q(x_t | x_0) = sqrt(alpha_prod) * x_0 + sqrt(1 - alpha_prod) * noise
        xt = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

        return xt, noise

    def _get_values_at_timesteps(self, values: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Get values from array at specified timesteps."""
        return values[t]

    def q_sample(
        self, x0: np.ndarray, t: np.ndarray, noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward diffusion process - add noise to x0 at timestep t."""
        xt, _ = self.add_noise(x0, t, noise)
        return xt

    def q_posterior_mean_variance(
        self, x0: np.ndarray, xt: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute mean and variance of posterior q(x_{t-1} | x_t, x_0).

        This is used for training to predict x0 from xt.
        """
        # Get coefficients
        posterior_mean_coef1 = self._expand_values(
            self.posterior_mean_coef1, t, xt.shape
        )
        posterior_mean_coef2 = self._expand_values(
            self.posterior_mean_coef2, t, xt.shape
        )
        posterior_log_variance = self._expand_values(
            self.posterior_log_variance_clipped, t, xt.shape
        )

        # Compute mean
        mean = posterior_mean_coef1 * x0 + posterior_mean_coef2 * xt

        # Compute variance (log variance for numerical stability)
        variance = np.exp(posterior_log_variance)

        return mean, variance, posterior_log_variance

    def _expand_values(
        self, values: np.ndarray, t: np.ndarray, shape: Tuple
    ) -> np.ndarray:
        """Expand values to match the shape for broadcasting."""
        result = values[t]
        for _ in range(len(shape) - 1):
            result = result[..., None]
        return result

    def predict_start_from_noise(
        self, xt: np.ndarray, t: np.ndarray, noise: np.ndarray
    ) -> np.ndarray:
        """
        Predict x0 from xt and the noise.

        Used in DDIM sampling.
        """
        sqrt_recip_alpha_prod = self._expand_values(
            self.sqrt_recip_alphas_cumprod, t, xt.shape
        )
        sqrt_recipm1_alpha_prod = self._expand_values(
            self.sqrt_recipm1_alphas_cumprod, t, xt.shape
        )

        # x0 = (xt - sqrt(1-alpha_prod) * noise) / sqrt(alpha_prod)
        model_pred = (xt - sqrt_recipm1_alpha_prod * noise) / sqrt_recip_alpha_prod

        return model_pred

    def q_posterior_sample(
        self,
        x0: np.ndarray,
        xt: np.ndarray,
        t: np.ndarray,
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample from posterior q(x_{t-1} | x_t, x_0)."""
        mean, variance, _ = self.q_posterior_mean_variance(x0, xt, t)

        if noise is None:
            noise = np.random.randn(*xt.shape)

        # Sample: x_{t-1} = mean + sqrt(variance) * noise
        return mean + np.sqrt(variance) * noise


class SinusoidalPositionEmbeddings(nn.Module if HAS_TORCH else object):
    """
    Sinusoidal position embeddings for timestep conditioning.

    Used to embed the timestep t into a high-dimensional space
    that the model can use for conditioning.
    """

    def __init__(self, dim: int, max_positions: int = 10000):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SinusoidalPositionEmbeddings")
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timestep tensor of shape (batch_size,)
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2

        # Create frequency bands
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Compute embeddings
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class ResBlock(nn.Module if HAS_TORCH else object):
    """
    Residual block with time embedding conditioning.

    Used in UNet architecture for diffusion models.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ResBlock")
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, channels, height, width)
            t_emb: Time embedding of shape (batch, time_emb_dim)
        Returns:
            Output of shape (batch, out_channels, height, width)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_emb(self.act(t_emb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module if HAS_TORCH else object):
    """
    Spatial attention block for feature maps.

    Applies self-attention over spatial dimensions while keeping
    channel dimension fixed.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for AttentionBlock")
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, channels, height, width)
        Returns:
            Output of shape (batch, channels, height, width)
        """
        b, c, h, w = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Reshape for attention: (b, h*w, c)
        x_flat = x_norm.reshape(b, c, h * w).transpose(1, 2)

        # Compute Q, K, V
        qkv = self.qkv(x_flat)
        q, k, v = qkv.split(self.channels, dim=-1)

        # Multi-head attention
        q = q.reshape(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.reshape(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.reshape(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)

        # Scaled dot-product attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, h * w, c)

        # Project and reshape back
        out = self.proj(out).transpose(1, 2).reshape(b, c, h, w)

        return x + out


class Downsample(nn.Module if HAS_TORCH else object):
    """Downsampling layer for UNet."""

    def __init__(self, channels: int):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Downsample")
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module if HAS_TORCH else object):
    """Upsampling layer for UNet."""

    def __init__(self, channels: int):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Upsample")
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module if HAS_TORCH else object):
    """
    UNet architecture for diffusion models.

    This is the core denoising network that predicts noise from
    noisy images at each timestep.

    Architecture:
    - Encoder (downsampling path)
    - Middle (bottleneck with attention)
    - Decoder (upsampling path with skip connections)
    """

    def __init__(self, config: DiffusionConfig):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for UNet")
        super().__init__()
        self.config = config

        # Time embedding
        time_dim = config.channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.channels),
            nn.Linear(config.channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(config.channels, 64, 3, padding=1)

        # Encoder blocks
        self.down1 = self._make_down_block(64, 128, time_dim, config.dropout)
        self.down2 = self._make_down_block(128, 256, time_dim, config.dropout)
        self.down3 = self._make_down_block(256, 512, time_dim, config.dropout)

        # Middle blocks
        self.mid1 = ResBlock(512, 512, time_dim, config.dropout)
        self.mid2 = ResBlock(512, 512, time_dim, config.dropout)

        # Decoder blocks
        self.up1 = self._make_up_block(512, 256, time_dim, config.dropout)
        self.up2 = self._make_up_block(256, 128, time_dim, config.dropout)
        self.up3 = self._make_up_block(128, 64, time_dim, config.dropout)

        # Output
        self.norm_out = nn.GroupNorm(8, 64)
        self.conv_out = nn.Conv2d(64, config.channels, 3, padding=1)

        # Downsampling for skip connections
        self.downsample = Downsample(64)

    def _make_down_block(
        self, in_ch: int, out_ch: int, time_dim: int, dropout: float
    ) -> nn.Module:
        return nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, time_dim, dropout),
                ResBlock(out_ch, out_ch, time_dim, dropout),
                Downsample(out_ch) if out_ch != 64 else nn.Identity(),
            ]
        )

    def _make_up_block(
        self, in_ch: int, out_ch: int, time_dim: int, dropout: float
    ) -> nn.Module:
        return nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, time_dim, dropout),
                ResBlock(out_ch, out_ch, time_dim, dropout),
                Upsample(out_ch),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy image of shape (batch, channels, height, width)
            t: Timesteps of shape (batch,)
        Returns:
            Predicted noise of shape (batch, channels, height, width)
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial convolution
        h = self.conv_in(x)

        # Encoder with skip connections
        hs = []
        for block in self.down1:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        hs.append(h)

        for block in self.down2:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        hs.append(h)

        for block in self.down3:
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
            else:
                h = block(h)
        hs.append(h)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder with skip connections
        h = self.up1[0](torch.cat([h, hs.pop()], dim=1), t_emb)
        h = self.up1[1](h, t_emb)
        h = self.up1[2](h)

        h = self.up2[0](torch.cat([h, hs.pop()], dim=1), t_emb)
        h = self.up2[1](h, t_emb)
        h = self.up2[2](h)

        h = self.up3[0](torch.cat([h, hs.pop()], dim=1), t_emb)
        h = self.up3[1](h, t_emb)
        h = self.up3[2](h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convenience method to predict noise."""
        return self.forward(x, t)


class DDPM(nn.Module if HAS_TORCH else object):
    """
    Denoising Diffusion Probabilistic Model (DDPM).

    Implementation of the paper "Denoising Diffusion Probabilistic Models"
    (Ho, Jain, Abbeel, 2020).

    Training:
    1. Sample x0 from data distribution
    2. Sample timestep t uniformly
    3. Add noise epsilon at timestep t
    4. Predict noise using UNet

    Sampling (DDPM):
    1. Start from random noise xT
    2. For t = T to 1:
       - Predict x0 from (xt, t)
       - Sample xt-1 from posterior
    """

    def __init__(self, config: DiffusionConfig):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DDPM")
        super().__init__()
        self.config = config
        self.schedule = NoiseSchedule(config)
        self.model = UNet(config)
        self.device = None

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x0: Clean images of shape (batch, channels, height, width)
            t: Timesteps of shape (batch,)
        Returns:
            Predicted noise of shape (batch, channels, height, width)
        """
        # Sample noise
        noise = torch.randn_like(x0)

        # Add noise to x0
        sqrt_alpha_prod = self.schedule.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_prod = self.schedule.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ]

        xt = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

        # Predict noise
        return self.model(xt, t)

    def training_loss(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss (simplified MSE on noise).

        Args:
            x0: Clean images
            t: Timesteps
            noise: Optional pre-generated noise

        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Add noise
        sqrt_alpha_prod = self.schedule.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_prod = self.schedule.sqrt_one_minus_alphas_cumprod[t][
            :, None, None, None
        ]

        xt = sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise

        # Predict and compute loss
        noise_pred = self.model(xt, t)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(
        self, shape: Tuple, device: str = "cpu", return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Sample images using DDPM sampling.

        Args:
            shape: Shape of images to generate (batch, channels, height, width)
            device: Device to run on
            return_intermediates: Whether to return all intermediate steps

        Returns:
            Generated images
        """
        self.device = device
        self.to(device)

        batch_size = shape[0]
        xt = torch.randn(shape, device=device)

        intermediates = [] if return_intermediates else None

        # Reverse diffusion process
        for t in reversed(range(self.config.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(xt, t_batch)

            # Compute predicted x0
            sqrt_recip_alpha = self.schedule.sqrt_recip_alphas_cumprod[t] ** 0.5
            sqrt_recipm1_alpha = self.schedule.sqrt_recipm1_alphas_cumprod[t] ** 0.5

            x0_pred = (xt - sqrt_recipm1_alpha * noise_pred) / sqrt_recip_alpha

            # Sample from posterior
            if t > 0:
                noise = torch.randn_like(xt)
                mean = (
                    self.schedule.posterior_mean_coef1[t] * x0_pred
                    + self.schedule.posterior_mean_coef2[t] * xt
                )
                variance = self.schedule.posterior_variance[t]
                xt = mean + variance**0.5 * noise
            else:
                xt = x0_pred

            if return_intermediates:
                intermediates.append(xt.cpu())

        return xt if not return_intermediates else torch.stack(intermediates)

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple,
        device: str = "cpu",
        num_steps: int = 50,
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Sample images using DDIM (Denoising Implicit Diffusion Models).

        DDIM is much faster than DDPM while often producing similar quality.

        Args:
            shape: Shape of images to generate
            device: Device to run on
            num_steps: Number of sampling steps (typically 20-100)
            eta: DDIM stochasticity parameter (0 = deterministic)
            return_intermediates: Whether to return intermediate steps
        """
        self.device = device
        self.to(device)

        batch_size = shape[0]
        step_size = self.config.num_timesteps // num_steps

        # Start from random noise
        xt = torch.randn(shape, device=device)

        intermediates = [] if return_intermediates else None

        # Selected timesteps
        timesteps = [i * step_size for i in range(num_steps)]

        for i, t in enumerate(reversed(timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Next timestep
            t_next = timesteps[max(0, i - 1)] if i > 0 else 0

            # Predict noise
            noise_pred = self.model(xt, t_batch)

            # Compute predicted x0
            sqrt_recip_alpha = self.schedule.sqrt_recip_alphas_cumprod[t] ** 0.5
            sqrt_recipm1_alpha = self.schedule.sqrt_recipm1_alphas_cumprod[t] ** 0.5

            x0_pred = (xt - sqrt_recipm1_alpha * noise_pred) / sqrt_recip_alpha

            # Direction pointing to xt
            pred_x0_coeff = self.schedule.sqrt_alphas_cumprod[t]
            pred_xt_direction = (1 - pred_x0_coeff**2) ** 0.5 * noise_pred

            # Compute previous sample
            alpha_prod_t = self.schedule.alphas_cumprod[t]
            alpha_prod_t_next = self.schedule.alphas_cumprod[t_next]

            # Predicted previous sample
            if t_next > 0:
                pred_x0_coeff_next = self.schedule.sqrt_alphas_cumprod[t_next]
                pred_xt_direction_next = (1 - pred_x0_coeff_next**2) ** 0.5 * noise_pred

                xt_next = pred_x0_coeff_next * x0_pred + pred_xt_direction_next

                # Add noise for DDIM stochasticity
                if eta > 0:
                    sigma = eta * ((1 - alpha_prod_t) / (1 - alpha_prod_t_next)) ** 0.5
                    xt_next = xt_next + sigma * torch.randn_like(xt)
            else:
                xt_next = x0_pred

            xt = xt_next

            if return_intermediates:
                intermediates.append(xt.cpu())

        return xt if not return_intermediates else torch.stack(intermediates)

    def to(self, device):
        """Move model to device."""
        self.device = device
        super().to(device)
        return self


class LatentDiffusionModel(nn.Module if HAS_TORCH else object):
    """
    Latent Diffusion Model (LDM) - base for Stable Diffusion.

    Diffuses in latent space instead of pixel space for efficiency.
    Uses a variational autoencoder (VAE) to encode/decode images.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LatentDiffusionModel")
        super().__init__()
        self.config = config
        self.schedule = NoiseSchedule(config)

        # VAE for encoding/decoding (if not provided, use identity)
        self.vae = vae if vae is not None else nn.Identity()

        # Text encoder for conditioning (if provided)
        self.text_encoder = text_encoder

        # Diffusion model in latent space
        latent_channels = config.latent_dim
        self.model = self._create_latent_unet(latent_channels, config)

        self.device = None

    def _create_latent_unet(self, channels: int, config: DiffusionConfig) -> nn.Module:
        """Create UNet for latent space."""
        # Simplified latent UNet
        return nn.ModuleDict(
            {
                "encoder": nn.ModuleList(
                    [
                        nn.Conv2d(channels, 128, 3, padding=1),
                        ResBlock(128, 128, 128, config.dropout),
                        nn.Conv2d(128, 256, 3, stride=2, padding=1),
                        ResBlock(256, 256, 128, config.dropout),
                    ]
                ),
                "middle": nn.ModuleList(
                    [
                        ResBlock(256, 256, 128, config.dropout),
                        ResBlock(256, 256, 128, config.dropout),
                    ]
                ),
                "decoder": nn.ModuleList(
                    [
                        ResBlock(256, 256, 128, config.dropout),
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                        ResBlock(128, 128, 128, config.dropout),
                        nn.Conv2d(128, channels, 3, padding=1),
                    ]
                ),
                "time_mlp": nn.Sequential(
                    SinusoidalPositionEmbeddings(128),
                    nn.Linear(128, 128),
                    nn.SiLU(),
                    nn.Linear(128, 128),
                ),
            }
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        return self.vae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.vae.decode(z)

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None
    ):
        """Forward pass in latent space."""
        t_emb = self.model["time_mlp"](t)

        # Encoder
        h = z
        for layer in self.model["encoder"]:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # Middle
        for layer in self.model["middle"]:
            h = layer(h, t_emb)

        # Decoder
        for layer in self.model["decoder"]:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        return h

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        device: str = "cpu",
        prompt: Optional[str] = None,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Sample from latent diffusion model."""
        self.device = device
        self.to(device)

        batch_size = shape[0]
        z = torch.randn(shape, device=device)

        # Encode prompt if provided
        if prompt is not None and self.text_encoder is not None:
            cond = self.text_encoder(prompt)

        # Simplified DDIM sampling
        step_size = self.config.num_timesteps // num_steps

        for i in range(num_steps):
            t = self.config.num_timesteps - 1 - i * step_size
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = self(z, t_batch, cond if "cond" in dir() else None)

            # Update z (simplified)
            alpha = self.schedule.alphas_cumprod[t]
            z = (z - (1 - alpha) ** 0.5 * noise_pred) / alpha**0.5

        return self.decode(z)


class ClassifierFreeGuidance:
    """
    Classifier-free guidance implementation.

    Allows for conditional generation without a classifier by
    training an unconditional model alongside the conditional one.

    guidance_scale = 0: unconditional
    guidance_scale = 1: conditional only
    guidance_scale > 1: more conditional (typical: 7-12)
    """

    def __init__(self, model: nn.Module, guidance_scale: float = 7.5):
        self.model = model
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple,
        condition: Optional[torch.Tensor] = None,
        unconditional_condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance.

        score = score_cond + scale * (score_cond - score_uncond)
        """
        # Duplicate for conditional and unconditional
        batch_size = shape[0]

        # Conditional prediction
        if condition is not None:
            cond_input = torch.cat([condition, condition], dim=0)
            noise_pred_cond = self.model(cond_input, **kwargs)[:batch_size]

        # Unconditional prediction
        if unconditional_condition is not None:
            uncond_input = torch.cat(
                [unconditional_condition, unconditional_condition], dim=0
            )
            noise_pred_uncond = self.model(uncond_input, **kwargs)[:batch_size]

        # Apply guidance
        if condition is not None and unconditional_condition is not None:
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        elif condition is not None:
            noise_pred = noise_pred_cond
        else:
            noise_pred = (
                noise_pred_uncond if "noise_pred_uncond" in dir() else noise_pred_cond
            )

        return noise_pred


def create_diffusion_model(
    config: DiffusionConfig, model_type: str = "ddpm"
) -> nn.Module:
    """
    Factory function to create diffusion models.

    Args:
        config: Diffusion configuration
        model_type: Type of model ("ddpm", "ddim", "ldm")

    Returns:
        Diffusion model
    """
    if model_type == "ddpm":
        return DDPM(config)
    elif model_type == "ddim":
        return DDPM(config)  # DDPM can do DDIM sampling
    elif model_type == "ldm":
        return LatentDiffusionModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Pure NumPy fallback implementations for educational purposes
class DiffusionModel_numpy:
    """
    Pure NumPy implementation of diffusion model concepts.

    This is for educational purposes to understand the math behind
    diffusion models without requiring PyTorch.
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.schedule = NoiseSchedule(config)

    def add_noise_numpy(
        self, x0: np.ndarray, t: int, noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Add noise to image using NumPy."""
        if noise is None:
            noise = np.random.randn(*x0.shape)

        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t]

        # Broadcast for multiple images
        for _ in range(len(x0.shape) - 1):
            sqrt_alpha = sqrt_alpha[..., None]
            sqrt_one_minus_alpha = sqrt_one_minus_alpha[..., None]

        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

    def forward_process(self, x0: np.ndarray, num_steps: int = 100) -> List[np.ndarray]:
        """
        Visualize the forward diffusion process.

        Shows how images become progressively noisier.
        """
        trajectory = [x0.copy()]

        for t in range(num_steps):
            xt_prev = trajectory[-1]
            t_batch = np.array([t])

            noise = np.random.randn(*x0.shape)
            xt = self.schedule.q_sample(xt_prev, t_batch, noise)
            trajectory.append(xt)

        return trajectory

    def reverse_process_numpy(
        self,
        xt: np.ndarray,
        num_steps: int = 100,
        model_pred_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Approximate reverse process (sampling) using NumPy.

        This is a simplified version - real implementations would
        use a learned model to predict noise.

        Args:
            xt: Final noisy image
            num_steps: Number of denoising steps
            model_pred_fn: Optional function to predict noise (for real sampling)
        """
        xt_current = xt.copy()
        step_size = self.config.num_timesteps // num_steps

        for i in range(num_steps):
            t = self.config.num_timesteps - 1 - i * step_size

            # In real implementation, this would use model prediction
            # For demo, we just add small amount of "denoising"
            if model_pred_fn is not None:
                noise_pred = model_pred_fn(xt_current, t)
            else:
                # Simplified: just reduce noise slightly
                noise_pred = np.random.randn(*xt.shape) * 0.1

            # Update
            alpha = self.schedule.alphas_cumprod[t]
            xt_current = (xt_current - (1 - alpha) ** 0.5 * noise_pred) / alpha**0.5

        return xt_current


# Export all classes
__all__ = [
    "DiffusionConfig",
    "NoiseSchedule",
    "SinusoidalPositionEmbeddings",
    "ResBlock",
    "AttentionBlock",
    "Downsample",
    "Upsample",
    "UNet",
    "DDPM",
    "LatentDiffusionModel",
    "ClassifierFreeGuidance",
    "create_diffusion_model",
    "DiffusionModel_numpy",
]
