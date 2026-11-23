"""Denoising Diffusion Probabilistic Model (DDPM) dengan U-Net Terkondisi Waktu untuk 28x28."""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Embedding waktu sinusoidal (Vaswani et al. / Ho et al. DDPM)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConvBlock(nn.Module):
    """Blok Konvolusi + GroupNorm + SiLU dengan injeksi time embedding dan residual skip."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        time_bias = F.silu(self.time_mlp(t_emb))[:, :, None, None]
        h = h + time_bias
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.residual(x)


class SimpleUNet28(nn.Module):
    """
    U-Net ringan yang dirancang tepat untuk citra 1x28x28:
      Down: 28x28 (base_ch) -> 14x14 (2*base_ch) -> 7x7 (4*base_ch)
      Mid : 7x7 (4*base_ch)
      Up  : 7x7 -> 14x14 -> 28x28 dengan skip connections
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 48, time_emb_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.enc1 = ConvBlock(in_channels, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)

        self.enc2 = ConvBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        self.bot1 = ConvBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        self.bot2 = ConvBlock(base_ch * 4, base_ch * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 4, base_ch, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 2, base_ch, time_emb_dim)

        self.out_conv = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)

        b = self.bot1(self.down2(e2), t_emb)
        b = self.bot2(b, t_emb)

        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1), t_emb)

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1), t_emb)

        return self.out_conv(d2)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020):
      Forward process : q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
      Training loss   : L_simple = E_{t, x_0, eps} [ || eps - eps_theta(x_t, t) ||^2 ]
      Reverse process : p_theta(x_{t-1} | x_t) iteratif dari x_T ~ N(0, I) hingga x_0.
    """

    def __init__(
        self,
        timesteps: int = 400,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        base_ch: int = 48,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.eps_model = SimpleUNet28(in_channels=1, base_ch=base_ch)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x_0 + sqrt_omab * noise

    def compute_loss(self, x_0_zero_one: torch.Tensor) -> torch.Tensor:
        x_0 = x_0_zero_one * 2.0 - 1.0
        batch_size = x_0.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        pred_noise = self.eps_model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        record_steps: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[int, torch.Tensor]]]]:
        x = torch.randn(num_samples, 1, 28, 28, device=device)
        trajectory = []

        if record_steps is not None and self.timesteps in record_steps:
            trajectory.append((self.timesteps, torch.clamp((x + 1.0) / 2.0, 0.0, 1.0).cpu()))

        for t_idx in reversed(range(self.timesteps)):
            t_batch = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            pred_noise = self.eps_model(x, t_batch)

            beta_t = self.betas[t_idx]
            sqrt_one_minus_ab_t = self.sqrt_one_minus_alphas_cumprod[t_idx]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_idx]

            mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_ab_t) * pred_noise)

            if t_idx > 0:
                sigma_t = torch.sqrt(self.posterior_variance[t_idx])
                x = mean + sigma_t * torch.randn_like(x)
            else:
                x = mean

            if record_steps is not None and t_idx in record_steps:
                trajectory.append((t_idx, torch.clamp((x + 1.0) / 2.0, 0.0, 1.0).cpu()))

        x_out = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        return x_out, (trajectory if record_steps is not None else None)
