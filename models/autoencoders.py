"""Keluarga Autoencoder: Vanilla AE, Sparse AE (penalti L1), dan Variational Autoencoder (VAE)."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAutoencoder(nn.Module):
    """
    Autoencoder deterministik standar:
      Encoder: 784 -> 256 (ReLU) -> latent_dim (64)
      Decoder: latent_dim (64) -> 256 (ReLU) -> 784 (Sigmoid)
    Loss: Pixel-wise Mean Squared Error (MSE).
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder dengan regularisasi L1 sparsity pada aktivasi laten h:
      Encoder: 784 -> 256 (ReLU) -> latent_dim (64, ReLU)
      Decoder: latent_dim (64) -> 256 (ReLU) -> 784 (Sigmoid)
    Loss: MSE(x, x_hat) + lambda * mean(|h|)
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    @staticmethod
    def loss_function(
        x_hat: torch.Tensor, x: torch.Tensor, h: torch.Tensor, l1_lambda: float = 2e-3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        l1_penalty = torch.mean(torch.abs(h))
        total_loss = recon_loss + l1_lambda * l1_penalty
        return total_loss, recon_loss, l1_penalty


class VAE(nn.Module):
    """
    Variational Autoencoder (Kingma & Welling, 2013):
      Encoder: 784 -> 256 (ReLU) -> (mu: 64, logvar: 64)
      Reparameterization Trick: z = mu + exp(0.5 * logvar) * eps, eps ~ N(0, I)
      Decoder: 64 -> 256 (ReLU) -> 784 (Sigmoid)
    Loss (Negative ELBO): BCE_sum(x_hat, x) + beta * D_KL(q(z|x) || N(0, I))
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_fc(x.view(x.size(0), -1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @staticmethod
    def loss_function(
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        recon_bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        total_loss = recon_bce + beta * kld
        return total_loss, recon_bce, kld
