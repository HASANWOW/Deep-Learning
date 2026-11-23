"""Keluarga GAN: Vanilla GAN (berbasis MLP) dan Deep Convolutional GAN (DCGAN)."""

import torch
import torch.nn as nn


# ========================== VANILLA GAN (MLP) ==========================
class VanillaGANGenerator(nn.Module):
    """Generator MLP penuh: 100 -> 256 -> 512 -> 784 (Tanh, rentang [-1, 1])."""

    def __init__(self, latent_dim: int = 100, output_dim: int = 784):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, 28, 28)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Menghasilkan sampel citra dalam rentang [0, 1]."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        imgs = self.forward(z)
        return (imgs + 1.0) / 2.0


class VanillaGANDiscriminator(nn.Module):
    """Discriminator MLP penuh: 784 -> 512 -> 256 -> 1 (Logit untuk BCEWithLogitsLoss)."""

    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# ========================== DCGAN (RADFORD ET AL.) ==========================
def dcgan_weights_init(m: nn.Module) -> None:
    """Inisialisasi bobot standar DCGAN (Radford et al., 2015): N(0, 0.02)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class DCGANGenerator(nn.Module):
    """
    Generator Konvolusional Penuh untuk citra 1x28x28:
      z (B, 100, 1, 1)
      -> ConvTranspose2d(100, 256, 7, 1, 0) + BN + ReLU  => (256, 7, 7)
      -> ConvTranspose2d(256, 128, 4, 2, 1) + BN + ReLU  => (128, 14, 14)
      -> ConvTranspose2d(128, 1,   4, 2, 1) + Tanh       => (1, 28, 28) dalam [-1, 1]
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 2, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(dcgan_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Menghasilkan sampel citra dalam rentang [0, 1]."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        imgs = self.forward(z)
        return (imgs + 1.0) / 2.0


class DCGANDiscriminator(nn.Module):
    """
    Discriminator Konvolusional Penuh untuk citra 1x28x28:
      (1, 28, 28)
      -> Conv2d(1,   64,  4, 2, 1) + LeakyReLU(0.2)      => (64, 14, 14)
      -> Conv2d(64,  128, 4, 2, 1) + BN + LeakyReLU(0.2) => (128, 7, 7)
      -> Conv2d(128, 1,   7, 1, 0)                       => (1, 1, 1) [Logit]
    """

    def __init__(self, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),
        )
        self.apply(dcgan_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)
