"""Fungsi evaluasi kuantitatif: MSE, PSNR, SSIM, FID Score, dan Inference Latency."""

import time
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_mse_psnr(x_true: torch.Tensor, x_recon: torch.Tensor, eps: float = 1e-10) -> Tuple[float, float]:
    """Menghitung rata-rata MSE dan PSNR (dB) pada rentang citra [0, 1]."""
    diff_sq = (x_true - x_recon) ** 2
    per_img_mse = diff_sq.view(x_true.size(0), -1).mean(dim=1)
    mse = float(per_img_mse.mean().item())
    psnr_per_img = 10.0 * torch.log10(1.0 / (per_img_mse + eps))
    psnr = float(psnr_per_img.mean().item())
    return mse, psnr


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)


def compute_ssim(
    x_true: torch.Tensor,
    x_recon: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Menghitung Structural Similarity Index (SSIM) Wang et al. (2004) pada GPU/CPU."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    window = _gaussian_window(window_size, sigma, device=x_true.device)
    pad = window_size // 2

    mu1 = F.conv2d(x_true, window, padding=pad)
    mu2 = F.conv2d(x_recon, window, padding=pad)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x_true * x_true, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(x_recon * x_recon, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(x_true * x_recon, window, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return float(ssim_map.mean().item())


@torch.no_grad()
def evaluate_reconstruction_loader(
    recon_fn: Callable[[torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Mengevaluasi Test MSE, Test PSNR (dB), dan Test SSIM secara penuh pada Test Set."""
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    for x, _ in dataloader:
        x = x.to(device)
        x_hat = torch.clamp(recon_fn(x), 0.0, 1.0)
        bs = x.size(0)
        mse, psnr = compute_mse_psnr(x, x_hat)
        ssim = compute_ssim(x, x_hat)
        total_mse += mse * bs
        total_psnr += psnr * bs
        total_ssim += ssim * bs
        total_samples += bs

    return {
        "test_mse": total_mse / total_samples,
        "test_psnr_db": total_psnr / total_samples,
        "test_ssim": total_ssim / total_samples,
    }


@torch.no_grad()
def measure_inference_latency(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int = 5,
    runs: int = 20,
) -> float:
    """Mengukur latensi inferensi rata-rata (milidetik per batch 64 sampel)."""
    for _ in range(warmup):
        _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    times_ms = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(times_ms))


class InceptionFeatureExtractor(nn.Module):
    """Ekstraktor fitur 2048-dim dari InceptionV3 pretrained untuk perhitungan FID Score."""

    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights

        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def extract(self, images_zero_one: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        feats = []
        n = images_zero_one.size(0)
        for start in range(0, n, batch_size):
            batch = images_zero_one[start:start + batch_size].to(self.device)
            if batch.size(1) == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            batch = (batch - self.mean) / self.std
            out = self.model(batch)
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Menghitung Frechet Distance antara dua distribusi Gaussian multivariat."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    return max(0.0, fid)


def compute_fid_score(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    extractor: InceptionFeatureExtractor,
) -> float:
    """Menghitung FID Score antara himpunan citra asli dan citra sintetis."""
    real_feats = extractor.extract(real_images)
    fake_feats = extractor.extract(fake_images)

    mu_r, sigma_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    return calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
