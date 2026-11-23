
import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _to_np(img_tensor: torch.Tensor) -> np.ndarray:
    arr = img_tensor.detach().cpu().squeeze().numpy()
    return np.clip(arr, 0.0, 1.0)


def plot_figure1_reconstruction(
    originals: torch.Tensor,
    recon_vanilla: torch.Tensor,
    recon_sparse: torch.Tensor,
    recon_vae: torch.Tensor,
    class_labels: List[str],
    save_path: str,
) -> None:
    """Figur 1: Grid Rekonstruksi Side-by-Side (8 sampel Test Set)."""
    n = originals.size(0)
    rows = [
        ("Ground Truth", originals),
        ("Vanilla AE", recon_vanilla),
        ("Sparse AE (L1)", recon_sparse),
        ("VAE", recon_vae),
    ]

    fig, axes = plt.subplots(4, n, figsize=(1.7 * n, 7.2))
    fig.suptitle("Figur 1: Perbandingan Rekonstruksi Citra Uji — Vanilla AE, Sparse AE, dan VAE", fontsize=13, fontweight="bold", y=0.98)

    for r_idx, (row_title, imgs) in enumerate(rows):
        for c_idx in range(n):
            ax = axes[r_idx, c_idx]
            ax.imshow(_to_np(imgs[c_idx]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(class_labels[c_idx], fontsize=9, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=10, fontweight="bold", rotation=0, labelpad=48, va="center")

    plt.tight_layout(rect=[0.06, 0.0, 1.0, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_figure2_random_sampling(
    samples_dict: Dict[str, torch.Tensor],
    save_path: str,
) -> None:
    """Figur 2: Grid Random Generative Sampling (4x4 = 16 sampel per model)."""
    panels = [
        ("Panel A: VAE  (z ~ N(0, I))", samples_dict["VAE"]),
        ("Panel B: Vanilla GAN (MLP)", samples_dict["Vanilla GAN"]),
        ("Panel C: DCGAN (Convolutional)", samples_dict["DCGAN"]),
        ("Panel D: DDPM (Reverse Denoising)", samples_dict["DDPM"]),
    ]

    fig = plt.figure(figsize=(12, 11.5))
    fig.suptitle("Figur 2: Grid Sintesis Sampel Acak (4x4) Antarmodel Generatif", fontsize=14, fontweight="bold", y=0.98)

    outer = fig.add_gridspec(2, 2, wspace=0.14, hspace=0.18)
    for p_idx, (title, imgs) in enumerate(panels):
        sub_ax = fig.add_subplot(outer[p_idx // 2, p_idx % 2])
        sub_ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
        sub_ax.axis("off")

        inner = outer[p_idx // 2, p_idx % 2].subgridspec(4, 4, wspace=0.04, hspace=0.04)
        for i in range(16):
            ax = fig.add_subplot(inner[i // 4, i % 4])
            ax.imshow(_to_np(imgs[i]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_figure3_mlp_limitations(
    originals: torch.Tensor,
    class_labels: List[str],
    topdown_imgs: torch.Tensor,
    latent_transpose_imgs: torch.Tensor,
    frozen_dec_imgs: torch.Tensor,
    vanilla_ae_imgs: torch.Tensor,
    prior_sample_imgs: torch.Tensor,
    metrics_frozen: Dict[str, float],
    metrics_ae: Dict[str, float],
    save_path: str,
) -> None:
    """Figur 3: Bukti Empiris Batasan Bobot (W^T) dan Fitur Laten (z_MLP) MLP Classifier."""
    n = originals.size(0)
    rows = [
        ("1. Ground Truth\n(Citra Uji Asli)", originals),
        ("2a. Top-Down W^T\ndari One-Hot y_c", topdown_imgs),
        ("2b. Transpose W^T\ndari z_MLP", latent_transpose_imgs),
        (
            f"3a. Frozen Decoder\n(z_MLP)\nMSE={metrics_frozen['test_mse']:.4f}\nSSIM={metrics_frozen['test_ssim']:.3f}",
            frozen_dec_imgs,
        ),
        (
            f"3b. Vanilla AE\n(z_AE)\nMSE={metrics_ae['test_mse']:.4f}\nSSIM={metrics_ae['test_ssim']:.3f}",
            vanilla_ae_imgs,
        ),
        ("4. Prior Sampling\nz_rand ~ N(mu, sig^2)", prior_sample_imgs),
    ]

    fig, axes = plt.subplots(len(rows), n, figsize=(1.85 * n + 1.8, 11.2))
    fig.suptitle(
        "Figur 3: Pembuktian Empiris Ketidakmampuan Bobot & Latent Features MLP Classifier untuk Image Generation",
        fontsize=12.5, fontweight="bold", y=0.985
    )

    for r_idx, (row_title, imgs) in enumerate(rows):
        for c_idx in range(n):
            ax = axes[r_idx, c_idx]
            ax.imshow(_to_np(imgs[c_idx]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(class_labels[c_idx], fontsize=9.5, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=8.8, fontweight="bold", rotation=0, labelpad=68, va="center")

    plt.tight_layout(rect=[0.11, 0.0, 1.0, 0.955])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_figure4_interpolation(
    interp_dict: Dict[str, torch.Tensor],
    alphas: List[float],
    pair_label: str,
    save_path: str,
) -> None:
    """Figur 4: Linear Latent Space Interpolation z(alpha) = (1 - alpha) z_A + alpha z_B."""
    model_order = ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN"]
    n_cols = len(alphas)
    n_rows = len(model_order)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.85 * n_cols + 1.4, 9.2))
    fig.suptitle(
        f"Figur 4: Interpolasi Linier Ruang Laten z(α) = (1-α)z_A + αz_B  [{pair_label}]",
        fontsize=12.5, fontweight="bold", y=0.98
    )

    for r_idx, name in enumerate(model_order):
        imgs = interp_dict[name]
        for c_idx, alpha in enumerate(alphas):
            ax = axes[r_idx, c_idx]
            ax.imshow(_to_np(imgs[c_idx]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"α = {alpha:.1f}", fontsize=10, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold", rotation=0, labelpad=48, va="center")

    plt.tight_layout(rect=[0.08, 0.0, 1.0, 0.945])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_figure5_losses_and_ddpm_trajectory(
    histories: Dict[str, dict],
    ddpm_trajectory: List[Tuple[int, torch.Tensor]],
    save_path: str,
) -> None:
    """Figur 5: Kurva Training Loss (6 Model) & Sekuens Reverse Diffusion DDPM (x_T -> x_0)."""
    fig = plt.figure(figsize=(15, 11.5))
    fig.suptitle(
        "Figur 5: Dinamika Kurva Training Loss (6 Model) & Sekuens Reverse Diffusion DDPM (x_T → x_0)",
        fontsize=13.5, fontweight="bold", y=0.985
    )

    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95], hspace=0.38, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ep = range(1, len(histories["Vanilla AE"]["train_loss"]) + 1)
    ax1.plot(ep, histories["Vanilla AE"]["train_loss"], label="Train MSE", color="#1f77b4", lw=2)
    ax1.plot(ep, histories["Vanilla AE"]["val_loss"], label="Val MSE", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Vanilla Autoencoder (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ep = range(1, len(histories["Sparse AE"]["train_loss"]) + 1)
    ax2.plot(ep, histories["Sparse AE"]["train_loss"], label="Total (MSE + λL1)", color="#2ca02c", lw=2)
    ax2.plot(ep, histories["Sparse AE"]["recon_loss"], label="Recon MSE", color="#9467bd", lw=1.8, linestyle="--")
    ax2.set_title("Sparse Autoencoder (MSE + λ||h||₁)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ep = range(1, len(histories["VAE"]["train_loss"]) + 1)
    ax3.plot(ep, histories["VAE"]["train_loss"], label="Total (-ELBO)", color="#d62728", lw=2)
    ax3.plot(ep, histories["VAE"]["recon_loss"], label="Recon BCE", color="#1f77b4", lw=1.6, linestyle="--")
    ax3.plot(ep, histories["VAE"]["kl_loss"], label="KL Divergence", color="#2ca02c", lw=1.6, linestyle=":")
    ax3.set_title("Variational Autoencoder (VAE)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0])
    ep = range(1, len(histories["Vanilla GAN"]["g_loss"]) + 1)
    ax4.plot(ep, histories["Vanilla GAN"]["g_loss"], label="Generator Loss", color="#e377c2", lw=1.8)
    ax4.plot(ep, histories["Vanilla GAN"]["d_loss"], label="Discriminator Loss", color="#17becf", lw=1.8)
    ax4.set_title("Vanilla GAN (MLP) — Minimax Dynamics")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("BCE Loss")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    ep = range(1, len(histories["DCGAN"]["g_loss"]) + 1)
    ax5.plot(ep, histories["DCGAN"]["g_loss"], label="Generator Loss", color="#ff7f0e", lw=1.8)
    ax5.plot(ep, histories["DCGAN"]["d_loss"], label="Discriminator Loss", color="#1f77b4", lw=1.8)
    ax5.set_title("DCGAN (Conv) — Minimax Dynamics")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("BCE Loss")
    ax5.grid(alpha=0.3)
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    ep = range(1, len(histories["DDPM"]["train_loss"]) + 1)
    ax6.plot(ep, histories["DDPM"]["train_loss"], label="Noise Pred MSE", color="#8c564b", lw=2)
    ax6.set_title("DDPM — Supervised Noise Prediction Loss")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("L_simple (MSE)")
    ax6.grid(alpha=0.3)
    ax6.legend(fontsize=8)

    n_steps = len(ddpm_trajectory)
    n_examples = min(2, ddpm_trajectory[0][1].size(0))
    traj_sub = gs[2, :].subgridspec(n_examples, n_steps, wspace=0.08, hspace=0.12)

    title_ax = fig.add_subplot(gs[2, :])
    title_ax.set_title("Sekuens Reverse Diffusion DDPM (Langkah demi Langkah: x_T → x_0)", fontsize=11.5, fontweight="bold", pad=18)
    title_ax.axis("off")

    for r in range(n_examples):
        for c, (step_t, imgs_t) in enumerate(ddpm_trajectory):
            ax = fig.add_subplot(traj_sub[r, c])
            ax.imshow(_to_np(imgs_t[r]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {step_t}", fontsize=9.5, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"Sampel #{r+1}", fontsize=9, fontweight="bold")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
