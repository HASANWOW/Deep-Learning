"""Evaluasi kuantitatif lengkap pada Test Set, perhitungan FID & Latency, serta pembangkitan Figur 1, 2, 4, 5."""

import csv
import json
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, set_seed
from data.dataset import pick_diverse_test_samples
from utils.metrics import (
    evaluate_reconstruction_loader,
    measure_inference_latency,
    InceptionFeatureExtractor,
    compute_fid_score,
)
from utils.visualization import (
    plot_figure1_reconstruction,
    plot_figure2_random_sampling,
    plot_figure4_interpolation,
    plot_figure5_losses_and_ddpm_trajectory,
)


def _collect_real_test_images(test_loader: DataLoader, num_samples: int) -> torch.Tensor:
    imgs = []
    collected = 0
    for x, _ in test_loader:
        imgs.append(x)
        collected += x.size(0)
        if collected >= num_samples:
            break
    return torch.cat(imgs, dim=0)[:num_samples]


@torch.no_grad()
def _generate_in_batches(sample_fn, total_samples: int, batch_size: int = 128) -> torch.Tensor:
    out = []
    remaining = total_samples
    while remaining > 0:
        cur = min(batch_size, remaining)
        out.append(sample_fn(cur).cpu())
        remaining -= cur
    return torch.cat(out, dim=0)


def _pick_contrasting_pair(test_loader: DataLoader, class_a: int = 0, class_b: int = 9):
    """Memilih dua sampel uji dari dua kelas kontras (default: T-shirt/top vs Ankle boot)."""
    img_a, img_b = None, None
    for x, y in test_loader:
        for img, lbl in zip(x, y):
            if int(lbl.item()) == class_a and img_a is None:
                img_a = img.unsqueeze(0)
            elif int(lbl.item()) == class_b and img_b is None:
                img_b = img.unsqueeze(0)
            if img_a is not None and img_b is not None:
                return img_a, img_b
    raise RuntimeError("Gagal menemukan pasangan kelas kontras pada Test Set.")


def run_evaluation_and_plots(
    models: Dict[str, torch.nn.Module],
    histories: Dict[str, dict],
    test_loader: DataLoader,
    device: torch.device,
    fid_num_samples: int = Config.FID_NUM_SAMPLES,
    skip_fid: bool = False,
) -> dict:
    print("\n" + "=" * 70)
    print("EVALUASI KUANTITATIF TEST SET & VISUALISASI KOMPARATIF (FIGUR 1, 2, 4, 5)")
    print("=" * 70)

    vanilla_ae = models["Vanilla AE"].eval()
    sparse_ae = models["Sparse AE"].eval()
    vae = models["VAE"].eval()
    vanilla_gan = models["Vanilla GAN"].eval()
    dcgan = models["DCGAN"].eval()
    ddpm = models["DDPM"].eval()

    print("  [1/5] Menghitung Test MSE, PSNR, dan SSIM pada Test Set...")
    recon_ae = evaluate_reconstruction_loader(lambda x: vanilla_ae(x)[0], test_loader, device)
    recon_sae = evaluate_reconstruction_loader(lambda x: sparse_ae(x)[0], test_loader, device)
    recon_vae = evaluate_reconstruction_loader(lambda x: vae.decode(vae.encode(x)[0]), test_loader, device)

    print("  [2/5] Mengukur Inference Latency per batch 64 sampel...")
    bs_lat = Config.LATENCY_BATCH_SIZE
    dummy_img_64 = torch.rand(bs_lat, 1, 28, 28, device=device)

    lat_ae = measure_inference_latency(lambda: vanilla_ae(dummy_img_64)[0], device, Config.LATENCY_WARMUP, Config.LATENCY_RUNS)
    lat_sae = measure_inference_latency(lambda: sparse_ae(dummy_img_64)[0], device, Config.LATENCY_WARMUP, Config.LATENCY_RUNS)
    lat_vae = measure_inference_latency(lambda: vae.sample(bs_lat, device), device, Config.LATENCY_WARMUP, Config.LATENCY_RUNS)
    lat_gan = measure_inference_latency(lambda: vanilla_gan.sample(bs_lat, device), device, Config.LATENCY_WARMUP, Config.LATENCY_RUNS)
    lat_dcgan = measure_inference_latency(lambda: dcgan.sample(bs_lat, device), device, Config.LATENCY_WARMUP, Config.LATENCY_RUNS)
    lat_ddpm = measure_inference_latency(
        lambda: ddpm.sample(bs_lat, device)[0], device, warmup=1, runs=Config.DDPM_LATENCY_RUNS
    )

    fid_scores = {"VAE": None, "Vanilla GAN": None, "DCGAN": None, "DDPM": None}
    if not skip_fid:
        print(f"  [3/5] Menghitung FID Score menggunakan InceptionV3 ({fid_num_samples} sampel real vs sintetis)...")
        set_seed(Config.SEED)
        extractor = InceptionFeatureExtractor(device)
        real_imgs = _collect_real_test_images(test_loader, fid_num_samples)

        fake_vae = _generate_in_batches(lambda n: vae.sample(n, device), fid_num_samples, batch_size=128)
        fid_scores["VAE"] = compute_fid_score(real_imgs, fake_vae, extractor)
        print(f"    -> FID VAE        : {fid_scores['VAE']:.2f}")

        fake_gan = _generate_in_batches(lambda n: vanilla_gan.sample(n, device), fid_num_samples, batch_size=128)
        fid_scores["Vanilla GAN"] = compute_fid_score(real_imgs, fake_gan, extractor)
        print(f"    -> FID Vanilla GAN: {fid_scores['Vanilla GAN']:.2f}")

        fake_dcgan = _generate_in_batches(lambda n: dcgan.sample(n, device), fid_num_samples, batch_size=128)
        fid_scores["DCGAN"] = compute_fid_score(real_imgs, fake_dcgan, extractor)
        print(f"    -> FID DCGAN      : {fid_scores['DCGAN']:.2f}")

        fake_ddpm = _generate_in_batches(lambda n: ddpm.sample(n, device)[0], fid_num_samples, batch_size=64)
        fid_scores["DDPM"] = compute_fid_score(real_imgs, fake_ddpm, extractor)
        print(f"    -> FID DDPM       : {fid_scores['DDPM']:.2f}")
    else:
        print("  [3/5] Perhitungan FID dilewati (--skip-fid aktif).")

    print("  [4/5] Membuat visualisasi komparatif (Figur 1, 2, 4, 5) pada 300 DPI...")

    sample_8, lbls_8 = pick_diverse_test_samples(test_loader, num_samples=8)
    sample_8_dev = sample_8.to(device)
    lbl_names_8 = [Config.CLASS_NAMES[int(l.item())] for l in lbls_8]
    with torch.no_grad():
        r_ae = vanilla_ae(sample_8_dev)[0]
        r_sae = sparse_ae(sample_8_dev)[0]
        r_vae = vae.decode(vae.encode(sample_8_dev)[0])

    plot_figure1_reconstruction(
        sample_8, r_ae, r_sae, r_vae, lbl_names_8,
        os.path.join(Config.FIGURES_DIR, "figur1_reconstruction_grid.png"),
    )

    set_seed(Config.SEED)
    T = ddpm.timesteps
    record_steps = sorted(list({T, int(0.8 * T), int(0.6 * T), int(0.4 * T), int(0.2 * T), int(0.1 * T), 0}), reverse=True)
    ddpm_samples_16, ddpm_traj = ddpm.sample(16, device=device, record_steps=record_steps)

    samples_fig2 = {
        "VAE": vae.sample(16, device),
        "Vanilla GAN": vanilla_gan.sample(16, device),
        "DCGAN": dcgan.sample(16, device),
        "DDPM": ddpm_samples_16,
    }
    plot_figure2_random_sampling(
        samples_fig2,
        os.path.join(Config.FIGURES_DIR, "figur2_random_sampling.png"),
    )

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    img_a, img_b = _pick_contrasting_pair(test_loader, class_a=0, class_b=9)
    img_a_dev, img_b_dev = img_a.to(device), img_b.to(device)

    interp_dict = {}
    with torch.no_grad():
        for name, enc_fn, dec_fn in [
            ("Vanilla AE", lambda x: vanilla_ae.encode(x), lambda z: vanilla_ae.decode(z)),
            ("Sparse AE", lambda x: sparse_ae.encode(x), lambda z: sparse_ae.decode(z)),
            ("VAE", lambda x: vae.encode(x)[0], lambda z: vae.decode(z)),
        ]:
            z_a = enc_fn(img_a_dev)
            z_b = enc_fn(img_b_dev)
            imgs = [dec_fn((1.0 - a) * z_a + a * z_b)[0] for a in alphas]
            interp_dict[name] = torch.stack(imgs, dim=0)

        set_seed(Config.SEED + 7)
        z_gan_a = torch.randn(1, Config.LATENT_DIM_GAN, device=device)
        z_gan_b = torch.randn(1, Config.LATENT_DIM_GAN, device=device)
        interp_dict["Vanilla GAN"] = torch.stack(
            [((vanilla_gan((1.0 - a) * z_gan_a + a * z_gan_b)[0] + 1.0) / 2.0) for a in alphas], dim=0
        )

        z_dc_a = torch.randn(1, Config.LATENT_DIM_DCGAN, device=device)
        z_dc_b = torch.randn(1, Config.LATENT_DIM_DCGAN, device=device)
        interp_dict["DCGAN"] = torch.stack(
            [((dcgan((1.0 - a) * z_dc_a + a * z_dc_b)[0] + 1.0) / 2.0) for a in alphas], dim=0
        )

    plot_figure4_interpolation(
        interp_dict,
        alphas,
        pair_label="AE/VAE: T-shirt → Ankle boot | GAN/DCGAN: z_A → z_B",
        save_path=os.path.join(Config.FIGURES_DIR, "figur4_latent_interpolation.png"),
    )

    plot_figure5_losses_and_ddpm_trajectory(
        histories,
        ddpm_traj,
        os.path.join(Config.FIGURES_DIR, "figur5_losses_and_ddpm_trajectory.png"),
    )

    print("  [5/5] Menyimpan tabel metrik komparatif ke results/metrics_summary.json & results/metrics_table.csv...")
    summary_rows = [
        {
            "model": "Vanilla Autoencoder",
            "latent_dim": str(Config.LATENT_DIM_AE),
            "train_time_per_epoch_s": round(float(np.mean(histories["Vanilla AE"]["epoch_times_s"])), 2),
            "latency_ms_batch64": round(lat_ae, 3),
            "test_mse": round(recon_ae["test_mse"], 5),
            "test_psnr_db": round(recon_ae["test_psnr_db"], 2),
            "test_ssim": round(recon_ae["test_ssim"], 4),
            "fid_score": "N/A (Latent Voids)",
            "training_stability": "Stabil / Konvergen",
        },
        {
            "model": "Sparse Autoencoder",
            "latent_dim": f"{Config.LATENT_DIM_SPARSE} (Sparse L1, zero={100*histories['Sparse AE']['sparsity_zero_ratio'][-1]:.1f}%)",
            "train_time_per_epoch_s": round(float(np.mean(histories["Sparse AE"]["epoch_times_s"])), 2),
            "latency_ms_batch64": round(lat_sae, 3),
            "test_mse": round(recon_sae["test_mse"], 5),
            "test_psnr_db": round(recon_sae["test_psnr_db"], 2),
            "test_ssim": round(recon_sae["test_ssim"], 4),
            "fid_score": "N/A (Latent Voids)",
            "training_stability": "Stabil / Konvergen",
        },
        {
            "model": "Variational Autoencoder (VAE)",
            "latent_dim": str(Config.LATENT_DIM_VAE),
            "train_time_per_epoch_s": round(float(np.mean(histories["VAE"]["epoch_times_s"])), 2),
            "latency_ms_batch64": round(lat_vae, 3),
            "test_mse": round(recon_vae["test_mse"], 5),
            "test_psnr_db": round(recon_vae["test_psnr_db"], 2),
            "test_ssim": round(recon_vae["test_ssim"], 4),
            "fid_score": round(fid_scores["VAE"], 2) if fid_scores["VAE"] is not None else "Skipped",
            "training_stability": "Stabil / Konvergen",
        },
        {
            "model": "Vanilla GAN (MLP)",
            "latent_dim": f"{Config.LATENT_DIM_GAN} (Prior Noise)",
            "train_time_per_epoch_s": round(float(np.mean(histories["Vanilla GAN"]["epoch_times_s"])), 2),
            "latency_ms_batch64": round(lat_gan, 3),
            "test_mse": "N/A (Tanpa Encoder)",
            "test_psnr_db": "N/A",
            "test_ssim": "N/A",
            "fid_score": round(fid_scores["Vanilla GAN"], 2) if fid_scores["Vanilla GAN"] is not None else "Skipped",
            "training_stability": "Rentan Osilasi / Mode Collapse",
        },
        {
            "model": "Deep Convolutional GAN (DCGAN)",
            "latent_dim": f"{Config.LATENT_DIM_DCGAN} (Prior Noise)",
            "train_time_per_epoch_s": round(float(np.mean(histories["DCGAN"]["epoch_times_s"])), 2),
            "latency_ms_batch64": round(lat_dcgan, 3),
            "test_mse": "N/A (Tanpa Encoder)",
            "test_psnr_db": "N/A",
            "test_ssim": "N/A",
            "fid_score": round(fid_scores["DCGAN"], 2) if fid_scores["DCGAN"] is not None else "Skipped",
            "training_stability": "Cukup Stabil (Sensitif Arsitektur)",
        },
        {
            "model": "DDPM (Diffusion)",
            "latent_dim": f"28x28 (T={ddpm.timesteps})",
            "train_time_per_epoch_s": round(float(np.mean(histories["DDPM"]["epoch_times_s"])), 2),
            "latency_ms_batch64": round(lat_ddpm, 3),
            "test_mse": "N/A (Denoising Model)",
            "test_psnr_db": "N/A",
            "test_ssim": "N/A",
            "fid_score": round(fid_scores["DDPM"], 2) if fid_scores["DDPM"] is not None else "Skipped",
            "training_stability": "Sangat Stabil",
        },
    ]

    summary_payload = {
        "device": str(device),
        "dataset": Config.DATASET_NAME,
        "seed": Config.SEED,
        "batch_size": Config.BATCH_SIZE,
        "fid_num_samples": fid_num_samples,
        "models_comparison": summary_rows,
    }

    with open(os.path.join(Config.RESULTS_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    csv_path = os.path.join(Config.RESULTS_DIR, "metrics_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nRingkasan Tabel Metrik Bagian IV:")
    for r in summary_rows:
        print(
            f"  - {r['model']:<32} | Epoch: {r['train_time_per_epoch_s']:>6}s | "
            f"Lat64: {r['latency_ms_batch64']:>8} ms | MSE: {str(r['test_mse']):<10} | "
            f"PSNR: {str(r['test_psnr_db']):<6} | SSIM: {str(r['test_ssim']):<6} | FID: {r['fid_score']}"
        )

    return summary_payload
