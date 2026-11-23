"""Pipeline pelatihan terstandarisasi untuk keenam model generatif (AE, Sparse AE, VAE, GAN, DCGAN, DDPM)."""

import json
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, set_seed
from models.autoencoders import VanillaAutoencoder, SparseAutoencoder, VAE
from models.gans import (
    VanillaGANGenerator,
    VanillaGANDiscriminator,
    DCGANGenerator,
    DCGANDiscriminator,
)
from models.ddpm import DDPM


def train_vanilla_ae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_AE,
) -> Tuple[VanillaAutoencoder, dict]:
    print("\n[1/6] Melatih Vanilla Autoencoder...")
    set_seed(Config.SEED)
    model = VanillaAutoencoder(Config.INPUT_DIM, 256, Config.LATENT_DIM_AE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR_AE)

    history = {"train_loss": [], "val_loss": [], "epoch_times_s": []}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, _ = model(x)
                val_loss += F.mse_loss(x_hat, x).item() * x.size(0)

        tr_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["epoch_times_s"].append(dt)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | Train MSE: {tr_loss:.5f} | Val MSE: {val_loss:.5f} | {dt:.2f}s")

    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "vanilla_ae.pt"))
    return model, history


def train_sparse_ae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_SPARSE_AE,
    l1_lambda: float = Config.SPARSE_LAMBDA,
) -> Tuple[SparseAutoencoder, dict]:
    print("\n[2/6] Melatih Sparse Autoencoder (Penalti L1)...")
    set_seed(Config.SEED)
    model = SparseAutoencoder(Config.INPUT_DIM, 256, Config.LATENT_DIM_SPARSE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR_SPARSE_AE)

    history = {"train_loss": [], "recon_loss": [], "l1_penalty": [], "val_loss": [], "sparsity_zero_ratio": [], "epoch_times_s": []}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_tot, tr_rec, tr_l1 = 0.0, 0.0, 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, h = model(x)
            total, rec, l1 = SparseAutoencoder.loss_function(x_hat, x, h, l1_lambda=l1_lambda)
            total.backward()
            opt.step()
            bs = x.size(0)
            tr_tot += total.item() * bs
            tr_rec += rec.item() * bs
            tr_l1 += l1.item() * bs
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        model.eval()
        val_rec = 0.0
        zero_frac = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, h = model(x)
                val_rec += F.mse_loss(x_hat, x).item() * x.size(0)
                zero_frac += (h <= 1e-4).float().mean().item() * x.size(0)

        n_tr = len(train_loader.dataset)
        n_va = len(val_loader.dataset)
        history["train_loss"].append(tr_tot / n_tr)
        history["recon_loss"].append(tr_rec / n_tr)
        history["l1_penalty"].append(tr_l1 / n_tr)
        history["val_loss"].append(val_rec / n_va)
        history["sparsity_zero_ratio"].append(zero_frac / n_va)
        history["epoch_times_s"].append(dt)
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs} | Total: {tr_tot/n_tr:.5f} | Recon MSE: {tr_rec/n_tr:.5f} "
                f"| Sparsity(zero): {100*zero_frac/n_va:.1f}% | {dt:.2f}s"
            )

    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "sparse_ae.pt"))
    return model, history


def train_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_VAE,
) -> Tuple[VAE, dict]:
    print("\n[3/6] Melatih Variational Autoencoder (VAE)...")
    set_seed(Config.SEED)
    model = VAE(Config.INPUT_DIM, 256, Config.LATENT_DIM_VAE).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR_VAE)

    history = {"train_loss": [], "recon_loss": [], "kl_loss": [], "val_loss": [], "epoch_times_s": []}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_tot, tr_rec, tr_kl = 0.0, 0.0, 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, mu, logvar, _ = model(x)
            total, rec, kl = VAE.loss_function(x_hat, x, mu, logvar, beta=Config.VAE_BETA)
            total.backward()
            opt.step()
            bs = x.size(0)
            tr_tot += total.item() * bs
            tr_rec += rec.item() * bs
            tr_kl += kl.item() * bs
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        model.eval()
        val_tot = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat, mu, logvar, _ = model(x)
                tot, _, _ = VAE.loss_function(x_hat, x, mu, logvar, beta=Config.VAE_BETA)
                val_tot += tot.item() * x.size(0)

        n_tr = len(train_loader.dataset)
        history["train_loss"].append(tr_tot / n_tr)
        history["recon_loss"].append(tr_rec / n_tr)
        history["kl_loss"].append(tr_kl / n_tr)
        history["val_loss"].append(val_tot / len(val_loader.dataset))
        history["epoch_times_s"].append(dt)
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs} | Total: {tr_tot/n_tr:.2f} | BCE: {tr_rec/n_tr:.2f} "
                f"| KL: {tr_kl/n_tr:.2f} | {dt:.2f}s"
            )

    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "vae.pt"))
    return model, history


def train_vanilla_gan(
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_GAN,
) -> Tuple[VanillaGANGenerator, dict]:
    print("\n[4/6] Melatih Vanilla GAN (MLP)...")
    set_seed(Config.SEED)
    G = VanillaGANGenerator(Config.LATENT_DIM_GAN, Config.INPUT_DIM).to(device)
    D = VanillaGANDiscriminator(Config.INPUT_DIM).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=Config.LR_GAN, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=Config.LR_GAN, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    history = {"g_loss": [], "d_loss": [], "epoch_times_s": []}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        G.train()
        D.train()
        g_sum, d_sum, n_batches = 0.0, 0.0, 0
        for x, _ in train_loader:
            real_imgs = x.to(device) * 2.0 - 1.0
            bs = real_imgs.size(0)
            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            opt_D.zero_grad()
            loss_real = criterion(D(real_imgs), real_lbl)
            z = torch.randn(bs, Config.LATENT_DIM_GAN, device=device)
            fake_imgs = G(z).detach()
            loss_fake = criterion(D(fake_imgs), fake_lbl)
            d_loss = 0.5 * (loss_real + loss_fake)
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            z = torch.randn(bs, Config.LATENT_DIM_GAN, device=device)
            gen_imgs = G(z)
            g_loss = criterion(D(gen_imgs), real_lbl)
            g_loss.backward()
            opt_G.step()

            g_sum += g_loss.item()
            d_sum += d_loss.item()
            n_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        history["g_loss"].append(g_sum / n_batches)
        history["d_loss"].append(d_sum / n_batches)
        history["epoch_times_s"].append(dt)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | D Loss: {d_sum/n_batches:.4f} | G Loss: {g_sum/n_batches:.4f} | {dt:.2f}s")

    torch.save(G.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "vanilla_gan_G.pt"))
    torch.save(D.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "vanilla_gan_D.pt"))
    return G, history


def train_dcgan(
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_DCGAN,
) -> Tuple[DCGANGenerator, dict]:
    print("\n[5/6] Melatih DCGAN (Convolutional)...")
    set_seed(Config.SEED)
    G = DCGANGenerator(Config.LATENT_DIM_DCGAN).to(device)
    D = DCGANDiscriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=Config.LR_DCGAN, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=Config.LR_DCGAN, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    history = {"g_loss": [], "d_loss": [], "epoch_times_s": []}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        G.train()
        D.train()
        g_sum, d_sum, n_batches = 0.0, 0.0, 0
        for x, _ in train_loader:
            real_imgs = x.to(device) * 2.0 - 1.0
            bs = real_imgs.size(0)
            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)

            opt_D.zero_grad()
            loss_real = criterion(D(real_imgs), real_lbl)
            z = torch.randn(bs, Config.LATENT_DIM_DCGAN, device=device)
            fake_imgs = G(z).detach()
            loss_fake = criterion(D(fake_imgs), fake_lbl)
            d_loss = 0.5 * (loss_real + loss_fake)
            d_loss.backward()
            opt_D.step()

            opt_G.zero_grad()
            z = torch.randn(bs, Config.LATENT_DIM_DCGAN, device=device)
            gen_imgs = G(z)
            g_loss = criterion(D(gen_imgs), real_lbl)
            g_loss.backward()
            opt_G.step()

            g_sum += g_loss.item()
            d_sum += d_loss.item()
            n_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        history["g_loss"].append(g_sum / n_batches)
        history["d_loss"].append(d_sum / n_batches)
        history["epoch_times_s"].append(dt)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | D Loss: {d_sum/n_batches:.4f} | G Loss: {g_sum/n_batches:.4f} | {dt:.2f}s")

    torch.save(G.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "dcgan_G.pt"))
    torch.save(D.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "dcgan_D.pt"))
    return G, history


def train_ddpm(
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_DDPM,
    timesteps: int = Config.DDPM_TIMESTEPS,
) -> Tuple[DDPM, dict]:
    print(f"\n[6/6] Melatih DDPM (T={timesteps} timesteps)...")
    set_seed(Config.SEED)
    model = DDPM(
        timesteps=timesteps,
        beta_start=Config.DDPM_BETA_START,
        beta_end=Config.DDPM_BETA_END,
        base_ch=48,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=Config.LR_DDPM)

    history = {"train_loss": [], "epoch_times_s": []}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            loss = model.compute_loss(x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        tr_loss /= len(train_loader.dataset)
        history["train_loss"].append(tr_loss)
        history["epoch_times_s"].append(dt)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | Noise Pred MSE: {tr_loss:.5f} | {dt:.2f}s")

    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "ddpm.pt"))
    return model, history


def train_all_generative_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epoch_overrides: Dict[str, int] = None,
    ddpm_timesteps: int = Config.DDPM_TIMESTEPS,
) -> Tuple[dict, dict]:
    """Melatih keenam model generatif secara berurutan dan menyimpan seluruh riwayat loss."""
    ep = {
        "ae": Config.EPOCHS_AE,
        "sparse_ae": Config.EPOCHS_SPARSE_AE,
        "vae": Config.EPOCHS_VAE,
        "gan": Config.EPOCHS_GAN,
        "dcgan": Config.EPOCHS_DCGAN,
        "ddpm": Config.EPOCHS_DDPM,
    }
    if epoch_overrides:
        ep.update(epoch_overrides)

    vanilla_ae, h_ae = train_vanilla_ae(train_loader, val_loader, device, epochs=ep["ae"])
    sparse_ae, h_sae = train_sparse_ae(train_loader, val_loader, device, epochs=ep["sparse_ae"])
    vae, h_vae = train_vae(train_loader, val_loader, device, epochs=ep["vae"])
    vanilla_gan, h_gan = train_vanilla_gan(train_loader, device, epochs=ep["gan"])
    dcgan, h_dcgan = train_dcgan(train_loader, device, epochs=ep["dcgan"])
    ddpm, h_ddpm = train_ddpm(train_loader, device, epochs=ep["ddpm"], timesteps=ddpm_timesteps)

    models = {
        "Vanilla AE": vanilla_ae,
        "Sparse AE": sparse_ae,
        "VAE": vae,
        "Vanilla GAN": vanilla_gan,
        "DCGAN": dcgan,
        "DDPM": ddpm,
    }
    histories = {
        "Vanilla AE": h_ae,
        "Sparse AE": h_sae,
        "VAE": h_vae,
        "Vanilla GAN": h_gan,
        "DCGAN": h_dcgan,
        "DDPM": h_ddpm,
    }

    with open(os.path.join(Config.RESULTS_DIR, "training_histories.json"), "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    return models, histories
