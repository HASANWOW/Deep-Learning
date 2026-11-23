"""Eksperimen Bagian 2.1: Pembuktian Empiris 5 Tahap Batasan MLP Classifier untuk Image Generation."""

import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, set_seed
from data.dataset import pick_diverse_test_samples
from models.mlp_classifier import MLPClassifier, FrozenLatentDecoder
from models.autoencoders import VanillaAutoencoder
from utils.metrics import evaluate_reconstruction_loader
from utils.visualization import plot_figure3_mlp_limitations


def train_mlp_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_MLP,
) -> tuple:
    """Tahap 1: Melatih MLP Classifier (784->256->64->10) hingga test accuracy >= 88%."""
    set_seed(Config.SEED)
    model = MLPClassifier(
        input_dim=Config.INPUT_DIM,
        hidden_dim=256,
        latent_dim=Config.LATENT_DIM_AE,
        num_classes=Config.NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR_MLP)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = _eval_accuracy(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  [MLP Classifier] Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    test_acc = _eval_accuracy(model, test_loader, device)
    print(f"  [MLP Classifier] Final Test Accuracy: {test_acc*100:.2f}% (Target >= 88.00%)")
    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "mlp_classifier.pt"))
    return model, test_acc, history


@torch.no_grad()
def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def train_frozen_latent_decoder(
    mlp_model: MLPClassifier,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = Config.EPOCHS_FROZEN_DECODER,
) -> FrozenLatentDecoder:
    """Tahap 4: Membekukan seluruh bobot MLP Classifier lalu melatih Decoder MLP terpisah (64->256->784)."""
    set_seed(Config.SEED)
    mlp_model.eval()
    for p in mlp_model.parameters():
        p.requires_grad = False

    decoder = FrozenLatentDecoder(
        latent_dim=Config.LATENT_DIM_AE,
        hidden_dim=256,
        output_dim=Config.INPUT_DIM,
    ).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=Config.LR_AE)

    for epoch in range(1, epochs + 1):
        decoder.train()
        running_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z_mlp = mlp_model.extract_latent(x)
            optimizer.zero_grad()
            x_hat = decoder(z_mlp)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  [Frozen Latent Decoder] Epoch {epoch:02d}/{epochs} | Train MSE: {running_loss / len(train_loader.dataset):.5f}")

    torch.save(decoder.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "mlp_frozen_decoder.pt"))
    return decoder


def run_mlp_limitation_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vanilla_ae: VanillaAutoencoder,
    device: torch.device,
    epochs_mlp: int = Config.EPOCHS_MLP,
    epochs_dec: int = Config.EPOCHS_FROZEN_DECODER,
) -> dict:
    """Menjalankan keseluruhan Tahap 1 s.d. Tahap 5 dan menyimpan Figur 3 serta laporan numerik."""
    print("\n" + "=" * 70)
    print("EKSPERIMEN BAGIAN 2: PEMBUKTIAN BATASAN MLP CLASSIFIER UNTUK GENERASI CITRA")
    print("=" * 70)

    t0 = time.perf_counter()
    mlp_model, test_acc, _ = train_mlp_classifier(train_loader, val_loader, test_loader, device, epochs=epochs_mlp)
    orth_errors = mlp_model.weight_orthogonality_error()
    frozen_decoder = train_frozen_latent_decoder(mlp_model, train_loader, device, epochs=epochs_dec)

    mlp_model.eval()
    frozen_decoder.eval()
    vanilla_ae.eval()

    metrics_frozen = evaluate_reconstruction_loader(
        lambda x: frozen_decoder(mlp_model.extract_latent(x)),
        test_loader,
        device,
    )
    metrics_ae = evaluate_reconstruction_loader(
        lambda x: vanilla_ae(x)[0],
        test_loader,
        device,
    )

    print(
        f"  [Komparasi Tahap 4] Frozen Decoder (z_MLP): MSE={metrics_frozen['test_mse']:.5f}, "
        f"PSNR={metrics_frozen['test_psnr_db']:.2f} dB, SSIM={metrics_frozen['test_ssim']:.4f}"
    )
    print(
        f"  [Komparasi Tahap 4] Vanilla AE     (z_AE) : MSE={metrics_ae['test_mse']:.5f}, "
        f"PSNR={metrics_ae['test_psnr_db']:.2f} dB, SSIM={metrics_ae['test_ssim']:.4f}"
    )

    all_z = []
    with torch.no_grad():
        for x, _ in test_loader:
            z = mlp_model.extract_latent(x.to(device))
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    mu_z = all_z.mean(dim=0).to(device)
    sigma_z = all_z.std(dim=0).to(device)
    dead_relu_ratio = float((all_z == 0).float().mean().item())

    sample_imgs, sample_lbls = pick_diverse_test_samples(test_loader, num_samples=6)
    sample_imgs_dev = sample_imgs.to(device)
    sample_lbls_dev = sample_lbls.to(device)
    class_names = [Config.CLASS_NAMES[int(l.item())] for l in sample_lbls]

    with torch.no_grad():
        y_onehot = F.one_hot(sample_lbls_dev, num_classes=Config.NUM_CLASSES).float()
        topdown_imgs = mlp_model.top_down_weight_transpose(y_onehot)

        z_sample = mlp_model.extract_latent(sample_imgs_dev)
        latent_transpose_imgs = mlp_model.latent_transpose_projection(z_sample)

        frozen_dec_imgs = frozen_decoder(z_sample)
        vanilla_ae_imgs, _ = vanilla_ae(sample_imgs_dev)

        set_seed(Config.SEED)
        z_rand = mu_z.unsqueeze(0) + sigma_z.unsqueeze(0) * torch.randn(sample_imgs.size(0), Config.LATENT_DIM_AE, device=device)
        prior_sample_imgs = frozen_decoder(z_rand)

    fig3_path = os.path.join(Config.FIGURES_DIR, "figur3_mlp_limitations.png")
    plot_figure3_mlp_limitations(
        originals=sample_imgs,
        class_labels=class_names,
        topdown_imgs=topdown_imgs,
        latent_transpose_imgs=latent_transpose_imgs,
        frozen_dec_imgs=frozen_dec_imgs,
        vanilla_ae_imgs=vanilla_ae_imgs,
        prior_sample_imgs=prior_sample_imgs,
        metrics_frozen=metrics_frozen,
        metrics_ae=metrics_ae,
        save_path=fig3_path,
    )
    print(f"  [Figur 3] Disimpan ke: {fig3_path}")

    mlp_results = {
        "test_accuracy": test_acc,
        "dead_relu_fraction_in_z_mlp": dead_relu_ratio,
        "weight_orthogonality_errors": orth_errors,
        "frozen_latent_decoder_metrics": metrics_frozen,
        "vanilla_ae_baseline_metrics": metrics_ae,
        "z_mlp_mean_norm": float(torch.norm(mu_z).item()),
        "z_mlp_std_mean": float(sigma_z.mean().item()),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    with open(os.path.join(Config.RESULTS_DIR, "mlp_experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(mlp_results, f, indent=2)

    return mlp_results
