import os
import random
import numpy as np
import torch


class Config:
    # Reproduktibilitas
    SEED = 42

    # Dataset & DataLoader
    DATASET_NAME = "FashionMNIST"
    DATA_DIR = "./dataset_cache"
    IMG_CHANNELS = 1
    IMG_SIZE = 28
    INPUT_DIM = 28 * 28
    NUM_CLASSES = 10
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10
    BATCH_SIZE = 128
    NUM_WORKERS = 2

    # Nama kelas Fashion-MNIST
    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Dimensi Laten
    LATENT_DIM_AE = 64
    LATENT_DIM_SPARSE = 64
    LATENT_DIM_VAE = 64
    LATENT_DIM_GAN = 100
    LATENT_DIM_DCGAN = 100

    # Hyperparameter Pelatihan (Default penuh untuk GPU / Colab)
    EPOCHS_MLP = 25
    EPOCHS_FROZEN_DECODER = 25
    EPOCHS_AE = 30
    EPOCHS_SPARSE_AE = 30
    EPOCHS_VAE = 35
    EPOCHS_GAN = 50
    EPOCHS_DCGAN = 40
    EPOCHS_DDPM = 40

    LR_MLP = 1e-3
    LR_AE = 1e-3
    LR_SPARSE_AE = 1e-3
    SPARSE_LAMBDA = 2e-3  # Penalti L1 sparsity: lambda * mean(|h|)
    LR_VAE = 1e-3
    VAE_BETA = 1.0
    LR_GAN = 2e-4
    LR_DCGAN = 2e-4
    LR_DDPM = 2e-4

    # Konfigurasi DDPM
    DDPM_TIMESTEPS = 400
    DDPM_BETA_START = 1e-4
    DDPM_BETA_END = 0.02

    # Konfigurasi Evaluasi
    FID_NUM_SAMPLES = 2000
    LATENCY_BATCH_SIZE = 64
    LATENCY_WARMUP = 5
    LATENCY_RUNS = 20
    DDPM_LATENCY_RUNS = 3

    # Direktori Luaran
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"
    FIGURES_DIR = "./figures"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = Config.SEED) -> None:
    """Mengunci semua random seed untuk menjamin reproduktibilitas eksperimen."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    """Membuat direktori luaran bila belum tersedia."""
    for path in [Config.DATA_DIR, Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.FIGURES_DIR]:
        os.makedirs(path, exist_ok=True)
