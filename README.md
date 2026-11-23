# Studi Komparatif Empiris Model Generatif (AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM) & Pembuktian Batasan MLP Classifier

Proyek eksperimen mandiri mata kuliah **Deep Learning (D6421) — Sesi 07: Generative Models** (terintegrasi Sesi 06: Autoencoders).

- **Nama:** HASAN
- **NIM:** [2802529203]
- **Kelas:** [LB01]
- **Dataset:** Fashion-MNIST (`28x28` grayscale, 70.000 citra)
- **Lingkungan Eksekusi:** PyTorch + CUDA GPU (Tesla T4, `seed=42`, `batch_size=128`)

---

## 1. Deskripsi Proyek
Repositori ini mengimplementasikan tujuh arsitektur jaringan saraf tiruan secara modular dengan protokol evaluasi terstandarisasi (**80% Train / 10% Validation / 10% Test**, `batch_size=128`, `seed=42`):

1. **MLP Classifier (`784 -> 256 -> 64 -> 10`)** — Eksperimen 5 tahap untuk membuktikan secara matematis dan empiris ketidakmampuan bobot terlatih ($W^T$) dan fitur laten diskriminatif ($z_{\text{MLP}}$) dalam merekonstruksi maupun membangkitkan citra baru.
2. **Vanilla Autoencoder (AE)** — Encoder-decoder deterministik dengan *Mean Squared Error* (MSE) loss.
3. **Sparse Autoencoder (Sparse AE)** — Autoencoder dengan penalti regularisasi $L_1$ pada aktivasi laten ($\lambda \cdot \text{mean}(\vert{}h\vert{})$).
4. **Variational Autoencoder (VAE)** — Model generatif probabilistik berbasis ELBO (BCE + KL Divergence) dengan *reparameterization trick*.
5. **Vanilla GAN** — Arsitektur *fully connected* (MLP) dengan objektif *minimax adversarial loss*.
6. **Deep Convolutional GAN (DCGAN)** — Generator & Discriminator konvolusional penuh (*ConvTranspose2d*, *BatchNorm*, *LeakyReLU*, inisialisasi bobot Radford et al.).
7. **Denoising Diffusion Probabilistic Model (DDPM)** — Arsitektur U-Net `28x28` terkondisi *sinusoidal timestep embedding* dengan objektif prediksi *noise* ($\mathcal{L}_{\text{simple}}$, $T=400$).

---

## 2. Struktur Repositori
```text
dl_sesi07_generative_models/
├── config.py                         # Hyperparameter terpusat, seed=42, dan path luaran
├── run_all.py                        # Skrip utama eksekusi penuh & mode --quick-test
├── colab_runner.ipynb                # Notebook runner untuk eksekusi GPU (Colab / Kaggle)
├── requirements.txt                  # Daftar dependensi Python
├── data/
│   ├── __init__.py
│   └── dataset.py                    # Unduh Fashion-MNIST & split 80/10/10 terstandarisasi
├── models/
│   ├── __init__.py
│   ├── mlp_classifier.py             # MLP Classifier, proyeksi W^T, & FrozenLatentDecoder
│   ├── autoencoders.py               # Vanilla AE, Sparse AE (L1), dan VAE
│   ├── gans.py                       # Vanilla GAN (MLP) dan DCGAN
│   └── ddpm.py                       # Time-Conditioned U-Net 28x28 dan DDPM
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    # Evaluasi Test MSE, PSNR, SSIM, FID (InceptionV3), & Latency
│   └── visualization.py              # Pembangkit 5 figur komparatif beresolusi 300 DPI
├── experiments/
│   ├── __init__.py
│   ├── exp_mlp_limits.py             # Eksperimen pembuktian 5 tahap MLP Classifier & Figur 3
│   ├── train_all.py                  # Pipeline pelatihan terstandarisasi 6 model generatif
│   └── evaluate_and_plot.py          # Evaluasi Test Set, ekspor CSV/JSON, & Figur 1, 2, 4, 5
├── figures/                          # Luaran 5 figur wajib PNG (300 DPI)
└── results/                          # Luaran metrik kuantitatif (CSV & JSON)