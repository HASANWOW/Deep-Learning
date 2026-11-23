
import argparse
from config import Config, set_seed, ensure_dirs
from data.dataset import get_dataloaders
from experiments.train_all import train_all_generative_models
from experiments.exp_mlp_limits import run_mlp_limitation_experiment
from experiments.evaluate_and_plot import run_evaluation_and_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Studi Komparatif Empiris 6 Model Generatif + Pembuktian Batasan MLP Classifier (D6421)."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Mode uji cepat (2 epoch per model, FID 128 sampel, DDPM T=50) untuk verifikasi pipeline sebelum training penuh.",
    )
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size DataLoader (128 atau 256).")
    parser.add_argument("--ddpm-steps", type=int, default=Config.DDPM_TIMESTEPS, help="Jumlah langkah difusi T pada DDPM.")
    parser.add_argument("--fid-samples", type=int, default=Config.FID_NUM_SAMPLES, help="Jumlah sampel untuk perhitungan FID.")
    parser.add_argument("--skip-fid", action="store_true", help="Lewati perhitungan FID Score (misal bila tanpa akses unduh InceptionV3).")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(Config.SEED)
    ensure_dirs()

    print("=" * 70)
    print(f"DEEP LEARNING (D6421) SESI 07 | Device: {Config.DEVICE} | Seed: {Config.SEED}")
    print(f"Dataset: {Config.DATASET_NAME} (80% Train / 10% Val / 10% Test) | Batch Size: {args.batch_size}")
    print("=" * 70)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, seed=Config.SEED)
    print(
        f"Split terverifikasi -> Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}"
    )

    if args.quick_test:
        print("\n[MODE QUICK-TEST AKTIF] Menjalankan 2 epoch per model untuk sanity-check pipeline...")
        epoch_overrides = {"ae": 2, "sparse_ae": 2, "vae": 2, "gan": 2, "dcgan": 2, "ddpm": 2}
        epochs_mlp = 2
        epochs_dec = 2
        ddpm_steps = 50
        fid_samples = 128
    else:
        epoch_overrides = None
        epochs_mlp = Config.EPOCHS_MLP
        epochs_dec = Config.EPOCHS_FROZEN_DECODER
        ddpm_steps = args.ddpm_steps
        fid_samples = args.fid_samples

    models, histories = train_all_generative_models(
        train_loader=train_loader,
        val_loader=val_loader,
        device=Config.DEVICE,
        epoch_overrides=epoch_overrides,
        ddpm_timesteps=ddpm_steps,
    )

    run_mlp_limitation_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vanilla_ae=models["Vanilla AE"],
        device=Config.DEVICE,
        epochs_mlp=epochs_mlp,
        epochs_dec=epochs_dec,
    )

    run_evaluation_and_plots(
        models=models,
        histories=histories,
        test_loader=test_loader,
        device=Config.DEVICE,
        fid_num_samples=fid_samples,
        skip_fid=args.skip_fid,
    )

    print("\nSELESAI!")


if __name__ == "__main__":
    main()
