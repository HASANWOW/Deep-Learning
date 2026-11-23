"""Pipeline dataset Fashion-MNIST terstandarisasi (80% Train / 10% Val / 10% Test)."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

from config import Config


def get_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    seed: int = Config.SEED,
    data_dir: str = Config.DATA_DIR,
    num_workers: int = Config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Mengunduh Fashion-MNIST (70.000 citra 28x28 grayscale), menggabungkan train+test
    bawaan lalu membagi secara terstandarisasi menjadi:
      - Train Set : 80% (56.000 citra), shuffle=True
      - Val Set   : 10% (7.000 citra),  shuffle=False
      - Test Set  : 10% (7.000 citra),  shuffle=False
    Rentang nilai piksel terstandardisasi pada [0.0, 1.0].
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1], shape (1, 28, 28)
    ])

    raw_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    raw_test = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([raw_train, raw_test])

    total_len = len(full_dataset)
    train_len = int(Config.TRAIN_RATIO * total_len)
    val_len = int(Config.VAL_RATIO * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len], generator=generator)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader


def pick_diverse_test_samples(test_loader: DataLoader, num_samples: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mengambil sampel uji dari kelas yang berbeda-beda agar visualisasi komparatif representatif."""
    selected_imgs = []
    selected_labels = []
    seen_classes = set()

    for x, y in test_loader:
        for img, label in zip(x, y):
            lbl = int(label.item())
            if lbl not in seen_classes:
                seen_classes.add(lbl)
                selected_imgs.append(img)
                selected_labels.append(lbl)
            if len(selected_imgs) >= num_samples:
                return torch.stack(selected_imgs, dim=0), torch.tensor(selected_labels, dtype=torch.long)

    return torch.stack(selected_imgs, dim=0), torch.tensor(selected_labels, dtype=torch.long)
