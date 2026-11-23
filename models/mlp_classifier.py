"""MLP Classifier (784->256->64->10) & Frozen Latent Decoder untuk Eksperimen Bagian 2."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    Arsitektur diskriminatif berjenjang:
      fc1: 784 -> 256 (ReLU)
      fc2: 256 -> 64  (ReLU)  --> ruang fitur laten z_MLP in R^64
      fc3: 64  -> 10  (Logits)
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Mengekstrak fitur laten z_MLP (dim 64) sebelum lapisan klasifikasi akhir."""
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        z_mlp = F.relu(self.fc2(h1))
        return z_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mlp = self.extract_latent(x)
        logits = self.fc3(z_mlp)
        return logits

    @torch.no_grad()
    def top_down_weight_transpose(self, y_onehot: torch.Tensor) -> torch.Tensor:
        """
        Tahap 2: Proyeksi balik langsung dari vektor one-hot kelas y_c in R^10.
        Konvensi PyTorch: bobot W_l berdimensi (out_features, in_features).
        Untuk vektor baris v, operasi matematis W_l^T * v_col setara dengan v_row @ W_l:
          z_hat  = ReLU(W_3^T y_c)
          h1_hat = ReLU(W_2^T z_hat)
          x_hat  = Sigmoid(W_1^T h1_hat)
        """
        W1 = self.fc1.weight  # (256, 784)
        W2 = self.fc2.weight  # (64, 256)
        W3 = self.fc3.weight  # (10, 64)

        z_hat = F.relu(torch.matmul(y_onehot, W3))      # (B, 64)
        h1_hat = F.relu(torch.matmul(z_hat, W2))        # (B, 256)
        x_hat = torch.sigmoid(torch.matmul(h1_hat, W1)) # (B, 784)
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def latent_transpose_projection(self, z_mlp: torch.Tensor) -> torch.Tensor:
        """
        Tahap 3: Proyeksi balik langsung dari fitur laten z_MLP in R^64 ke ruang piksel:
          x_hat = Sigmoid(W_1^T ReLU(W_2^T z_MLP))
        """
        W1 = self.fc1.weight  # (256, 784)
        W2 = self.fc2.weight  # (64, 256)

        h1_hat = F.relu(torch.matmul(z_mlp, W2))        # (B, 256)
        x_hat = torch.sigmoid(torch.matmul(h1_hat, W1)) # (B, 784)
        return x_hat.view(-1, 1, 28, 28)

    def weight_orthogonality_error(self) -> dict:
        """Mengukur penyimpangan ortogonalitas ||W^T W - I||_F sebagai bukti empiris W^T != W^{-1}."""
        errors = {}
        for name, layer in [("W1", self.fc1), ("W2", self.fc2), ("W3", self.fc3)]:
            W = layer.weight.detach()  # (out, in)
            wt_w = torch.matmul(W.t(), W)
            eye_in = torch.eye(wt_w.size(0), device=W.device)
            w_wt = torch.matmul(W, W.t())
            eye_out = torch.eye(w_wt.size(0), device=W.device)
            errors[name] = {
                "shape": list(W.shape),
                "frob_err_WTW_minus_I": float(torch.norm(wt_w - eye_in, p="fro").item()),
                "frob_err_WWT_minus_I": float(torch.norm(w_wt - eye_out, p="fro").item()),
            }
        return errors


class FrozenLatentDecoder(nn.Module):
    """
    Tahap 4: Decoder MLP terpisah (64 -> 256 -> 784) yang dilatih menggunakan MSE Loss
    untuk merekonstruksi citra asli dari z_MLP yang bobot encoder-nya dibekukan.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, 28, 28)
