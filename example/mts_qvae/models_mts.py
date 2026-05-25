from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEBaseline(nn.Module):
    """Baseline VAE matching Zhao-Group/MTS-VAE `MTS/scripts/train.py`.

    Input is flattened one-hot: 70 * 22 = 1540.
    Architecture:
      1540 -> 512 -> (mu, logvar) both 32
      32 -> 512 -> 1540 with sigmoid
    """

    def __init__(
        self,
        input_dim: int = 1540,
        hidden_dim: int = 512,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class MTSQVAEEncoder(nn.Module):
    """Encoder for QVAE: returns logits for each latent variable."""

    def __init__(
        self,
        input_dim: int = 1540,
        hidden_dim: int = 512,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        return self.fc2(h)


class MTSQVAEDecoderLogits(nn.Module):
    """Decoder for QVAE: outputs Bernoulli logits (NOT sigmoid probabilities)."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 512,
        output_dim: int = 1540,
    ) -> None:
        super().__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, zeta: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc3(zeta))
        return self.fc4(h)
