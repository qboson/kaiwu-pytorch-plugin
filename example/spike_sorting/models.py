# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Autoencoder / VAE / QBM-VAE feature extractors for spike sorting.

The baselines follow Ardelean et al. (2023): symmetric fully connected
encoders/decoders with ReLU hidden activations, a ``tanh`` code layer and MSE
reconstruction trained with Adam (lr = 1e-3).

``SpikeQBMVAE`` replaces the Gaussian latent of a vanilla VAE with the
Boltzmann-machine prior implemented by :class:`kaiwu.torch_plugin.QVAE`, which
is the contribution this example reproduces.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine


class MLPEncoder(nn.Module):
    """Fully connected encoder ending in a ``tanh`` code layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, latent_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        layers.append(output_activation if output_activation is not None else nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map waveforms to code values."""
        return self.net(x)


class MLPDecoder(nn.Module):
    """Fully connected decoder mirroring :class:`MLPEncoder`."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        dims = [latent_dim, *reversed(hidden_dims), output_dim]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveforms from latent codes."""
        return self.net(z)


class Autoencoder(nn.Module):
    """Plain autoencoder used as the main feature-extraction baseline."""

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [70, 60, 50, 40, 30, 20, 10]
        self.encoder = MLPEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dims, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, code)``."""
        code = self.encoder(x)
        return self.decoder(code), code


class GaussianVAE(nn.Module):
    """Vanilla VAE with a Gaussian latent, used as the second baseline."""

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 10,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [70, 60, 50, 40, 30, 20, 10]
        self.encoder = MLPEncoder(input_dim, hidden_dims, latent_dim * 2)
        self.decoder = MLPDecoder(latent_dim, hidden_dims, input_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, mu, logvar)``."""
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        recon = self.decoder(mu + eps * std)
        return recon, mu, logvar


class SpikeQBMVAE(QVAE):
    """QVAE with a Boltzmann-machine prior on the latent spike codes.

    The encoder/decoder follow the same MLP layout as the baselines; the
    latent vector is split into the visible and hidden halves of a
    :class:`RestrictedBoltzmannMachine` prior. Sampling uses simulated
    annealing by default so the example runs on a laptop without quantum
    hardware.
    """

    def _create_encoder(self) -> MLPEncoder:
        return MLPEncoder(
            self._input_dimension,
            list(self.config.encoder_hidden_nodes),
            self._latent_dimensions,
            output_activation=nn.Identity(),
        )

    def _create_decoder(self) -> MLPDecoder:
        return MLPDecoder(
            self._latent_dimensions,
            list(self.config.decoder_hidden_nodes),
            self._input_dimension,
        )

    def _create_bm(self) -> RestrictedBoltzmannMachine:
        n_visible = self._latent_dimensions // 2
        n_hidden = self._latent_dimensions - n_visible
        return RestrictedBoltzmannMachine(num_visible=n_visible, num_hidden=n_hidden)

    def _create_sampler(self, sampler_type: str | None = None):
        sampler_type = sampler_type or getattr(self.config, "sampler_type", "sa")
        if sampler_type == "sa":
            return SimulatedAnnealingOptimizer(alpha=0.95)
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


def train_autoencoder(
    model: Autoencoder,
    spikes: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Autoencoder:
    """Train a plain autoencoder with MSE reconstruction."""
    torch.manual_seed(seed)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(spikes), batch_size=batch_size, shuffle=True
    )
    for _ in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def train_gaussian_vae(
    model: GaussianVAE,
    spikes: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> GaussianVAE:
    """Train a Gaussian VAE with MSE reconstruction and a KL term."""
    torch.manual_seed(seed)
    model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(spikes), batch_size=batch_size, shuffle=True
    )
    for _ in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def train_qbm_vae(
    model: SpikeQBMVAE,
    spikes: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    bm_lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> SpikeQBMVAE:
    """Train a QBM-VAE with the two-phase update used by the plugin.

    Phase 1 optimises the reconstruction + KL objective (encoder, decoder and
    the BM terms that appear in the cross-entropy). Phase 2 detaches the code
    and trains the Boltzmann-machine prior with its own objective.
    """
    torch.manual_seed(seed)
    model.to(device)
    gen_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("bm.")
    ]
    gen_optimiser = torch.optim.Adam(gen_params, lr=lr)
    bm_optimiser = torch.optim.Adam(
        [p for name, p in model.named_parameters() if name.startswith("bm.")], lr=bm_lr
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(spikes), batch_size=batch_size, shuffle=True
    )
    for _ in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            recon, posterior, q, _ = model(batch)
            loss = model.loss(batch, recon, posterior)
            gen_optimiser.zero_grad()
            bm_optimiser.zero_grad()
            loss.backward()
            gen_optimiser.step()

            bm_loss = model.bm_loss(q.detach(), getattr(model.config, "weight_decay", 0.0))
            bm_loss.backward()
            bm_optimiser.step()
    return model


@torch.no_grad()
def extract_features(
    model: nn.Module,
    spikes: torch.Tensor,
    model_type: str,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> "torch.Tensor":
    """Return latent codes for every waveform.

    Args:
        model: trained ``Autoencoder``, ``GaussianVAE`` or ``SpikeQBMVAE``.
        spikes: ``(n_spikes, 79)`` waveforms.
        model_type: ``"ae"``, ``"vae"`` or ``"qbmvae"``.
        device: torch device.
        batch_size: inference batch size.

    Returns:
        CPU tensor of shape ``(n_spikes, latent_dim)``.
    """
    model.eval()
    model.to(device)
    features = []
    for start in range(0, len(spikes), batch_size):
        batch = spikes[start:start + batch_size].to(device)
        if model_type == "ae":
            code = model.encoder(batch)
        elif model_type == "vae":
            mu, _ = model.encoder(batch).chunk(2, dim=-1)
            code = mu
        elif model_type == "qbmvae":
            code = model.encoder(batch)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        features.append(code.cpu())
    return torch.cat(features, dim=0)

