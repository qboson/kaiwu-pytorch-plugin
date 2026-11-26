# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-modal binary pattern learning and denoising with DBM + joint Ising sampling.

Trains a 2-hidden-layer ``DeepBoltzmannMachine`` on structured 12-bit multi-modal
patterns using variational mean-field inference for the positive phase and
single-shot joint multi-layer Ising sampling (``dbm.sample(sampler)``) for the
negative phase. Runs offline on CPU in a few seconds with ``--sampler local``
and also supports Kaiwu ``SimulatedAnnealingOptimizer`` (``--sampler sa``) and
``CIMOptimizer`` (``--sampler cim``).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import kaiwu as kw
from kaiwu.cim import CIMOptimizer
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import DeepBoltzmannMachine
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer


class LocalSimulatedAnnealingSampler:
    """License-free NumPy Ising SA sampler maximizing ``s^T J s``."""

    def __init__(
        self,
        num_reads: int = 16,
        num_sweeps: int = 40,
        t_start: float = 6.0,
        t_end: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.t_start = t_start
        self.t_end = t_end
        self.rng = np.random.default_rng(seed)

    def solve(self, ising_mat: np.ndarray) -> np.ndarray:
        """Sample ``num_reads`` annealed spin rows in ``{-1, +1}``."""
        coupling = (ising_mat + ising_mat.T) * 0.5
        np.fill_diagonal(coupling, 0.0)
        num_spins = coupling.shape[0]
        temps = np.geomspace(self.t_start, self.t_end, self.num_sweeps)
        rows = []
        for _ in range(self.num_reads):
            spins = self.rng.choice([-1.0, 1.0], size=num_spins)
            field = coupling @ spins
            for temp in temps:
                for idx in self.rng.permutation(num_spins):
                    delta_e = 4.0 * spins[idx] * field[idx]
                    if delta_e <= 0.0 or self.rng.random() < np.exp(-delta_e / temp):
                        spins[idx] = -spins[idx]
                        field += 2.0 * spins[idx] * coupling[:, idx]
            rows.append(spins.copy())
        return np.stack(rows, axis=0)

    def reset_seed(self, seed: int) -> None:
        """Reset the sampler's NumPy random number generator."""
        self.rng = np.random.default_rng(seed)


def build_multimodal_dataset(
    samples_per_mode: int = 16,
    noise_prob: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a 4-mode 12-bit structured binary dataset and out-of-distribution probes."""
    prototypes = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    ood_patterns = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    gen = torch.Generator().manual_seed(seed)
    chunks = []
    for idx in range(prototypes.shape[0]):
        base = prototypes[idx].unsqueeze(0).expand(samples_per_mode, -1)
        flips = (torch.rand(base.shape, generator=gen) < noise_prob).float()
        chunks.append(torch.abs(base - flips))
    return torch.cat(chunks, dim=0), prototypes, ood_patterns


def build_sampler(sampler_name: str, seed: int, t_end: float = 1.0):
    """Build an Ising sampler: local NumPy SA, Kaiwu SA, or Kaiwu CIM."""
    if sampler_name == "local":
        return LocalSimulatedAnnealingSampler(t_end=t_end, seed=seed)

    user_id = os.getenv("USER_ID")
    sdk_code = os.getenv("SDK_CODE")
    if user_id and sdk_code:
        kw.license.init(user_id, sdk_code)

    if sampler_name == "cim":
        cim = CIMOptimizer(task_name="dbm_example", wait=True)
        return PrecisionReducer(
            cim,
            precision=8,
            truncated_precision=10,
            target_bits=550,
            only_feasible_solution=False,
        )
    return SimulatedAnnealingOptimizer(
        initial_temperature=8.0,
        alpha=0.92,
        cutoff_temperature=0.05,
        iterations_per_t=20,
        size_limit=16,
    )


def evaluate_model(
    model: DeepBoltzmannMachine,
    sampler,
    prototypes: torch.Tensor,
    ood_patterns: torch.Tensor,
) -> dict[str, float]:
    """Measure variational free-energy gap, denoising accuracy, and mode generation."""
    with torch.no_grad():
        vfe_in = model.variational_free_energy(prototypes, num_steps=25).mean().item()
        vfe_ood = model.variational_free_energy(ood_patterns, num_steps=25).mean().item()

        # Denoising from corrupted inputs (2 bits flipped per 12-bit prototype)
        corrupted = prototypes.clone()
        corrupted[:, 0] = 1.0 - corrupted[:, 0]
        corrupted[:, 7] = 1.0 - corrupted[:, 7]
        recon_prob = model.reconstruct(corrupted, num_mf_steps=25, bernoulli=False)
        recon_bits = (recon_prob >= 0.5).float()
        denoise_bit_acc = float((recon_bits == prototypes).float().mean().item())
        denoise_exact_acc = float(
            (recon_bits == prototypes).all(dim=-1).float().mean().item()
        )

    samples = model.sample(sampler)[:, : model.num_visible]
    matched_valid = 0
    for row in samples:
        if (prototypes == row.unsqueeze(0)).all(dim=-1).any():
            matched_valid += 1
    gen_valid_rate = matched_valid / max(int(samples.shape[0]), 1)

    return {
        "vfe_gap": vfe_ood - vfe_in,
        "denoise_bit_acc": denoise_bit_acc,
        "denoise_exact_acc": denoise_exact_acc,
        "gen_valid_rate": gen_valid_rate,
    }


def main() -> None:
    """Train a 2-hidden-layer DBM with joint Ising sampling and report metrics."""
    parser = argparse.ArgumentParser(
        description="Multi-modal pattern learning and denoising with DeepBoltzmannMachine."
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--sampler",
        choices=("local", "sa", "cim"),
        default="local",
        help="Ising sampler backend: local (license-free SA), sa (Kaiwu SA), or cim (Kaiwu CIM).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.perf_counter()

    train_data, prototypes, ood_patterns = build_multimodal_dataset(
        samples_per_mode=16, noise_prob=0.05, seed=args.seed
    )
    train_sampler = build_sampler(args.sampler, seed=args.seed, t_end=1.0)
    eval_sampler = build_sampler(args.sampler, seed=args.seed + 1, t_end=0.08)

    model = DeepBoltzmannMachine(
        num_visible=prototypes.shape[1],
        hidden_dims=(8, 4),
        h_range=(-4.0, 4.0),
        j_range=(-4.0, 4.0),
        device="cpu",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    before = evaluate_model(model, eval_sampler, prototypes, ood_patterns)
    print(
        f"[Init] vfe_gap={before['vfe_gap']:.3f}, "
        f"denoise_bit_acc={before['denoise_bit_acc']:.2%}, "
        f"denoise_exact_acc={before['denoise_exact_acc']:.2%}, "
        f"gen_valid_rate={before['gen_valid_rate']:.2%}"
    )

    for epoch in range(1, args.epochs + 1):
        s_pos = model.mean_field_inference(train_data, num_steps=15, tol=1e-5)
        s_neg = model.sample(train_sampler)
        optimizer.zero_grad()
        loss = model.objective(s_pos, s_neg)
        loss.backward()
        optimizer.step()
        model.clip_parameters()

        if epoch in (1, 20, 40, args.epochs):
            print(f"[Epoch {epoch:02d}/{args.epochs}] objective={loss.item():.4f}")

    after = evaluate_model(model, eval_sampler, prototypes, ood_patterns)
    elapsed = time.perf_counter() - start
    print(
        f"[Final] vfe_gap={after['vfe_gap']:.3f}, "
        f"denoise_bit_acc={after['denoise_bit_acc']:.2%}, "
        f"denoise_exact_acc={after['denoise_exact_acc']:.2%}, "
        f"gen_valid_rate={after['gen_valid_rate']:.2%}, "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
