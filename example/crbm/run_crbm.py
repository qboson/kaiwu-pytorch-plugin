# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Conditional binary pattern generation and completion with CRBM + Ising SA/CIM.

Trains a ``ConditionalRestrictedBoltzmannMachine`` to generate 8-bit structured
binary patterns conditioned on 4 one-hot class labels. Runs completely offline
on CPU in a few seconds with a built-in NumPy simulated annealing sampler
(``--sampler local``), and also supports Kaiwu SDK's
``SimulatedAnnealingOptimizer`` (``--sampler sa``) and ``CIMOptimizer``
(``--sampler cim``) when a Kaiwu license is initialized.
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
from kaiwu.torch_plugin import ConditionalRestrictedBoltzmannMachine
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer


class LocalSimulatedAnnealingSampler:
    """License-free NumPy Ising SA sampler maximizing ``s^T J s``."""

    def __init__(
        self,
        num_reads: int = 8,
        num_sweeps: int = 35,
        t_start: float = 6.0,
        t_end: float = 0.08,
        seed: int = 42,
    ) -> None:
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.t_start = t_start
        self.t_end = t_end
        self.rng = np.random.default_rng(seed)

    def solve(self, ising_mat: np.ndarray) -> np.ndarray:
        """Sample ``num_reads`` spin vectors in ``{-1, +1}``."""
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


def build_pattern_dataset(
    samples_per_class: int = 16,
    noise_prob: float = 0.05,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a 4-class 8-bit binary pattern dataset with light bit-flip noise."""
    prototypes = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # class 0: left half on
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],  # class 1: right half on
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # class 2: even bits on
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # class 3: odd bits on
        ],
        dtype=torch.float32,
    )
    class_conditions = torch.eye(prototypes.shape[0], dtype=torch.float32)

    gen = torch.Generator().manual_seed(seed)
    visible_chunks = []
    cond_chunks = []
    for idx in range(prototypes.shape[0]):
        base = prototypes[idx].unsqueeze(0).expand(samples_per_class, -1)
        flips = (torch.rand(base.shape, generator=gen) < noise_prob).float()
        noisy = torch.abs(base - flips)
        visible_chunks.append(noisy)
        cond_chunks.append(class_conditions[idx].unsqueeze(0).expand(samples_per_class, -1))

    return (
        torch.cat(visible_chunks, dim=0),
        torch.cat(cond_chunks, dim=0),
        prototypes,
        class_conditions,
    )


def build_sampler(sampler_name: str, seed: int, t_end: float = 1.0):
    """Build an Ising sampler: local NumPy SA, Kaiwu SA, or Kaiwu CIM."""
    if sampler_name == "local":
        return LocalSimulatedAnnealingSampler(t_end=t_end, seed=seed)

    user_id = os.getenv("USER_ID")
    sdk_code = os.getenv("SDK_CODE")
    if user_id and sdk_code:
        kw.license.init(user_id, sdk_code)

    if sampler_name == "cim":
        cim = CIMOptimizer(task_name="crbm_example", wait=True)
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
        iterations_per_t=15,
        size_limit=8,
    )


def evaluate_model(
    model: ConditionalRestrictedBoltzmannMachine,
    sampler,
    prototypes: torch.Tensor,
    class_conditions: torch.Tensor,
) -> dict[str, float]:
    """Evaluate free-energy margin, Ising conditional generation, and completion."""
    num_classes = prototypes.shape[0]
    with torch.no_grad():
        fe_true = model.free_energy(prototypes, class_conditions).mean().item()
        wrong_proto = torch.roll(prototypes, shifts=1, dims=0)
        fe_wrong = model.free_energy(wrong_proto, class_conditions).mean().item()

        # One-step conditional reconstruction from corrupted inputs (25% bits flipped)
        corrupted = prototypes.clone()
        corrupted[:, 0] = 1.0 - corrupted[:, 0]
        corrupted[:, 4] = 1.0 - corrupted[:, 4]
        hidden = model.get_hidden(corrupted, class_conditions, bernoulli=False)[
            :, model.num_visible :
        ]
        recon_prob = model.get_visible(hidden, class_conditions, bernoulli=False)[
            :, : model.num_visible
        ]
        recon_bits = (recon_prob >= 0.5).float()
        recon_bit_acc = float((recon_bits == prototypes).float().mean().item())

    exact_matches = 0
    total_reads = 0
    bit_matches = 0
    total_bits = 0
    for idx in range(num_classes):
        sampled = model.condition_sample(sampler, class_conditions[idx])
        vis = sampled[:, : model.num_visible]
        target = prototypes[idx].unsqueeze(0)
        exact_matches += int((vis == target).all(dim=-1).sum().item())
        total_reads += int(vis.shape[0])
        bit_matches += int((vis == target).sum().item())
        total_bits += int(vis.numel())

    return {
        "free_energy_gap": fe_wrong - fe_true,
        "gen_exact_acc": exact_matches / max(total_reads, 1),
        "gen_bit_acc": bit_matches / max(total_bits, 1),
        "recon_bit_acc": recon_bit_acc,
    }


def main() -> None:
    """Train a CRBM with Ising sampling and report conditional generation metrics."""
    parser = argparse.ArgumentParser(
        description="Conditional binary pattern generation with CRBM."
    )
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.2, help="SGD learning rate.")
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

    visible_data, cond_data, prototypes, class_conditions = build_pattern_dataset(
        samples_per_class=16, noise_prob=0.05, seed=args.seed
    )
    train_sampler = build_sampler(args.sampler, seed=args.seed, t_end=1.0)
    eval_sampler = build_sampler(args.sampler, seed=args.seed + 1, t_end=0.08)

    model = ConditionalRestrictedBoltzmannMachine(
        num_visible=prototypes.shape[1],
        num_condition=class_conditions.shape[1],
        num_hidden=6,
        use_cond_visible=True,
        h_range=(-4.0, 4.0),
        j_range=(-4.0, 4.0),
        device="cpu",
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    before = evaluate_model(model, eval_sampler, prototypes, class_conditions)
    print(
        f"[Init] free_energy_gap={before['free_energy_gap']:.3f}, "
        f"gen_exact_acc={before['gen_exact_acc']:.2%}, "
        f"gen_bit_acc={before['gen_bit_acc']:.2%}, "
        f"recon_bit_acc={before['recon_bit_acc']:.2%}"
    )

    for epoch in range(1, args.epochs + 1):
        s_pos = model.get_hidden(visible_data, cond_data, bernoulli=False)
        s_neg, c_neg = model.condition_sample(
            train_sampler, class_conditions, return_conditions=True
        )
        optimizer.zero_grad()
        loss = model.objective(
            s_pos, s_neg, condition=cond_data, condition_negative=c_neg
        )
        loss.backward()
        optimizer.step()
        model.clip_parameters()

        if epoch in (1, 10, 20, args.epochs):
            print(f"[Epoch {epoch:02d}/{args.epochs}] objective={loss.item():.4f}")

    after = evaluate_model(model, eval_sampler, prototypes, class_conditions)
    elapsed = time.perf_counter() - start
    print(
        f"[Final] free_energy_gap={after['free_energy_gap']:.3f}, "
        f"gen_exact_acc={after['gen_exact_acc']:.2%}, "
        f"gen_bit_acc={after['gen_bit_acc']:.2%}, "
        f"recon_bit_acc={after['recon_bit_acc']:.2%}, "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
