# -*- coding: utf-8 -*-
# Copyright (C) 2022-2026 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""Ising portfolio solvers (local_sa/sa/cim) and RBM scenario covariance enhancer."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from kaiwu.cim import CIMOptimizer
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer, normalize_task_mode

from .qubo import PortfolioQUBO, decode_ising_spins, evaluate_qubo_objective


class LocalIsingPortfolioSampler:
    """License-free NumPy Metropolis-Hastings Ising annealing sampler with polish.

    Implements the exact ``solve(ising_matrix)`` contract of Kaiwu optimizers,
    minimizing ``E(s) = -s^T J s`` over spins ``s in {-1, +1}^N``.
    """

    def __init__(
        self,
        num_reads: int = 48,
        num_sweeps: int = 60,
        initial_temperature: float = 4.0,
        final_temperature: float = 0.02,
        seed: int = 42,
    ) -> None:
        if num_reads < 1 or num_sweeps < 1:
            raise ValueError("num_reads and num_sweeps must be at least 1.")
        if initial_temperature <= 0.0 or final_temperature <= 0.0:
            raise ValueError("Temperatures must be strictly positive.")
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)
        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def compute_energy(ising_matrix: np.ndarray, spins: np.ndarray) -> np.ndarray:
        """Compute Ising Hamiltonian energies ``E(s) = -s^T J s`` per row."""
        mat = np.asarray(ising_matrix, dtype=np.float64)
        coupling = 0.5 * (mat + mat.T)
        np.fill_diagonal(coupling, 0.0)
        spin_arr = np.atleast_2d(np.asarray(spins, dtype=np.float64))
        return -np.einsum("ri,ij,rj->r", spin_arr, coupling, spin_arr)

    @staticmethod
    def _polish_spins(
        spins: np.ndarray,
        fields: np.ndarray,
        coupling: np.ndarray,
    ) -> None:
        """Apply generic Glauber (1-spin) and Kawasaki (2-spin) Ising polish."""
        num_spins = coupling.shape[0]
        for _ in range(4):
            improved = False
            for idx in range(num_spins):
                delta_e = 4.0 * spins[:, idx] * fields[:, idx]
                accept = delta_e < -1e-12
                if np.any(accept):
                    improved = True
                    old_s = spins[accept, idx].copy()
                    spins[accept, idx] = -old_s
                    fields[accept] += (-2.0 * old_s)[:, None] * coupling[idx]

            # Kawasaki pair-flip explores energy-conserving spin swaps without
            # getting trapped behind high cardinality penalty barriers.
            if num_spins <= 64:
                for i in range(num_spins - 1):
                    for j in range(i + 1, num_spins):
                        delta_pair = (
                            4.0 * spins[:, i] * fields[:, i]
                            + 4.0 * spins[:, j] * fields[:, j]
                            - 8.0 * spins[:, i] * spins[:, j] * coupling[i, j]
                        )
                        accept = delta_pair < -1e-12
                        if np.any(accept):
                            improved = True
                            old_i = spins[accept, i].copy()
                            old_j = spins[accept, j].copy()
                            spins[accept, i] = -old_i
                            spins[accept, j] = -old_j
                            fields[accept] += (-2.0 * old_i)[:, None] * coupling[i]
                            fields[accept] += (-2.0 * old_j)[:, None] * coupling[j]
            if not improved:
                break

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        """Sample low-energy spin rows ``s in {-1, +1}^N`` via annealing + polish."""
        mat = np.asarray(ising_matrix, dtype=np.float64)
        coupling = 0.5 * (mat + mat.T)
        np.fill_diagonal(coupling, 0.0)
        num_spins = coupling.shape[0]

        spins = self._rng.choice([-1.0, 1.0], size=(self.num_reads, num_spins))
        fields = spins @ coupling
        temps = np.geomspace(
            self.initial_temperature, self.final_temperature, self.num_sweeps
        )

        for temp in temps:
            inv_temp = 1.0 / temp
            for idx in self._rng.permutation(num_spins):
                delta_e = 4.0 * spins[:, idx] * fields[:, idx]
                accept = delta_e <= 0.0
                uphill = ~accept
                if np.any(uphill):
                    probs = np.exp(-np.clip(delta_e[uphill] * inv_temp, 0.0, 50.0))
                    accept[uphill] = self._rng.random(int(uphill.sum())) < probs
                if np.any(accept):
                    old_spin = spins[accept, idx].copy()
                    spins[accept, idx] = -old_spin
                    fields[accept] += (-2.0 * old_spin)[:, None] * coupling[idx]

        self._polish_spins(spins, fields, coupling)
        return spins.astype(np.float32)


@dataclass
class PortfolioOptimizerConfig:
    """Configuration for Ising portfolio sampling and RBM scenario enhancement."""

    sampler_type: str = "local_sa"
    local_sa_reads: int = 48
    local_sa_sweeps: int = 60
    sa_alpha: float = 0.99
    sa_initial_temperature: float = 10.0
    sa_cutoff_temperature: float = 0.01
    sa_iterations_per_t: int = 20
    sa_size_limit: int = 48
    cim_task_name: str = "kpp_portfolio_qubo"
    cim_task_mode: Optional[str] = None
    cim_precision: int = 8
    cim_truncated_precision: int = 10
    cim_target_bits: int = 550
    rbm_hidden: int = 8
    rbm_epochs: int = 25
    rbm_lr: float = 0.12
    rbm_blend: float = 0.20
    rbm_samples: int = 64
    seed: int = 42


@dataclass
class PortfolioSolution:
    """Decoded portfolio selection returned by an Ising or exact solver."""

    selection: np.ndarray
    weights: np.ndarray
    qubo_objective: float
    feasible: bool
    num_candidates: int


def build_portfolio_sampler(config: PortfolioOptimizerConfig):
    """Construct the Ising sampler for QUBO optimization and RBM sampling."""
    sampler_type = config.sampler_type.lower()
    if sampler_type == "local_sa":
        return LocalIsingPortfolioSampler(
            num_reads=config.local_sa_reads,
            num_sweeps=config.local_sa_sweeps,
            seed=config.seed,
        )
    if sampler_type == "sa":
        return SimulatedAnnealingOptimizer(
            initial_temperature=config.sa_initial_temperature,
            alpha=config.sa_alpha,
            cutoff_temperature=config.sa_cutoff_temperature,
            iterations_per_t=config.sa_iterations_per_t,
            size_limit=config.sa_size_limit,
        )
    if sampler_type == "cim":
        cim_kwargs: Dict[str, object] = {
            "task_name": config.cim_task_name,
            "wait": True,
        }
        task_mode = normalize_task_mode(config.cim_task_mode)
        if task_mode is not None:
            cim_kwargs["task_mode"] = task_mode
        cim_opt = CIMOptimizer(**cim_kwargs)
        return PrecisionReducer(
            cim_opt,
            precision=config.cim_precision,
            truncated_precision=config.cim_truncated_precision,
            target_bits=config.cim_target_bits,
            only_feasible_solution=False,
        )
    raise ValueError(
        f"Unsupported sampler_type '{config.sampler_type}'. "
        "Expected one of: 'local_sa', 'sa', 'cim'."
    )


def solve_portfolio_ising(qubo: PortfolioQUBO, sampler) -> PortfolioSolution:
    """Solve the portfolio Ising Hamiltonian and decode the best feasible selection."""
    max_abs = float(np.max(np.abs(qubo.ising_matrix)))
    scaled_ising = qubo.ising_matrix / max(max_abs, 1e-12)
    raw_spins = np.atleast_2d(np.asarray(sampler.solve(scaled_ising), dtype=np.float64))
    if raw_spins.shape[0] == 0 or raw_spins.shape[1] != qubo.num_assets + 1:
        raise RuntimeError(
            f"Sampler returned shape {raw_spins.shape}; expected (*, {qubo.num_assets + 1})."
        )

    candidates = decode_ising_spins(raw_spins)
    objectives = evaluate_qubo_objective(qubo, candidates)
    counts = candidates.sum(axis=1)
    feasible_idx = np.flatnonzero(counts == qubo.target_k)

    if len(feasible_idx) > 0:
        best_pos = int(feasible_idx[int(np.argmin(objectives[feasible_idx]))])
        feasible = True
    else:
        best_pos = int(np.argmin(objectives))
        feasible = False

    best_sel = candidates[best_pos].astype(np.int64)
    total_selected = int(best_sel.sum())
    weights = (
        best_sel.astype(np.float64) / float(total_selected)
        if total_selected > 0
        else np.full(qubo.num_assets, 1.0 / float(qubo.num_assets), dtype=np.float64)
    )

    return PortfolioSolution(
        selection=best_sel,
        weights=weights,
        qubo_objective=float(objectives[best_pos]),
        feasible=feasible,
        num_candidates=int(candidates.shape[0]),
    )


class RBMScenarioEnhancer:
    """RBM joint market-state model for covariance denoising and scenario stress tests.

    Fits ``kaiwu.torch_plugin.RestrictedBoltzmannMachine`` strictly on binarized
    in-sample training return regimes via ``rbm.sample(sampler)`` and
    ``rbm.objective(s_pos, s_neg)``, then generates synthetic joint regime
    scenarios to regularize the empirical training covariance matrix.
    """

    def __init__(
        self,
        num_assets: int,
        config: PortfolioOptimizerConfig,
        sampler=None,
    ) -> None:
        self.num_assets = int(num_assets)
        self.config = config
        self.sampler = sampler if sampler is not None else build_portfolio_sampler(config)
        self.rbm: Optional[RestrictedBoltzmannMachine] = None
        self.history_: List[float] = []
        self.scenario_corr_: Optional[np.ndarray] = None

    def sample_scenarios(self) -> np.ndarray:
        """Sample binary joint market-regime scenarios from the fitted RBM."""
        if self.rbm is None:
            raise RuntimeError("RBMScenarioEnhancer must be fit before sampling scenarios.")
        with torch.no_grad():
            return self.rbm.sample(self.sampler)[:, : self.num_assets].cpu().numpy()

    def fit_and_enhance_covariance(
        self,
        train_returns: np.ndarray,
        empirical_cov: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Train the RBM on training return regimes and blend generative covariance."""
        ret_np = np.asarray(train_returns, dtype=np.float64)
        emp_cov = 0.5 * (
            np.asarray(empirical_cov, dtype=np.float64)
            + np.asarray(empirical_cov, dtype=np.float64).T
        )
        if ret_np.ndim != 2 or ret_np.shape[1] != self.num_assets:
            raise ValueError(f"Expected train_returns (*, {self.num_assets}).")

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        medians = np.median(ret_np, axis=0, keepdims=True)
        binary_states = (ret_np > medians).astype(np.float32)
        data_t = torch.tensor(binary_states, dtype=torch.float32)

        rbm = RestrictedBoltzmannMachine(
            num_visible=self.num_assets,
            num_hidden=self.config.rbm_hidden,
            device=torch.device("cpu"),
        )
        self.rbm = rbm
        optimizer = torch.optim.SGD(rbm.parameters(), lr=self.config.rbm_lr, momentum=0.5)
        self.history_ = []

        for _ in range(self.config.rbm_epochs):
            s_pos = rbm.get_hidden(data_t, bernoulli=False)
            s_neg = rbm.sample(self.sampler)
            optimizer.zero_grad()
            obj = rbm.objective(s_pos, s_neg)
            reg = 1e-3 * torch.sum(rbm.quadratic_coef ** 2)
            loss = obj + reg
            loss.backward()
            optimizer.step()
            rbm.clip_parameters((-3.0, 3.0), (-3.0, 3.0))
            self.history_.append(float(obj.item()))

        generated = self.sample_scenarios()

        # Convert generated binary regime samples into a PSD correlation matrix.
        centered = generated - generated.mean(axis=0, keepdims=True)
        gen_std = np.sqrt(np.mean(centered ** 2, axis=0)) + 1e-6
        normed = centered / gen_std[None, :]
        gen_corr = (normed.T @ normed) / max(float(generated.shape[0]), 1.0)
        gen_corr = 0.5 * (gen_corr + gen_corr.T)
        np.fill_diagonal(gen_corr, 1.0)
        self.scenario_corr_ = gen_corr

        vols = np.sqrt(np.maximum(np.diag(emp_cov), 1e-12))
        rbm_cov = np.outer(vols, vols) * gen_corr
        blend = float(np.clip(self.config.rbm_blend, 0.0, 1.0))
        enhanced_cov = (1.0 - blend) * emp_cov + blend * rbm_cov
        enhanced_cov = 0.5 * (enhanced_cov + enhanced_cov.T)

        # Quantify fraction of RBM joint downside scenarios (>=75% assets down simultaneously).
        down_fraction = float(np.mean(generated.mean(axis=1) <= 0.25))
        stats = {
            "rbm_final_objective": self.history_[-1] if self.history_ else 0.0,
            "rbm_generated_scenarios": float(generated.shape[0]),
            "rbm_joint_downside_rate": down_fraction,
        }
        return enhanced_cov, stats
