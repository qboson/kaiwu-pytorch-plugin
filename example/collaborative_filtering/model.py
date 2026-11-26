# -*- coding: utf-8 -*-
# Copyright (C) 2022-2025 Beijing QBoson Quantum Technology Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""RBM Collaborative Filtering built on kaiwu.torch_plugin.RestrictedBoltzmannMachine."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from kaiwu.cim import CIMOptimizer
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin._kaiwu_compat import PrecisionReducer, normalize_task_mode


class LocalIsingAnnealingSampler:
    """License-free NumPy Metropolis-Hastings Ising annealing sampler.

    Implements the same ``solve(ising_mat)`` interface as Kaiwu optimizers so
    ``RestrictedBoltzmannMachine.sample(sampler)`` executes the full Ising
    conversion and auxiliary-spin decoding pipeline offline without requiring
    a remote license. Follows KPP's Ising energy convention ``E(s) = -s^T J s``.
    """

    def __init__(
        self,
        num_reads: int = 16,
        num_sweeps: int = 20,
        initial_temperature: float = 5.0,
        final_temperature: float = 1.0,
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
        spin_arr = np.asarray(spins, dtype=np.float64)
        return -np.einsum("ri,ij,rj->r", spin_arr, coupling, spin_arr)

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        """Sample Ising spin configurations ``s in {-1, +1}^N`` via annealing."""
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

        return spins.astype(np.float32)


@dataclass
class RBMCFConfig:
    """Hyperparameters for RBM-based collaborative filtering."""

    num_hidden: int = 24
    epochs: int = 45
    batch_size: int = 32
    learning_rate: float = 0.8
    momentum: float = 0.7
    weight_decay: float = 2e-4
    annealing_weight: float = 0.02
    cd_steps: int = 1
    sampler_type: str = "local_sa"
    local_sa_reads: int = 16
    local_sa_sweeps: int = 20
    sa_alpha: float = 0.99
    sa_initial_temperature: float = 10.0
    sa_cutoff_temperature: float = 0.01
    sa_iterations_per_t: int = 10
    sa_size_limit: int = 32
    cim_task_name: str = "kpp_cf_rbm"
    cim_task_mode: Optional[str] = None
    cim_precision: int = 8
    cim_truncated_precision: int = 10
    cim_target_bits: int = 550
    param_clip: float = 4.0
    rating_min: float = 1.0
    rating_max: float = 5.0
    prior_weight: float = 5.0
    seed: int = 42


def build_sampler(config: RBMCFConfig):
    """Construct the annealing sampler for RBM negative-phase sampling."""
    sampler_type = config.sampler_type.lower()
    if sampler_type == "cd":
        return None
    if sampler_type == "local_sa":
        return LocalIsingAnnealingSampler(
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
        cim_optimizer = CIMOptimizer(**cim_kwargs)
        return PrecisionReducer(
            cim_optimizer,
            precision=config.cim_precision,
            truncated_precision=config.cim_truncated_precision,
            target_bits=config.cim_target_bits,
            only_feasible_solution=False,
        )
    raise ValueError(
        f"Unsupported sampler_type '{config.sampler_type}'. "
        "Expected one of: 'local_sa', 'sa', 'cim', 'cd'."
    )


class RBMCollaborativeFiltering:
    """Restricted Boltzmann Machine model for sparse collaborative filtering."""

    def __init__(
        self,
        num_items: int,
        config: Optional[RBMCFConfig] = None,
        sampler=None,
        device: Optional[torch.device] = None,
    ) -> None:
        if num_items < 2:
            raise ValueError("num_items must be at least 2.")
        self.num_items = int(num_items)
        self.config = config if config is not None else RBMCFConfig()
        if self.config.num_hidden < 1:
            raise ValueError("config.num_hidden must be at least 1.")
        if not 0.0 <= self.config.annealing_weight <= 1.0:
            raise ValueError("config.annealing_weight must be in [0.0, 1.0].")
        if self.config.cd_steps < 1:
            raise ValueError("config.cd_steps must be at least 1.")
        if self.config.rating_max <= self.config.rating_min:
            raise ValueError("config.rating_max must be greater than config.rating_min.")

        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.sampler = sampler if sampler is not None else build_sampler(self.config)
        self.rbm: Optional[RestrictedBoltzmannMachine] = None
        self.item_prior_: Optional[torch.Tensor] = None
        self.history_: List[Dict[str, float]] = []

    def _normalize_ratings(self, ratings: torch.Tensor) -> torch.Tensor:
        """Map raw ratings from [rating_min, rating_max] into [0, 1]."""
        span = self.config.rating_max - self.config.rating_min
        return torch.clamp((ratings - self.config.rating_min) / span, 0.0, 1.0)

    def _denormalize_ratings(self, norm_ratings: torch.Tensor) -> torch.Tensor:
        """Map normalized visible expectations from [0, 1] back to rating scale."""
        span = self.config.rating_max - self.config.rating_min
        return self.config.rating_min + torch.clamp(norm_ratings, 0.0, 1.0) * span

    def _impute_visible(
        self, norm_ratings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Fill unobserved entries with the training-set item prior."""
        if self.item_prior_ is None:
            raise RuntimeError("Model has not been initialized with training priors.")
        return mask * norm_ratings + (1.0 - mask) * self.item_prior_.unsqueeze(0)

    def _masked_energy(
        self,
        visible: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RBM energy restricted to observed visible entries."""
        if self.rbm is None:
            raise RuntimeError("RBM has not been initialized.")
        masked_v = visible * mask
        vis_term = masked_v @ self.rbm.visible_bias
        hid_term = hidden @ self.rbm.hidden_bias
        quad_term = torch.sum((masked_v @ self.rbm.quadratic_coef) * hidden, dim=-1)
        return -(vis_term + hid_term + quad_term)

    def _init_model(self, train_ratings: np.ndarray, train_mask: np.ndarray) -> None:
        """Initialize KPP RBM weights and visible biases from training priors."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        ratings_f = np.asarray(train_ratings, dtype=np.float32)
        mask_b = np.asarray(train_mask, dtype=bool)
        span = self.config.rating_max - self.config.rating_min
        norm_obs = np.clip((ratings_f[mask_b] - self.config.rating_min) / span, 0.0, 1.0)
        global_norm_mean = float(norm_obs.mean())

        item_counts = mask_b.sum(axis=0).astype(np.float32)
        norm_mat = np.where(
            mask_b,
            np.clip((ratings_f - self.config.rating_min) / span, 0.0, 1.0),
            0.0,
        )
        item_prior = (
            norm_mat.sum(axis=0) + self.config.prior_weight * global_norm_mean
        ) / (item_counts + self.config.prior_weight)
        item_prior = np.clip(item_prior, 0.05, 0.95).astype(np.float32)
        self.item_prior_ = torch.tensor(item_prior, dtype=torch.float32, device=self.device)

        init_visible_bias = np.log(item_prior / (1.0 - item_prior)).astype(np.float32)
        init_linear_bias = np.concatenate(
            [init_visible_bias, np.zeros(self.config.num_hidden, dtype=np.float32)]
        )
        init_weights = (
            torch.randn((self.num_items, self.config.num_hidden), device=self.device)
            * 0.05
        )
        self.rbm = RestrictedBoltzmannMachine(
            num_visible=self.num_items,
            num_hidden=self.config.num_hidden,
            quadratic_coef=init_weights,
            linear_bias=torch.tensor(init_linear_bias, dtype=torch.float32, device=self.device),
            device=self.device,
        )

    def _train_step(
        self,
        batch_v: torch.Tensor,
        batch_m: torch.Tensor,
        batch_raw: torch.Tensor,
        s_anneal: Optional[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float, float]:
        """Run one masked CD + optional Ising annealing SGD update step."""
        assert self.rbm is not None
        v_pos = self._impute_visible(batch_v, batch_m)
        s_pos = self.rbm.get_hidden(v_pos, bernoulli=False)
        h_pos = s_pos[:, self.num_items :]

        h_chain = (h_pos > torch.rand_like(h_pos)).float()
        v_recon = v_pos
        v_neg = v_pos
        h_neg = h_pos
        for _ in range(self.config.cd_steps):
            v_recon = self.rbm.get_visible(h_chain, bernoulli=False)[:, : self.num_items]
            v_neg = self._impute_visible(v_recon, batch_m)
            s_neg_cd = self.rbm.get_hidden(v_neg, bernoulli=False)
            h_neg = s_neg_cd[:, self.num_items :]
            h_chain = (h_neg > torch.rand_like(h_neg)).float()

        cd_obj = (
            self._masked_energy(v_pos, h_pos, batch_m).mean()
            - self._masked_energy(v_neg, h_neg, batch_m).mean()
        )
        if s_anneal is not None:
            anneal_obj = self.rbm.objective(s_pos, s_anneal) * batch_m.mean()
            w_sa = self.config.annealing_weight
            energy_obj = (1.0 - w_sa) * cd_obj + w_sa * anneal_obj
        else:
            energy_obj = cd_obj

        l2_reg = 0.5 * self.config.weight_decay * torch.sum(self.rbm.quadratic_coef ** 2)
        loss = energy_obj + l2_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clip_bound = float(self.config.param_clip)
        self.rbm.clip_parameters((-clip_bound, clip_bound), (-clip_bound, clip_bound))

        with torch.no_grad():
            recon_err = (self._denormalize_ratings(v_recon) - batch_raw) * batch_m
            recon_se = float(torch.sum(recon_err ** 2).item())
            obs_count = float(batch_m.sum().item())
        return float(energy_obj.item()), recon_se, obs_count

    def fit(
        self,
        train_ratings: np.ndarray,
        train_mask: np.ndarray,
        val_ratings: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
    ) -> "RBMCollaborativeFiltering":
        """Train the RBM collaborative filtering model on sparse ratings."""
        ratings_np = np.asarray(train_ratings, dtype=np.float32)
        mask_np = np.asarray(train_mask, dtype=bool)
        if ratings_np.shape != mask_np.shape or ratings_np.ndim != 2:
            raise ValueError("train_ratings and train_mask must be matching 2D arrays.")
        if ratings_np.shape[1] != self.num_items:
            raise ValueError(
                f"Expected {self.num_items} items, got {ratings_np.shape[1]}."
            )
        if not np.any(mask_np):
            raise ValueError("train_mask must contain at least one observed rating.")

        self._init_model(ratings_np, mask_np)
        assert self.rbm is not None

        optimizer = torch.optim.SGD(
            self.rbm.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )
        ratings_t = torch.tensor(ratings_np, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask_np.astype(np.float32), dtype=torch.float32, device=self.device)
        norm_ratings_t = self._normalize_ratings(ratings_t) * mask_t

        num_users = ratings_np.shape[0]
        rng = np.random.default_rng(self.config.seed)
        self.history_ = []
        use_annealing = self.sampler is not None and self.config.annealing_weight > 0.0
        has_val = val_ratings is not None and val_mask is not None and bool(np.any(val_mask))

        for epoch in range(1, self.config.epochs + 1):
            perm = rng.permutation(num_users)
            stats = [0.0, 0.0, 0.0, 0]
            s_anneal = self.rbm.sample(self.sampler) if use_annealing else None

            for start in range(0, num_users, self.config.batch_size):
                batch_idx = perm[start : start + self.config.batch_size]
                batch_m = mask_t[batch_idx]
                if float(batch_m.sum().item()) == 0.0:
                    continue
                obj_val, recon_se, obs_count = self._train_step(
                    norm_ratings_t[batch_idx],
                    batch_m,
                    ratings_t[batch_idx],
                    s_anneal,
                    optimizer,
                )
                stats[0] += obj_val
                stats[1] += recon_se
                stats[2] += obs_count
                stats[3] += 1

            record: Dict[str, float] = {
                "epoch": float(epoch),
                "objective": stats[0] / max(stats[3], 1),
                "train_rmse": float(np.sqrt(stats[1] / max(stats[2], 1.0))),
            }
            if has_val:
                val_preds = self.predict_ratings(ratings_np, mask_np)
                val_m = np.asarray(val_mask, dtype=bool)
                val_err = (
                    np.asarray(val_ratings, dtype=np.float64)[val_m] - val_preds[val_m]
                )
                record["val_rmse"] = float(np.sqrt(np.mean(val_err ** 2)))
            self.history_.append(record)

        return self

    def predict_ratings(
        self,
        observed_ratings: np.ndarray,
        observed_mask: np.ndarray,
    ) -> np.ndarray:
        """Predict ratings for all items given each user's observed history."""
        if self.rbm is None or self.item_prior_ is None:
            raise RuntimeError("RBMCollaborativeFiltering must be fit before prediction.")

        ratings_np = np.asarray(observed_ratings, dtype=np.float32)
        mask_np = np.asarray(observed_mask, dtype=bool)
        if ratings_np.shape != mask_np.shape or ratings_np.ndim != 2:
            raise ValueError("observed_ratings and observed_mask must be matching 2D arrays.")
        if ratings_np.shape[1] != self.num_items:
            raise ValueError(
                f"Expected {self.num_items} items, got {ratings_np.shape[1]}."
            )

        with torch.no_grad():
            ratings_t = torch.tensor(ratings_np, dtype=torch.float32, device=self.device)
            mask_t = torch.tensor(
                mask_np.astype(np.float32), dtype=torch.float32, device=self.device
            )
            norm_v = self._normalize_ratings(ratings_t) * mask_t
            v_in = self._impute_visible(norm_v, mask_t)
            h_prob = self.rbm.get_hidden(v_in, bernoulli=False)[:, self.num_items :]
            v_prob = self.rbm.get_visible(h_prob, bernoulli=False)[:, : self.num_items]
            preds = self._denormalize_ratings(v_prob)

        return preds.cpu().numpy().astype(np.float32)

    def recommend_top_k(
        self,
        observed_ratings: np.ndarray,
        observed_mask: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend top-K unseen items for each user."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        preds = self.predict_ratings(observed_ratings, observed_mask)
        mask_b = np.asarray(observed_mask, dtype=bool)
        min_unseen = int(np.min((~mask_b).sum(axis=1)))
        effective_k = min(top_k, min_unseen)
        if effective_k < 1:
            raise ValueError("At least one user has no unseen candidate items left.")

        masked_preds = np.where(mask_b, -np.inf, preds)
        top_items = np.argsort(-masked_preds, axis=1, kind="mergesort")[:, :effective_k]
        top_scores = np.take_along_axis(preds, top_items, axis=1)
        return top_items, top_scores
