"""
Conditional Restricted Boltzmann Machine (CRBM).

From the paper (Methods – Energy-based model):
  "We adopt a conditional generative model called conditional restricted
   Boltzmann machine (CRBM) to incorporate molecular property targets as
   binary variables."

Energy function:
  E(y, h | f) = -b^T y  -  (W1 f)^T y  -  c^T h  -  (W2 f)^T h  -  y^T W h

where
  y  ∈ {0,1}^{n_vis}    concatenated one-hot property bins  [QED | LogP | SAS]
  h  ∈ {0,1}^{n_hid}    binary latent variables
  f  ∈ R^{fp_dim}       molecular fingerprint (external conditioning input)
  b  ∈ R^{n_vis}        visible bias
  c  ∈ R^{n_hid}        hidden bias
  W  ∈ R^{n_vis×n_hid}  visible–hidden coupling
  W1 ∈ R^{fp_dim×n_vis} fingerprint → visible effective bias
  W2 ∈ R^{fp_dim×n_hid} fingerprint → hidden effective bias

Effective biases (fingerprint-conditioned):
  b_eff(f) = b + W1^T f   (n_vis,)
  c_eff(f) = c + W2^T f   (n_hid,)

Simplified energy:
  E(y, h | f) = -b_eff(f)^T y  -  c_eff(f)^T h  -  y^T W h

Conditional distributions:
  P(h_j = 1 | y, f) = σ( c_eff(f)_j + [W^T y]_j )
  P(y | h, f) factorises per property group via softmax:
    P(y_g | h, f) = softmax( b_eff_g(f) + W_g h )  for group g ∈ {QED, LogP, SAS}

Free energy (marginal over h):
  F(y, f) = -b_eff(f)^T y  -  Σ_j softplus( c_eff(f)_j + [W^T y]_j )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from qc_mol_design.config import CFG


class CRBM(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        n_vis = CFG.n_vis
        n_hid = CFG.n_hidden
        fp    = CFG.fp_dim

        self.b  = nn.Parameter(torch.zeros(n_vis))
        self.c  = nn.Parameter(torch.zeros(n_hid))
        self.W  = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.W1 = nn.Parameter(torch.randn(fp, n_vis)   * 0.01)  # f → visible
        self.W2 = nn.Parameter(torch.randn(fp, n_hid)  * 0.01)  # f → hidden

    # ──────────────────────────────────────────────────────────────────────
    # Effective biases
    # ──────────────────────────────────────────────────────────────────────

    def effective_biases(
        self, f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        b_eff (B, n_vis),  c_eff (B, n_hid)
        f: (B, fp_dim)
        """
        b_eff = self.b + f @ self.W1   # (B, n_vis)
        c_eff = self.c + f @ self.W2   # (B, n_hid)
        return b_eff, c_eff

    # ──────────────────────────────────────────────────────────────────────
    # Energy and free energy
    # ──────────────────────────────────────────────────────────────────────

    def energy(
        self,
        y: torch.Tensor,   # (B, n_vis)
        h: torch.Tensor,   # (B, n_hid)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B,)
        b_eff, c_eff = self.effective_biases(f)
        return (
            -(y * b_eff).sum(-1)
            -(h * c_eff).sum(-1)
            -(y @ self.W * h).sum(-1)
        )

    def free_energy(
        self,
        y: torch.Tensor,   # (B, n_vis)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B,)
        """
        F(y, f) = -b_eff^T y  -  Σ_j softplus(c_eff_j + [W^T y]_j)

        Used as the objective function in the molecular generation QUBO.
        Lower value → molecule better fits target property.
        """
        b_eff, c_eff = self.effective_biases(f)
        vis_term = -(y * b_eff).sum(-1)
        h_input  = c_eff + y @ self.W              # (B, n_hid)
        hid_term = -F.softplus(h_input).sum(-1)
        return vis_term + hid_term

    # ──────────────────────────────────────────────────────────────────────
    # Conditional distributions
    # ──────────────────────────────────────────────────────────────────────

    def p_h_given_yf(
        self,
        y: torch.Tensor,   # (B, n_vis)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B, n_hid) — P(h_j=1 | y, f)
        _, c_eff = self.effective_biases(f)
        return torch.sigmoid(c_eff + y @ self.W)

    def p_y_given_hf_logits(
        self,
        h: torch.Tensor,   # (B, n_hid)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B, n_vis) — unnormalised logits
        b_eff, _ = self.effective_biases(f)
        return b_eff + h @ self.W.t()

    def sample_y_from_logits(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample y from P(y | h, f).

        For each property group the logits are split and a categorical sample
        is drawn.  The result is a concatenated one-hot vector.
        """
        offsets = CFG.prop_offsets
        counts  = CFG.prop_bin_counts
        parts   = []
        for start, count in zip(offsets, counts):
            group_logits = logits[:, start : start + count]   # (B, count)
            idx = torch.distributions.Categorical(logits=group_logits).sample()
            parts.append(F.one_hot(idx, count).float())
        return torch.cat(parts, dim=-1)

    @staticmethod
    def sample_h(probs: torch.Tensor) -> torch.Tensor:
        """Bernoulli sample from P(h | y, f)."""
        return (torch.rand_like(probs) < probs).float()

    # ──────────────────────────────────────────────────────────────────────
    # Encoding  (latent representation for property prediction)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        y: torch.Tensor,   # (B, n_vis)
        f: torch.Tensor,   # (B, fp_dim)
    ) -> torch.Tensor:     # (B, n_hid)  — mean-field h̄ = E[h | y, f]
        """
        Extract latent representation h̄ = σ(c_eff + W^T y).
        Used as input features for the property prediction head.
        """
        return self.p_h_given_yf(y, f)


    @torch.no_grad()
    def encode_from_f(self, f: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
        """Non-paper mean-field alternative retained for exploratory analyses."""
        b_eff, c_eff = self.effective_biases(f)
        parts = [
            torch.full(
                (f.shape[0], count), 1.0 / count,
                device=f.device, dtype=f.dtype,
            )
            for count in CFG.prop_bin_counts
        ]
        y_mean = torch.cat(parts, dim=-1)
        h_mean = torch.sigmoid(c_eff + y_mean @ self.W)
        for _ in range(max(int(n_iter), 1)):
            logits = b_eff + h_mean @ self.W.t()
            y_parts = []
            for start, count in zip(CFG.prop_offsets, CFG.prop_bin_counts):
                y_parts.append(F.softmax(logits[:, start:start + count], dim=-1))
            y_mean = torch.cat(y_parts, dim=-1)
            h_mean = torch.sigmoid(c_eff + y_mean @ self.W)
        return h_mean

    @torch.no_grad()
    def infer_y_map(
        self,
        f: torch.Tensor,
        combination_chunk_size: int = 100,
    ) -> torch.Tensor:
        """Infer y*=argmax_y p(y|f), then return its concatenated one-hot form.

        Methods p.12 specifies this first step for molecules with unknown
        targets.  Since Z(f) is constant with respect to y, maximizing p(y|f)
        is exactly equivalent to minimizing F(y,f).  All 10^3 categorical
        label combinations are enumerated; chunking limits GPU memory without
        introducing an approximation.
        """
        if f.ndim != 2 or f.shape[1] != CFG.fp_dim:
            raise ValueError(
                f"expected fingerprints with shape (B, {CFG.fp_dim}), got "
                f"{tuple(f.shape)}"
            )
        if combination_chunk_size <= 0:
            raise ValueError("combination_chunk_size must be positive")

        grids = torch.cartesian_prod(*[
            torch.arange(count, device=f.device)
            for count in CFG.prop_bin_counts
        ])
        if grids.ndim == 1:
            grids = grids.unsqueeze(-1)
        labels = torch.zeros(
            grids.shape[0], CFG.n_vis, device=f.device, dtype=f.dtype
        )
        row_indices = torch.arange(grids.shape[0], device=f.device)
        for group, (offset, _count) in enumerate(
            zip(CFG.prop_offsets, CFG.prop_bin_counts)
        ):
            labels[row_indices, offset + grids[:, group]] = 1.0

        b_eff, c_eff = self.effective_biases(f)
        best_energy = torch.full(
            (f.shape[0],), float("inf"), device=f.device, dtype=f.dtype
        )
        best_index = torch.zeros(f.shape[0], dtype=torch.long, device=f.device)
        for start in range(0, labels.shape[0], combination_chunk_size):
            y_chunk = labels[start:start + combination_chunk_size]
            visible = -(b_eff @ y_chunk.t())
            hidden_input = (
                c_eff.unsqueeze(1)
                + (y_chunk @ self.W).unsqueeze(0)
            )
            energies = visible - F.softplus(hidden_input).sum(dim=-1)
            chunk_energy, chunk_index = energies.min(dim=1)
            improved = chunk_energy < best_energy
            best_energy = torch.where(improved, chunk_energy, best_energy)
            best_index = torch.where(
                improved, chunk_index + start, best_index
            )
        return labels.index_select(0, best_index)

    @torch.no_grad()
    def encode_unknown(self, f: torch.Tensor) -> torch.Tensor:
        """Paper inference path: MAP target label followed by p(h|y*,f)."""
        return self.encode(self.infer_y_map(f), f)

    def target_free_energy(
        self,
        prop: str,
        bin_index: int,
        f: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalise the two non-target categorical property groups."""
        props = ("qed", "logp", "sas")
        if prop not in props:
            raise ValueError(f"unknown property: {prop}")
        target_group = props.index(prop)
        target_count = CFG.prop_bin_counts[target_group]
        if not 0 <= int(bin_index) < target_count:
            raise ValueError(f"bin index {bin_index} outside {prop} bins")

        non_target = [i for i in range(len(props)) if i != target_group]
        combos = torch.cartesian_prod(*[
            torch.arange(CFG.prop_bin_counts[i], device=f.device)
            for i in non_target
        ])
        if combos.ndim == 1:
            combos = combos.unsqueeze(-1)
        n_combo = combos.shape[0]
        y = torch.zeros(n_combo, CFG.n_vis, device=f.device, dtype=f.dtype)
        y[:, CFG.prop_offsets[target_group] + int(bin_index)] = 1.0
        for col, group in enumerate(non_target):
            y[
                torch.arange(n_combo, device=f.device),
                CFG.prop_offsets[group] + combos[:, col],
            ] = 1.0

        batch = f.shape[0]
        y_all = y.unsqueeze(0).expand(batch, -1, -1).reshape(-1, CFG.n_vis)
        f_all = f.unsqueeze(1).expand(-1, n_combo, -1).reshape(-1, CFG.fp_dim)
        energies = self.free_energy(y_all, f_all).reshape(batch, n_combo)
        return -torch.logsumexp(-energies, dim=1)

    # ──────────────────────────────────────────────────────────────────────
    # QUBO mapping (for QC-assisted training)
    # ──────────────────────────────────────────────────────────────────────

    def build_neg_phase_qubo(
        self, f_single: torch.Tensor, onehot_penalty: float | None = None,
    ) -> torch.Tensor:
        """
        Build QUBO matrix Q for the joint (y, h) distribution given ONE
        fingerprint f_single (fp_dim,).

        The QUBO is:  min_{x} x^T Q x   where x = [y; h] ∈ {0,1}^{n_vis + n_hid}

        Mapping E → QUBO:
          Diagonal entries (linear terms):
            Q[i,i]          = -b_eff_i       (for y_i)
            Q[n_vis+j, n_vis+j] = -c_eff_j   (for h_j)
          Off-diagonal (coupling terms):
            Q[i, n_vis+j]   = -W[i,j]        (i < n_vis, j < n_hid; upper-tri)

        Returns: Q (n_vis + n_hid, n_vis + n_hid) float32 CPU tensor.
        """
        with torch.no_grad():
            f1 = f_single.to(
                device=self.b.device, dtype=self.b.dtype
            ).unsqueeze(0)                          # (1, fp_dim)
            b_eff  = (self.b + f1 @ self.W1)[0]     # (n_vis,)
            c_eff  = (self.c + f1 @ self.W2)[0]     # (n_hid,)

        n_vis = CFG.n_vis
        n_hid = CFG.n_hidden
        n_tot = n_vis + n_hid
        Q = torch.zeros(n_tot, n_tot)

        # Linear (diagonal) terms
        Q.diagonal()[:n_vis] = -b_eff.cpu()
        Q.diagonal()[n_vis:] = -c_eff.cpu()

        # Bilinear (upper-triangle) coupling terms
        W_cpu = self.W.detach().cpu()               # (n_vis, n_hid)
        for i in range(n_vis):
            for j in range(n_hid):
                Q[i, n_vis + j] = -W_cpu[i, j]


        # rho * (sum(y_g) - 1)^2, omitting the constant term.
        if onehot_penalty is None or onehot_penalty <= 0:
            row_scale = float(Q.abs().sum(dim=1).max().item())
            rho = max(10.0, 2.0 * row_scale + 1.0)
        else:
            rho = float(onehot_penalty)
        for start, count in zip(CFG.prop_offsets, CFG.prop_bin_counts):
            for i in range(start, start + count):
                Q[i, i] -= rho
                for j in range(i + 1, start + count):
                    Q[i, j] += 2.0 * rho


        return Q
