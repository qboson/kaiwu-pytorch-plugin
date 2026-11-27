"""
Graph Convolutional Neural Fingerprint  (Duvenaud et al. 2015).

From the paper (Methods – Molecular representation):
  "The constructed network uses a convolution operator … consisting of distinct
   weight matrices for each possible degree of vertices in the input graph.
   A softmax operation along the node features is then performed for each node
   to facilitate information flow between the atomic features of the molecule.
   Finally, a global pooling operation is applied to the updated hidden node
   features to generate a fixed-length vector as the network output.
   We FIX the weights of the graph neural network to generate a direct
   non-adaptive mapping between the molecular structure and the fingerprint."

Implementation notes:
  • All parameters are initialised with large random weights (N(0,1)) and frozen
    (requires_grad=False), matching the paper: "initialized with large random
    weights without training".
  • Each layer contributes a pooled vector; these are concatenated and projected to fp_dim.
  • The same fixed model is shared between CRBM training and the generation pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from qc_mol_design.config import CFG


class DuvenaudConvLayer(nn.Module):
    """
    Single Duvenaud-style graph convolution layer.

    For each node i:
      aggregated_i = H_i + Σ_{j ∈ N(i)} H_j
      H'_i = ReLU( W_{deg(i)} · aggregated_i + b )

    Then a per-node softmax over the feature dimension accumulates into a
    molecule-level fingerprint contribution via sum pooling.
    """

    def __init__(self, in_dim: int, out_dim: int, max_degree: int) -> None:
        super().__init__()
        self.max_degree = max_degree
        # One weight matrix per degree 0 … max_degree
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim), requires_grad=False)
            for _ in range(max_degree + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
        self._init_weights()

    def _init_weights(self) -> None:
        # Paper: "initialized with large random weights without training"
        # Use standard normal N(0,1) — much larger variance than Xavier
        # (Xavier std ≈ sqrt(2/(in+out)) ≈ 0.1, which is "small").
        for w in self.weights:
            nn.init.normal_(w.data, mean=0.0, std=1.0)

    def forward(
        self,
        H:   torch.Tensor,   # (B, N, in_dim)
        adj: torch.Tensor,   # (B, N, N)
        deg: torch.Tensor,   # (B, N)  — integer degrees
        node_mask: torch.Tensor | None = None,  # (B, N), real atoms
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          H_new  (B, N, out_dim)  — updated node embeddings
          pooled (B, out_dim)     — molecule fingerprint contribution
        """
        out_dim = self.weights[0].shape[1]
        B, N, _ = H.shape
        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=H.device)

        # Neighbourhood aggregation:  H_agg[b, i] = H[b, i] + Σ_j adj[b,i,j] H[b,j]
        H_agg = H + torch.bmm(adj, H)                 # (B, N, in_dim)

        # Apply degree-specific weight matrices
        H_new = torch.zeros(B, N, out_dim, device=H.device, dtype=H.dtype)
        for d in range(self.max_degree + 1):
            # Boolean mask: which nodes have degree d
            mask = ((deg == d) & node_mask).float().unsqueeze(-1)
            W    = self.weights[min(d, self.max_degree)]  # (in_dim, out_dim)
            H_new = H_new + mask * (H_agg @ W)

        H_new = F.relu(H_new + self.bias)              # (B, N, out_dim)

        # Sum only real atoms; padded nodes otherwise change the fingerprint.
        soft   = F.softmax(H_new, dim=2)               # (B, N, out_dim)
        pooled = (soft * node_mask.unsqueeze(-1)).sum(dim=1)

        return H_new, pooled


class NeuralFingerprint(nn.Module):
    """
    Multi-layer neural molecular fingerprint with FIXED (frozen) weights.

    Architecture:
      atom_features → [DuvenaudConvLayer × n_layers] → concat pooled vectors
      → Linear (no bias) → fp_dim fingerprint

    The output is a real-valued vector of length CFG.fp_dim that encodes the
    molecular graph structure via a random non-adaptive projection.
    """

    def __init__(self) -> None:
        super().__init__()
        in_dim   = CFG.atom_feat_dim
        hidden   = CFG.gcn_hidden
        fp_dim   = CFG.fp_dim
        n_layers = CFG.gcn_layers
        max_deg  = CFG.max_degree

        layers, current_dim = [], in_dim
        for _ in range(n_layers):
            layers.append(DuvenaudConvLayer(current_dim, hidden, max_deg))
            current_dim = hidden
        self.layers = nn.ModuleList(layers)

        # Project concatenated per-layer pooled vectors → fp_dim
        # Paper: "initialized with large random weights without training"
        self.proj = nn.Parameter(
            torch.empty(hidden * n_layers, fp_dim), requires_grad=False
        )
        nn.init.normal_(self.proj.data, mean=0.0, std=1.0)

        # Freeze all parameters (enforced by requires_grad=False above)
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def forward(
        self,
        X:   torch.Tensor,   # (B, N, atom_dim)
        adj: torch.Tensor,   # (B, N, N)
        deg: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:       # (B, fp_dim)
        H = X
        node_mask = X.abs().sum(dim=-1) > 0
        pooled_list = []
        for layer in self.layers:
            H, pooled = layer(H, adj, deg, node_mask=node_mask)
            pooled_list.append(pooled)

        concat = torch.cat(pooled_list, dim=-1)   # (B, hidden * n_layers)
        fp     = concat @ self.proj               # (B, fp_dim)
        return fp

    @torch.no_grad()
    def encode_numpy(
        self,
        X:   np.ndarray,   # (N, atom_dim)
        adj: np.ndarray,   # (N, N)
        deg: np.ndarray,   # (N,)
    ) -> np.ndarray:       # (fp_dim,)
        """Convenience: encode a single molecule → numpy fingerprint."""
        self.eval()
        dev = self.device
        X_t   = torch.FloatTensor(X[None]).to(dev)
        adj_t = torch.FloatTensor(adj[None]).to(dev)
        deg_t = torch.LongTensor(deg[None]).to(dev)
        return self.forward(X_t, adj_t, deg_t).cpu().numpy()[0]


def build_fingerprint_model() -> NeuralFingerprint:
    """Instantiate a frozen neural fingerprint model on the configured device."""
    model = NeuralFingerprint().to(CFG.device)
    model.eval()
    return model
