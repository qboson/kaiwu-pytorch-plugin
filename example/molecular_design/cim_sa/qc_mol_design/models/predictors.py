"""
Property prediction MLP heads.

From the paper (Baseline methods, p.13):
  "the latent vector representations of the conditional energy-based models
   trained with both contrastive divergence (CD) learning and QC-assisted
   generative training are used as inputs to a single-layered neural network
   for predicting the corresponding property targets."

Strict architecture (Methods, Baseline methods):
  CRBM latent h (64-dim) → one fully-connected output

The paper calls this a "single-layered neural network" and does not disclose
an additional hidden width.  The previous 128-unit ReLU layer was therefore
an undocumented reproduction choice.

Three separate heads are trained, one per property, sharing the same CRBM
latent representation as input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict

from qc_mol_design.config import CFG


class PropertyMLP(nn.Module):
    """
    Single fully-connected layer: n_hidden → 1.
    Predicts a single continuous molecular property from the CRBM latent h.
    Paper (Methods): latent variables feed a single-layered neural network.
    """

    def __init__(self, in_dim: int = None) -> None:
        super().__init__()
        d = in_dim or CFG.n_hidden
        self.net = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B,)
        return self.net(h).squeeze(-1)


class MultiPropertyPredictor(nn.Module):
    """
    Three independent PropertyMLP heads predicting QED, LogP, and SAS
    simultaneously from the shared CRBM latent representation h.
    """

    PROPS = ("qed", "logp", "sas")

    def __init__(self, in_dim: int = None) -> None:
        super().__init__()
        self.heads = nn.ModuleDict({
            p: PropertyMLP(in_dim) for p in self.PROPS
        })

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dict: prop → (B,) predictions."""
        return {p: head(h) for p, head in self.heads.items()}

    def predict_single(self, h: torch.Tensor, prop: str) -> torch.Tensor:
        return self.heads[prop](h)


class DirectPropertyMLP(nn.Module):
    """
    Baseline predictor that operates directly on a pre-computed fingerprint
    (ECFP / MACCS / GraphConv neural) instead of the CRBM latent h.
    Used for Table 1 Rule-based and GraphConv rows.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,)
        return self.net(x).squeeze(-1)
