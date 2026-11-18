import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

try:
    # try to import a KPP library if provided in the environment
    import kpp
    _HAS_KPP = True
except Exception:
    kpp = None
    _HAS_KPP = False

from .full_boltzmann_machine import BoltzmannMachine


class QBMModel(nn.Module):
    """A lightweight QBM adapter that tries to call a KPP transform
    (if available) then uses a BoltzmannMachine + Gibbs sampling to
    produce an energy-based adjustment of reward `r`.

    The class implements a drop-in `forward(embedding, reward)` compatible
    with the EBRM `model` expected interface: returns a (B,1) tensor of scores.
    """

    def __init__(
        self,
        embedding_size: int = 512,
        num_nodes: int = 64,
        num_visible: int = 16,
        device=None,
        energy_scale: float = 1.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_nodes = num_nodes
        self.num_visible = num_visible
        self.device = device if device is not None else torch.device('cpu')
        self.energy_scale = energy_scale

        # small mapper from embedding -> visible bias
        self.embedding_to_visible_bias = nn.Linear(self.embedding_size, self.num_visible)

        # optional small quadratic initialisation for BM
        quadratic = torch.randn((self.num_nodes, self.num_nodes), device=self.device) * 0.01
        linear_bias = torch.zeros(self.num_nodes, device=self.device)

        # create BM with parameters; BoltzmannMachine will register params
        self.bm = BoltzmannMachine(num_nodes=self.num_nodes, quadratic_coef=quadratic, linear_bias=linear_bias, device=self.device)

        # small projection used when KPP is not available
        self._pca_mean = None
        self._pca_components = None

    def _kpp_transform(self, embedding: torch.Tensor) -> torch.Tensor:
        # If external KPP lib available, call it to obtain a compact representation
        if _HAS_KPP and kpp is not None:
            try:
                reps = kpp.transform(embedding.detach().cpu().numpy())
                return torch.tensor(reps, dtype=embedding.dtype, device=embedding.device)
            except Exception:
                pass

        # fallback: simple linear projection (PCA-like) using weights
        # compute on-the-fly mean/components if not set
        if self._pca_components is None:
            # fit a tiny PCA using random weights seeded from embedding size
            rng = np.random.RandomState(0)
            comp = rng.normal(size=(self.embedding_size, min(self.num_visible, self.embedding_size)))
            self._pca_components = torch.tensor(comp.astype(np.float32), device=embedding.device)
        projected = embedding @ self._pca_components
        return projected

    def forward(self, embedding: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """Compute an adjusted score given `embedding` and `reward`.

        embedding: (B, embedding_size)
        reward: (B,) or (B,1)
        returns: (B,1) score tensor
        """
        if reward.dim() == 2 and reward.size(-1) == 1:
            r = reward.squeeze(-1)
        else:
            r = reward

        # batch size
        B = embedding.size(0)

        # get a low-dim representation from KPP or fallback
        rep = self._kpp_transform(embedding)

        # map embedding (or rep) to visible bias
        # use rep if its last dim matches num_visible
        if rep.size(-1) >= self.num_visible:
            visible_bias = rep[:, : self.num_visible]
        else:
            visible_bias = self.embedding_to_visible_bias(embedding)

        # build linear_bias for BM: visible_bias + zeros for hidden
        linear_bias = torch.zeros(self.num_nodes, device=self.device, dtype=visible_bias.dtype).unsqueeze(0).expand(B, -1).clone()
        linear_bias[:, : self.num_visible] = visible_bias

        # update bm parameters in-place for this forward (keep as params but set data)
        with torch.no_grad():
            # if bm.linear_bias is a Parameter of size num_nodes, broadcast per-batch by creating per-sample copies later
            # set base linear_bias (mean over batch)
            self.bm.linear_bias.data = linear_bias.mean(dim=0)

        # convert reward into a visible initialization (probabilities)
        prob = torch.sigmoid(r.unsqueeze(-1) * 2.0)  # (B,1)
        s_visible = prob.expand(-1, self.num_visible)

        # run a short Gibbs sampler conditioned on visible
        try:
            s_all = self.bm.gibbs_sample(num_steps=20, s_visible=s_visible, num_sample=None)
        except Exception:
            # if sampler fails, fall back to a deterministic approximate state
            s_all = torch.cat([s_visible, torch.zeros((B, self.num_nodes - self.num_visible), device=self.device)], dim=-1)

        # compute energy (Hamiltonian) using bm.forward
        energy = self.bm.forward(s_all)  # (B,)

        # adjusted reward: r - scale * energy
        adjusted = r - self.energy_scale * energy

        return adjusted.unsqueeze(-1)
