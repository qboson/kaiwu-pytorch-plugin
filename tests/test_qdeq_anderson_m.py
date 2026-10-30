"""QDEQConfig.anderson_m must name a real Anderson mixing window."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class LocalSearchSampler:
    """Deterministic spin sampler that needs no license."""

    def __init__(self, restarts=3, seed=0):
        self.restarts = restarts
        self.seed = seed

    def solve(self, ising_mat):
        """Returns one locally optimal spin row per restart."""
        rng = np.random.default_rng(self.seed)
        coupling = (ising_mat + ising_mat.T) / 2
        np.fill_diagonal(coupling, 0.0)
        rows = []
        for _ in range(self.restarts):
            spins = rng.choice([-1.0, 1.0], size=coupling.shape[0])
            field = coupling @ spins
            while True:
                gain = -4.0 * spins * field
                index = int(np.argmax(gain))
                if gain[index] <= 1e-12:
                    break
                spins[index] = -spins[index]
                field += 2.0 * spins[index] * coupling[:, index]
            rows.append(spins.copy())
        return np.stack(rows)


class Contraction(torch.nn.Module):
    """Plain contraction ``z / 2 + x`` with a unique fixed point."""

    def forward(self, z, x):
        """Returns ``z / 2 + x``."""
        return 0.5 * z + x


def test_single_iterate_window_is_rejected():
    with pytest.raises(ValueError, match="anderson_m must be at least 2"):
        QDEQ(Contraction(), LocalSearchSampler(), QDEQConfig(anderson_m=1))


def test_two_iterate_window_trains_through_the_implicit_backward():
    layer = Contraction()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    deq = QDEQ(layer, LocalSearchSampler(), QDEQConfig(anderson_m=2, f_thres=8))
    deq(x).sum().backward()
    assert deq.get_last_stats()["backward_nstep"] >= 1
    assert x.grad is not None
