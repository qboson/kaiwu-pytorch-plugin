"""QDEQConfig.stop_mode must pick the residual norm that stops the solve."""

import numpy as np
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class DriftLayer(torch.nn.Module):
    """``layer(z, x) = z + 1e-3``: a constant absolute residual, no fixed point."""

    def __init__(self, shift=1e-3):
        super().__init__()
        self.shift = shift

    def forward(self, z, x):
        """Returns ``z + shift``; ``x`` is unused."""
        del x
        return z + self.shift


class CountingSampler:
    """Returns one valid spin row and counts how often it is asked."""

    def __init__(self):
        self.calls = 0

    def solve(self, ising_mat):
        """Returns a single all-plus spin row including the auxiliary spin."""
        self.calls += 1
        return np.ones((1, ising_mat.shape[0]))


def _solver(stop_mode):
    sampler = CountingSampler()
    config = QDEQConfig(f_thres=2, f_eps=1e-4, num_basis=1, coeff_bits=2, stop_mode=stop_mode)
    deq = QDEQ(DriftLayer(), sampler, config).eval()
    x = torch.full((2, 4), 1e6, dtype=torch.float64)
    deq(x, torch.full_like(x, 1e6))
    return sampler, deq


def test_abs_mode_keeps_iterating_while_the_absolute_residual_is_large():
    sampler, deq = _solver("abs")
    stats = deq.get_last_stats()

    assert stats["forward_nstep"] == 2
    assert sampler.calls == 2
    assert stats["forward_residual"] >= 1e-4


def test_rel_mode_stops_even_though_the_absolute_residual_is_large():
    sampler, deq = _solver("rel")
    stats = deq.get_last_stats()

    assert stats["forward_nstep"] == 0
    assert sampler.calls == 0
    assert stats["forward_residual"] < 1e-4
