"""Exercise QDEQ's Ising forward search with low-precision states."""

import math

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class FiniteMatrixSampler:
    """Return one valid spin row and count calls with finite coefficients."""

    def __init__(self):
        self.calls = 0

    def solve(self, matrix):
        assert np.isfinite(matrix).all()
        self.calls += 1
        return np.ones((1, matrix.shape[0]))


@pytest.mark.parametrize(
    "dtype, residual",
    [
        (torch.float16, 1.0),
        (torch.float16, 32768.0),
        (torch.bfloat16, 1.0),
        (torch.float32, 1.0),
        (torch.float64, 1.0),
    ],
)
def test_ising_search_handles_zero_history_and_large_norms(dtype, residual):
    # This map's residual stays constant, so its history includes a zero
    # residual-change direction even though the solve still needs updates.
    config = QDEQConfig(
        f_thres=3, f_eps=1e-6, stop_mode="abs", num_basis=4,
        coeff_bits=2, coeff_range=0.0001, fd_eps=0.125,
    )
    sampler = FiniteMatrixSampler()
    layer = QDEQ(lambda z, x: z + x, sampler, config).eval()
    values = torch.full((1, 8), residual, dtype=dtype)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        output = layer(values)

    stats = layer.get_last_stats()
    assert sampler.calls == config.f_thres
    assert stats["forward_nstep"] == config.f_thres
    assert stats["forward_residual"] == pytest.approx(residual * math.sqrt(8))
    torch.testing.assert_close(output, values)
