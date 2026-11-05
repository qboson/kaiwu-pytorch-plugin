"""Reject malformed optimizer results before decoding QDEQ coefficients."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class Contraction(torch.nn.Module):
    """A nonzero residual forces QDEQ to call the supplied sampler."""

    def forward(self, z, x):
        """Return a contractive fixed-point update."""
        return z / 2 + x


class FixedSampler:
    """Return a result derived from the requested Ising matrix width."""

    def __init__(self, result):
        self.result = result

    def solve(self, ising_mat):
        """Supply the configured optimizer response."""
        return self.result(ising_mat.shape[0])


@pytest.mark.parametrize("value", [0.0, 0.5, 2.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", [0, -1])
def test_rejects_non_spin_values(value, index):
    """Invalid entries must not silently change the encoded search grid."""
    def result(width):
        spins = np.ones((2, width))
        spins[1, index] = value
        return spins

    deq = QDEQ(Contraction(), FixedSampler(result), QDEQConfig(f_thres=1))
    with pytest.raises(RuntimeError, match="spin"):
        deq(torch.ones(1, 2))


def test_rejects_rank_three_results():
    """A coincidentally correct second dimension is not a matrix of rows."""
    sampler = FixedSampler(lambda width: np.ones((1, width, 1)))
    deq = QDEQ(Contraction(), sampler, QDEQConfig(f_thres=1))
    with pytest.raises(RuntimeError, match="spin"):
        deq(torch.ones(1, 2))


@pytest.mark.parametrize("batched", [False, True])
def test_accepts_valid_spin_vectors_and_batches(batched):
    """Both supported result layouts retain finite forward outputs."""
    def result(width):
        row = np.ones(width)
        row[::2] = -1
        return np.stack([row, -row]) if batched else row

    deq = QDEQ(Contraction(), FixedSampler(result), QDEQConfig(f_thres=1))
    output = deq(torch.ones(1, 2))
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    assert deq.get_last_stats()["forward_nstep"] == 1
