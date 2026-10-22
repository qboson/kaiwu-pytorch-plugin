"""QDEQ must reject sampler output that is not one row per encoded variable."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QDEQ, QDEQConfig


class FixedShapeSampler:
    """Returns spin rows of a fixed, possibly wrong, width."""

    def __init__(self, width_delta, rows=2):
        self.width_delta = width_delta
        self.rows = rows

    def solve(self, ising_mat):
        """Returns ``rows`` spin rows of width ``size + width_delta``."""
        width = ising_mat.shape[0] + self.width_delta
        return np.ones((self.rows, width))


class Contraction(torch.nn.Module):
    """Plain contraction ``z / 2 + x`` with a unique fixed point."""

    def forward(self, z, x):
        """Returns ``z / 2 + x``."""
        return 0.5 * z + x


@pytest.mark.parametrize("width_delta", [-1, 1])
def test_rows_of_the_wrong_width_are_rejected(width_delta):
    deq = QDEQ(
        Contraction(),
        FixedShapeSampler(width_delta),
        QDEQConfig(f_thres=2, num_basis=2, coeff_bits=2),
    ).eval()
    with pytest.raises(RuntimeError, match="sampler must return spin rows of length"):
        deq(torch.ones(2, 4))


def test_an_empty_solution_is_rejected():
    deq = QDEQ(
        Contraction(),
        FixedShapeSampler(0, rows=0),
        QDEQConfig(f_thres=2, num_basis=2, coeff_bits=2),
    ).eval()
    with pytest.raises(RuntimeError, match="sampler must return spin rows of length"):
        deq(torch.ones(2, 4))


def test_a_well_shaped_solution_still_solves():
    deq = QDEQ(
        Contraction(),
        FixedShapeSampler(0),
        QDEQConfig(f_thres=2, num_basis=2, coeff_bits=2),
    ).eval()
    out = deq(torch.ones(2, 4))
    assert out.shape == (2, 4)
    assert deq.get_last_stats()["forward_nstep"] >= 1
