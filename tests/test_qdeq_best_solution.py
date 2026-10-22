"""The coefficient decoder must keep the best row, not the first one."""

import itertools
import math

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QDEQConfig
from kaiwu.torch_plugin.qdeq import _sample_coefficients


class OrderedRowSampler:
    """Returns every spin row, ordered from worst to best objective."""

    def __init__(self, rows):
        self.rows = rows

    def solve(self, ising_mat):
        """Returns all stored rows."""
        del ising_mat
        return np.stack(self.rows)


def _decode(row, num_basis, coeff_bits, scale):
    weights = 2.0 * scale / (2 ** coeff_bits - 1) * 2.0 ** np.arange(coeff_bits)
    bits = ((row[:-1] * row[-1] + 1) / 2).reshape(num_basis, coeff_bits)
    return -scale + bits @ weights


def test_the_best_returned_row_is_kept():
    rng = np.random.default_rng(0)
    g_mat = torch.tensor(rng.normal(size=(2, 3)))
    residual = torch.tensor(rng.normal(size=3))
    residual_norm = float(residual.norm())
    config = QDEQConfig(num_basis=2, coeff_bits=2, coeff_range=1.5)
    hess = (g_mat @ g_mat.T).numpy()
    grad = (g_mat @ residual).numpy()
    scale = config.coeff_range * residual_norm / math.sqrt(config.num_basis)

    rows = []
    for spins in itertools.product([1.0, -1.0], repeat=config.num_basis * config.coeff_bits + 1):
        row = np.array(spins)
        coeff = _decode(row, config.num_basis, config.coeff_bits, scale)
        objective = 0.5 * coeff @ hess @ coeff + coeff @ grad
        rows.append((objective, row))
    rows.sort(key=lambda item: item[0], reverse=True)
    sampler = OrderedRowSampler([row for _, row in rows])

    coeff = _sample_coefficients(g_mat, residual, residual_norm, sampler, config)

    best = 0.5 * coeff @ hess @ coeff + coeff @ grad
    assert best == pytest.approx(rows[-1][0], abs=1e-9)
