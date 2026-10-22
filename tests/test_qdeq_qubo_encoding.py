"""The Ising coefficient encoding must return the exact grid optimum."""

import itertools
import math

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import QDEQConfig
from kaiwu.torch_plugin.qdeq import _sample_coefficients


class ExhaustiveBestSampler:
    """Returns only the spin row maximising ``s^T J s``, found by brute force."""

    def __init__(self):
        self.matrices = []

    def solve(self, ising_mat):
        """Returns the best row of the exhaustive search."""
        self.matrices.append(np.array(ising_mat, copy=True))
        size = ising_mat.shape[0]
        best, best_value = None, -np.inf
        for spins in itertools.product([1.0, -1.0], repeat=size):
            row = np.array(spins)
            value = float(row @ ising_mat @ row)
            if value > best_value + 1e-12:
                best, best_value = row, value
        return best[None, :]


def _grid(config, scale):
    """Returns every coefficient vector reachable by the binary encoding."""
    step = 2.0 * scale / (2 ** config.coeff_bits - 1)
    levels = [-scale + index * step for index in range(2 ** config.coeff_bits)]
    return list(itertools.product(levels, repeat=config.num_basis))


def test_sampler_receives_the_auxiliary_spin_matrix():
    rng = np.random.default_rng(0)
    g_mat = torch.tensor(rng.normal(size=(2, 5)))
    residual = torch.tensor(rng.normal(size=5))
    config = QDEQConfig(num_basis=2, coeff_bits=3, coeff_range=1.5)
    sampler = ExhaustiveBestSampler()

    _sample_coefficients(g_mat, residual, float(residual.norm()), sampler, config)

    assert len(sampler.matrices) == 1
    matrix = sampler.matrices[0]
    assert matrix.shape == (config.num_basis * config.coeff_bits + 1,) * 2
    assert np.allclose(matrix, matrix.T)


def test_returns_the_exact_grid_minimum():
    rng = np.random.default_rng(1)
    g_mat = torch.tensor(rng.normal(size=(2, 5)))
    residual = torch.tensor(rng.normal(size=5))
    residual_norm = float(residual.norm())
    config = QDEQConfig(num_basis=2, coeff_bits=3, coeff_range=1.5)
    sampler = ExhaustiveBestSampler()

    coeff = _sample_coefficients(g_mat, residual, residual_norm, sampler, config)

    hess = (g_mat @ g_mat.T).numpy()
    grad = (g_mat @ residual).numpy()

    def objective(values):
        return 0.5 * values @ hess @ values + values @ grad

    scale = config.coeff_range * residual_norm / math.sqrt(config.num_basis)
    best = min(objective(np.array(values)) for values in _grid(config, scale))
    got = np.asarray(coeff)
    assert got.shape == (config.num_basis,)
    assert objective(got) == pytest.approx(best, abs=1e-9)

    step = 2.0 * scale / (2 ** config.coeff_bits - 1)
    levels = np.array([-scale + index * step for index in range(2 ** config.coeff_bits)])
    for value in got:
        assert np.min(np.abs(levels - value)) < 1e-9


def test_non_finite_directional_derivatives_return_no_step():
    g_mat = torch.tensor([[np.nan, 1.0]])
    residual = torch.tensor([1.0, 0.0])
    config = QDEQConfig(num_basis=1, coeff_bits=2)

    assert _sample_coefficients(g_mat, residual, 1.0, ExhaustiveBestSampler(), config) is None
