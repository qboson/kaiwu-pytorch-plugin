"""Numerical and allocation regressions for Gaussian marginalization."""

import itertools

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


def make_model(dtype, gaussian_visible, gaussian_size=5, bernoulli_size=3):
    """Use fixed parameters independent of constructor initialization."""
    visible, hidden = (gaussian_size, bernoulli_size)
    if not gaussian_visible:
        visible, hidden = hidden, visible
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        visible, hidden, is_visible_gaussian=gaussian_visible,
        dtype=dtype, device="cpu",
    )
    with torch.no_grad():
        model.mu.copy_(torch.linspace(-0.7, 0.9, gaussian_size, dtype=dtype))
        model.log_var.copy_(torch.linspace(-2, 2, gaussian_size, dtype=dtype))
        model.quadratic_coef.data = torch.linspace(
            -0.8, 1.1, gaussian_size * bernoulli_size, dtype=dtype,
        ).reshape(gaussian_size, bernoulli_size)
        model.linear_bias.data = torch.linspace(-0.3, 0.5, bernoulli_size, dtype=dtype)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gaussian_visible", [True, False])
def test_ising_energy_matches_independent_gaussian_integral(dtype, gaussian_visible):
    """Integrating Gaussian units fixes the relative energy of every binary state."""
    model = make_model(dtype, gaussian_visible)
    states = torch.tensor(list(itertools.product((0., 1.), repeat=3)), dtype=dtype)
    means = model.mu + states @ model.quadratic_coef.T
    marginal_energy = (
        -states @ model.linear_bias
        - 0.5 * (means.square() / model.var).sum(dim=1)
    )
    spins = torch.cat((2 * states - 1, torch.ones(8, 1, dtype=dtype)), dim=1)
    parameters_before = {name: value.detach().clone()
                         for name, value in model.named_parameters()}
    matrix = model.get_ising_matrix()
    assert isinstance(matrix, np.ndarray)
    assert matrix.dtype == (np.float32 if dtype == torch.float32 else np.float64)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-6, atol=1e-7)
    ising_energy = -torch.einsum("bi,ij,bj->b", spins, torch.from_numpy(matrix), spins)
    torch.testing.assert_close(
        ising_energy - ising_energy[0], marginal_energy - marginal_energy[0],
        rtol=1e-5, atol=1e-5,
    )
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, parameters_before[name], rtol=0, atol=0)


class NoGaussianSquare(TorchDispatchMode):
    """Reject quadratic-size Gaussian temporaries, regardless of the operator."""

    def __init__(self, gaussian_size):
        super().__init__()
        self.gaussian_size = gaussian_size

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        values = result if isinstance(result, (tuple, list)) else (result,)
        for value in values:
            if isinstance(value, torch.Tensor):
                assert value.shape != (self.gaussian_size, self.gaussian_size), (
                    "Ising conversion allocated a dense Gaussian square matrix"
                )
        return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gaussian_visible", [True, False])
def test_ising_conversion_avoids_gaussian_square(dtype, gaussian_visible):
    """Large Gaussian partitions must not allocate a G by G precision tensor."""
    model = make_model(dtype, gaussian_visible, gaussian_size=257)
    with NoGaussianSquare(model.num_gaussian):
        matrix = model.get_ising_matrix()
    assert matrix.shape == (4, 4)
    assert np.isfinite(matrix).all()


def test_public_diagonal_precision_remains_available():
    """The allocation optimization leaves the explicit public property intact."""
    model = make_model(torch.float64, True)
    torch.testing.assert_close(model.diag_precision, torch.diag(1 / model.var))
