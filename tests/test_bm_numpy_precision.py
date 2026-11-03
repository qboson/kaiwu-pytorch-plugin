"""Public samplers accept BF16 models without changing supported NumPy precision."""

import itertools

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
NUMPY_DTYPES = {
    torch.bfloat16: np.float32,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


class EnumeratedSpinAdapter:
    """Offline sampler interface returning legal spins, without claiming equilibrium."""

    def __init__(self):
        self.matrices = []

    def solve(self, matrix):
        self.matrices.append(matrix.copy())
        return np.array([
            [gauge * (2 * bit - 1) for bit in bits] + [gauge]
            for bits in itertools.product((0, 1), repeat=len(matrix) - 1)
            for gauge in (-1, 1)
        ], dtype=np.int8)


def _convert(model, dtype):
    return {
        torch.bfloat16: model.bfloat16,
        torch.float16: model.half,
        torch.float32: model.float,
        torch.float64: model.double,
    }[dtype]()


def _binary_model(kind, dtype):
    model = (BoltzmannMachine(3, device="cpu") if kind == "bm"
             else RestrictedBoltzmannMachine(2, 1, device="cpu"))
    model = _convert(model, dtype)
    weights = ([[0, .25, -.125], [0, 0, .375], [0, 0, 0]] if kind == "bm"
               else [[.25], [-.125]])
    with torch.no_grad():
        model.quadratic_coef.copy_(torch.tensor(weights, dtype=dtype))
        model.linear_bias.copy_(torch.tensor([.125, -.25, .375], dtype=dtype))
    return model


def _energy(model, bits):
    """Independent binary Hamiltonian using Python arithmetic on stored coefficients."""
    bias = model.linear_bias.detach().tolist()
    weights = model.quadratic_coef.detach().tolist()
    result = -sum(bit * value for bit, value in zip(bits, bias))
    if isinstance(model, BoltzmannMachine):
        return result - sum(weights[i][j] * bits[i] * bits[j]
                            for i in range(3) for j in range(i + 1, 3))
    return result - sum(weights[i][0] * bits[i] * bits[2] for i in range(2))


def _assert_energy_differences(matrix, binary_energy):
    baseline = np.array([-1.] * (len(matrix) - 1) + [1.])
    reference = -baseline @ matrix @ baseline
    for bits in itertools.product((0, 1), repeat=len(matrix) - 1):
        for gauge in (-1, 1):
            spins = np.array([gauge * (2 * bit - 1) for bit in bits] + [gauge])
            assert -spins @ matrix @ spins - reference == pytest.approx(
                binary_energy(bits) - binary_energy((0,) * len(bits)), abs=1e-12
            )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["bm", "rbm"])
def test_public_sampling_scores_and_trains_in_model_precision(kind, dtype):
    model = _binary_model(kind, dtype)
    adapter = EnumeratedSpinAdapter()

    negative = model.sample(adapter)

    assert negative.dtype == dtype
    assert not negative.requires_grad
    assert adapter.matrices[0].dtype == NUMPY_DTYPES[dtype]
    expected = [list(bits) for bits in itertools.product((0, 1), repeat=3) for _ in range(2)]
    assert negative.tolist() == expected
    torch.testing.assert_close(model(negative), torch.tensor(
        [_energy(model, bits) for bits in expected], dtype=dtype
    ))
    _assert_energy_differences(adapter.matrices[0], lambda bits: _energy(model, bits))
    model.objective(torch.zeros_like(negative), negative).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("dtype", DTYPES)
def test_conditional_sampling_preserves_clamps_and_explicit_output_dtype(dtype):
    model = _binary_model("bm", dtype)
    visible = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    adapter = EnumeratedSpinAdapter()

    # The existing public dtype argument controls conditional outputs independently.
    samples = model.condition_sample(adapter, visible, dtype=dtype)

    assert samples.dtype == dtype
    assert samples.tolist() == [[0, 1, 0], [0, 1, 0], [0, 1, 1], [0, 1, 1],
                                [1, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 1]]
    assert torch.isfinite(model(samples)).all()
    for matrix, clamped in zip(adapter.matrices, visible.tolist()):
        assert matrix.dtype == NUMPY_DTYPES[dtype]
        _assert_energy_differences(matrix, lambda bits: _energy(model, clamped + list(bits)))
    model.objective(torch.zeros_like(samples), samples).backward()
    assert torch.isfinite(model.linear_bias.grad).all()


def _gaussian_model(dtype, gaussian_visible):
    dimensions = (2, 1) if gaussian_visible else (1, 2)
    model = _convert(GaussianBernoulliRestrictedBoltzmannMachine(
        *dimensions, is_visible_gaussian=gaussian_visible, device="cpu"
    ), dtype)
    with torch.no_grad():
        model.mu.copy_(torch.tensor([.25, -.5], dtype=dtype))
        model.log_var.zero_()
        model.quadratic_coef.copy_(torch.tensor([[.25], [-.125]], dtype=dtype))
        model.linear_bias.copy_(torch.tensor([.375], dtype=dtype))
    return model


def _gaussian_minimum_energy(bit):
    """Original continuous energy at its analytic Gaussian minimum, with variance one."""
    result = -.375 * bit[0]
    for mean, weight in ((.25, .25), (-.5, -.125)):
        coupling = weight * bit[0]
        value = mean + coupling
        result += .5 * (value - mean)**2 - value * coupling
    return result


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gaussian_visible", [False, True])
def test_gaussian_reconstruction_and_objective_after_external_sampling(dtype, gaussian_visible):
    model = _gaussian_model(dtype, gaussian_visible)
    adapter = EnumeratedSpinAdapter()

    samples = model.sample(adapter)

    assert samples.dtype == dtype
    assert samples.tolist() == [[.25, -.5, 0], [.25, -.5, 0],
                                [.5, -.625, 1], [.5, -.625, 1]]
    assert adapter.matrices[0].dtype == NUMPY_DTYPES[dtype]
    _assert_energy_differences(adapter.matrices[0], _gaussian_minimum_energy)
    assert torch.isfinite(model(samples)).all()
    model.objective(torch.zeros_like(samples), samples).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("gaussian_visible", [False, True])
def test_bfloat16_gaussian_gibbs_sampler_initialization(gaussian_visible):
    model = _gaussian_model(torch.bfloat16, gaussian_visible)
    adapter = EnumeratedSpinAdapter()

    samples = model.gibbs_sample(n_step=1, sampler=adapter)

    assert samples.shape == (4, 3)
    assert samples.dtype == torch.bfloat16
    assert adapter.matrices[0].dtype == np.float32
    assert torch.isfinite(model(samples)).all()
    model.objective(torch.zeros_like(samples), samples).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["bm", "rbm"])
def test_double_export_keeps_coefficients_beyond_float32_resolution(kind):
    model = _binary_model(kind, torch.float64)
    with torch.no_grad():
        model.linear_bias[0] = .125 + 2**-35
    matrix = model.get_ising_matrix()

    assert matrix.dtype == np.float64
    assert matrix[0, -1] != float(np.float32(matrix[0, -1]))
    _assert_energy_differences(matrix, lambda bits: _energy(model, bits))
