"""Generated Boltzmann states retain model precision through sampling and scoring."""

import itertools
import math

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.abstract_boltzmann_machine import AbstractBoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


class SpinSampler:
    """Return fixed spin states with both auxiliary-spin signs."""

    def solve(self, ising_matrix):
        assert ising_matrix.shape == (4, 4)
        return np.array([[1, -1, 1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1]])


def _model(kind, dtype, supplied=False):
    """Use inherited dtype conversion or directly supplied coefficients, never .to()."""
    weights = (
        [[0.0, 0.2, -0.7], [0.2, 0.0, 0.8], [-0.7, 0.8, 0.0]]
        if kind == "bm" else [[0.2], [-0.7]]
    )
    parameters = {
        "quadratic_coef": torch.tensor(weights, dtype=dtype),
        "linear_bias": torch.tensor([0.1, -0.3, 0.25], dtype=dtype),
    }
    model_type = BoltzmannMachine if kind == "bm" else RestrictedBoltzmannMachine
    dimensions = (3,) if kind == "bm" else (2, 1)
    model = model_type(*dimensions, device="cpu", **(parameters if supplied else {}))
    if not supplied:
        model = model.double() if dtype == torch.float64 else model.float()
        with torch.no_grad():
            model.quadratic_coef.copy_(parameters["quadratic_coef"])
            model.linear_bias.copy_(parameters["linear_bias"])
    return model


def _binary_energy(model, state):
    """Evaluate the binary Hamiltonian directly with Python arithmetic."""
    weights = model.quadratic_coef.detach().tolist()
    biases = model.linear_bias.detach().tolist()
    energy = -sum(value * bias for value, bias in zip(state, biases))
    if isinstance(model, BoltzmannMachine):
        return energy - sum(
            weights[left][right] * state[left] * state[right]
            for left in range(len(state)) for right in range(left + 1, len(state))
        )
    return energy - sum(
        weights[visible][hidden] * state[visible] * state[model.num_visible + hidden]
        for visible in range(model.num_visible) for hidden in range(model.num_hidden)
    )


@pytest.mark.parametrize("kind", ["bm", "rbm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("supplied", [False, True])
def test_external_samples_can_score_and_backpropagate_objective(kind, dtype, supplied):
    """Model-precision negative samples work in real contrastive training."""
    model = _model(kind, dtype, supplied)
    negative = model.sample(SpinSampler())
    positive = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=dtype)
    expected_states = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    expected_energy = torch.tensor(
        [_binary_energy(model, state) for state in expected_states.tolist()], dtype=dtype
    )

    # Scoring first exposes the original float32/double matmul failure directly.
    energies = model(negative)
    objective = model.objective(positive, negative)
    objective.backward()

    assert negative.dtype == dtype
    assert negative.device == model.linear_bias.device
    assert not negative.requires_grad
    assert torch.equal(negative, expected_states)
    torch.testing.assert_close(energies, expected_energy)
    expected_objective = (
        sum(_binary_energy(model, state) for state in positive.tolist()) / len(positive)
        - sum(expected_energy.tolist()) / len(negative)
    )
    torch.testing.assert_close(objective, torch.tensor(expected_objective, dtype=dtype))
    torch.testing.assert_close(model.linear_bias.grad, negative.mean(0) - positive.mean(0))
    assert torch.isfinite(model.quadratic_coef.grad).all()


def test_parameterless_abstract_sampler_retains_float32_fallback():
    """Usage-only abstract subclasses remain usable without model parameters."""
    class ParameterlessMachine(AbstractBoltzmannMachine):
        def _to_ising_matrix(self):
            return np.zeros((4, 4), dtype=np.float64)

    model = ParameterlessMachine(device="cpu")
    samples = model.sample(SpinSampler())

    assert samples.dtype == torch.float32
    assert samples.shape == (3, 3)
    assert torch.equal(samples, torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bernoulli", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_hidden_states_retain_precision_and_gradient_contract(dtype, bernoulli, requires_grad, monkeypatch):
    """Hidden inference retains observed inputs and differentiable probabilities."""
    model = _model("rbm", dtype)
    visible = torch.tensor([[0.123456789012, 0.987654321234], [0.9, 0.2]], dtype=dtype, requires_grad=True)
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.5))
    weights = model.quadratic_coef.detach().tolist()
    bias = model.hidden_bias.item()
    probabilities = [
        1.0 / (1.0 + math.exp(-(row[0] * weights[0][0] + row[1] * weights[1][0] + bias)))
        for row in visible.detach().tolist()
    ]
    expected_hidden = torch.tensor(
        [[float(probability > 0.5) if bernoulli else probability] for probability in probabilities],
        dtype=dtype,
    )

    states = model.get_hidden(visible, requires_grad=requires_grad, bernoulli=bernoulli)

    assert states.dtype == dtype
    assert torch.equal(states[:, :2], visible)
    torch.testing.assert_close(states[:, 2:], expected_hidden)
    assert states.requires_grad == requires_grad
    assert torch.isfinite(model(states)).all()
    if requires_grad:
        gradient = torch.autograd.grad(states.sum(), visible)[0]
        expected_gradient = torch.tensor(
            [
                [1.0 + (0.0 if bernoulli else probability * (1 - probability) * weight[0]) for weight in weights]
                for probability in probabilities
            ],
            dtype=dtype,
        )
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bernoulli", [False, True])
def test_visible_and_hidden_sampling_chain_can_score_without_downcasting(dtype, bernoulli, monkeypatch):
    """Alternating RBM inference returns states usable by the next layer and energy."""
    model = _model("rbm", dtype)
    hidden = torch.tensor([[0.123456789012], [0.987654321234]], dtype=dtype, requires_grad=True)
    monkeypatch.setattr(torch, "rand_like", lambda value: torch.full_like(value, 0.5))

    states = model.get_visible(hidden, bernoulli=bernoulli)

    assert states.dtype == dtype
    assert torch.equal(states[:, 2:], hidden)
    assert not states.requires_grad
    assert torch.isfinite(model(states)).all()
    next_states = model.get_hidden(states[:, :2], bernoulli=bernoulli)
    assert next_states.dtype == dtype
    assert torch.isfinite(model(next_states)).all()
    if bernoulli:
        assert torch.all((states[:, :2] == 0) | (states[:, :2] == 1))
    else:
        expected_visible = torch.tensor(
            [
                [
                    1.0 / (1.0 + math.exp(-(row[0] * weight[0] + bias)))
                    for weight, bias in zip(model.quadratic_coef.detach().tolist(), model.visible_bias.detach().tolist())
                ]
                for row in hidden.detach().tolist()
            ],
            dtype=dtype,
        )
        torch.testing.assert_close(states[:, :2], expected_visible)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rbm_ising_precision_matches_enumerated_binary_energy_differences(dtype):
    """Enumerated Hamiltonians detect loss of supplied double coefficients."""
    model = RestrictedBoltzmannMachine(
        2, 2,
        quadratic_coef=torch.tensor([[0.123456789012, -0.98765432101], [1.23456789012, -0.3333333333]], dtype=dtype),
        linear_bias=torch.tensor([-0.17320508075, 0.27182818284, -0.31415926535, 0.69314718056], dtype=dtype),
        device="cpu",
    )
    ising_matrix = model.get_ising_matrix()
    expected_dtype = np.float64 if dtype == torch.float64 else np.float32
    assert ising_matrix.dtype == expected_dtype
    baseline_spin = np.array([-1.0, -1.0, -1.0, -1.0, 1.0])
    baseline_energy = -baseline_spin @ ising_matrix @ baseline_spin
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    for state in itertools.product((0.0, 1.0), repeat=4):
        for gauge in (-1.0, 1.0):
            spins = np.array([gauge * (2 * value - 1) for value in state] + [gauge])
            energy_difference = -spins @ ising_matrix @ spins - baseline_energy
            assert energy_difference == pytest.approx(_binary_energy(model, state), rel=tolerance, abs=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("observed_visible", [False, True])
def test_gibbs_generated_states_work_in_energy_and_objective(dtype, observed_visible):
    """Both Gibbs initializers use model precision and retain clamped visible states."""
    model = _model("bm", dtype)
    visible = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype, requires_grad=True)
    samples = (
        model.gibbs_sample(num_steps=3, s_visible=visible)
        if observed_visible else model.gibbs_sample(num_steps=3, num_sample=2)
    )
    positive = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=dtype)

    assert samples.dtype == dtype
    assert not samples.requires_grad
    assert torch.all((samples == 0) | (samples == 1))
    assert torch.isfinite(model(samples)).all()
    model.objective(positive, samples).backward()
    assert torch.isfinite(model.linear_bias.grad).all()
    assert torch.isfinite(model.quadratic_coef.grad).all()
    if observed_visible:
        assert torch.equal(samples[:, :2], visible)


def _gbrbm(dtype, gaussian_visible):
    """Convert default parameters without relying on constructor dtype fixes."""
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        3, 2, is_visible_gaussian=gaussian_visible, device="cpu"
    )
    model = model.double() if dtype == torch.float64 else model.float()
    with torch.no_grad():
        model.mu.copy_(torch.tensor([0.123456789012, -0.27182818284, 0.31415926535][:model.num_gaussian], dtype=dtype))
        model.log_var.copy_(torch.log(torch.arange(1, model.num_gaussian + 1, dtype=dtype)))
        model.quadratic_coef.copy_(torch.tensor(
            [[0.123456789012 * (gaussian + 1) - 0.17320508075 * bernoulli
              for bernoulli in range(model.num_bernoulli)]
             for gaussian in range(model.num_gaussian)], dtype=dtype,
        ))
        model.linear_bias.copy_(torch.tensor([0.123456789012 * (index + 1) for index in range(model.num_bernoulli)], dtype=dtype))
    return model


class BernoulliSpinSampler:
    """Return complementary binary states after auxiliary-spin decoding."""

    def solve(self, ising_matrix):
        spins = np.array([1 if index % 2 == 0 else -1 for index in range(len(ising_matrix) - 1)])
        return np.stack([np.append(spins, 1), np.append(spins, -1)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gaussian_visible", [False, True])
@pytest.mark.parametrize("binarize", [False, True])
def test_gbrbm_converted_inference_chain_retains_parameter_dtype(dtype, gaussian_visible, binarize):
    """Both layouts retain observed Gaussian precision through two-way inference."""
    model = _gbrbm(dtype, gaussian_visible)
    gaussian = torch.tensor([[0.123456789012] * model.num_gaussian, [-0.987654321234] * model.num_gaussian], dtype=dtype, requires_grad=True)

    states = model.infer_from_gaussian(gaussian, binarize=binarize, no_random=True)

    assert states.dtype == dtype
    assert torch.equal(states[:, :model.num_gaussian], gaussian)
    assert not states.requires_grad
    assert torch.isfinite(model(states)).all()
    bernoulli = states[:, model.num_gaussian:]
    if binarize:
        assert torch.all((bernoulli == 0) | (bernoulli == 1))
    else:
        assert torch.all((bernoulli >= 0) & (bernoulli <= 1))
    reconstructed = model.infer_from_bernoulli(bernoulli, no_random=True)
    expected_gaussian = torch.tensor(
        [[mean + sum(weight * bit for weight, bit in zip(row, state))
          for mean, row in zip(model.mu.detach().tolist(), model.quadratic_coef.detach().tolist())]
         for state in bernoulli.tolist()], dtype=dtype,
    )
    assert reconstructed.dtype == dtype
    assert not reconstructed.requires_grad
    torch.testing.assert_close(reconstructed[:, :model.num_gaussian], expected_gaussian)
    assert torch.isfinite(model(reconstructed)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gaussian_visible", [False, True])
def test_gbrbm_external_sampling_supports_converted_objective(dtype, gaussian_visible):
    """The sampler-to-Gaussian reconstruction works after inherited dtype conversion."""
    model = _gbrbm(dtype, gaussian_visible)
    negative = model.sample(BernoulliSpinSampler())
    positive = torch.zeros((2, model.num_nodes), dtype=dtype)

    assert negative.dtype == dtype
    assert not negative.requires_grad
    assert torch.isfinite(model(negative)).all()
    model.objective(positive, negative).backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gaussian_visible", [False, True])
@pytest.mark.parametrize("initialization", ["gaussian", "bernoulli", "sampler", "random"])
def test_gbrbm_all_gibbs_initializers_retain_converted_dtype(dtype, gaussian_visible, initialization):
    """Gaussian, Bernoulli, external, and random starts all use current model precision."""
    model = _gbrbm(dtype, gaussian_visible)
    starts = {
        "gaussian": {"s_gaussian": torch.zeros((2, model.num_gaussian), dtype=dtype)},
        "bernoulli": {"s_bernoulli": torch.ones((2, model.num_bernoulli), dtype=dtype)},
        "sampler": {"sampler": BernoulliSpinSampler()},
        "random": {"n_sample": 2},
    }
    samples = model.gibbs_sample(n_step=2, **starts[initialization])

    assert samples.shape == (4, model.num_nodes)
    assert samples.dtype == dtype
    assert not samples.requires_grad
    bernoulli = samples[:, model.num_gaussian:]
    assert torch.all((bernoulli == 0) | (bernoulli == 1))
    assert torch.isfinite(model(samples)).all()
    model.objective(torch.zeros_like(samples), samples).backward()
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all()


def _minimum_gaussian_energy(model, bernoulli):
    """Evaluate the original Hamiltonian at each analytic Gaussian minimum."""
    means = model.mu.detach().tolist()
    variances = model.var.detach().tolist()
    weights = model.quadratic_coef.detach().tolist()
    energy = -sum(bias * bit for bias, bit in zip(model.linear_bias.detach().tolist(), bernoulli))
    for mean, variance, row in zip(means, variances, weights):
        coupling = sum(weight * bit for weight, bit in zip(row, bernoulli))
        gaussian = mean + coupling
        energy += 0.5 * (gaussian - mean)**2 / variance - gaussian * coupling / variance
    return energy


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gaussian_visible", [False, True])
def test_gbrbm_ising_precision_matches_enumerated_minimum_energies(dtype, gaussian_visible):
    """Ising coefficients retain precision after conversion in either Gaussian layout."""
    model = _gbrbm(dtype, gaussian_visible)
    ising_matrix = model.get_ising_matrix()
    assert ising_matrix.dtype == (np.float64 if dtype == torch.float64 else np.float32)
    baseline = np.array([-1.0] * model.num_bernoulli + [1.0])
    baseline_energy = -baseline @ ising_matrix @ baseline
    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    for state in itertools.product((0.0, 1.0), repeat=model.num_bernoulli):
        spins = np.array([2 * value - 1 for value in state] + [1.0])
        energy_difference = -spins @ ising_matrix @ spins - baseline_energy
        assert energy_difference == pytest.approx(_minimum_gaussian_energy(model, state), rel=tolerance, abs=tolerance)
