"""External Ising results must remain valid negative-phase states."""
from itertools import product

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.abstract_boltzmann_machine import AbstractBoltzmannMachine
from kaiwu.torch_plugin.full_boltzmann_machine import BoltzmannMachine
from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine
from kaiwu.torch_plugin.restricted_boltzmann_machine import RestrictedBoltzmannMachine


class ResultSampler:
    """Offline optimizer adapter returning recorded complete Ising batches."""

    def __init__(self, *results):
        self.results = results
        self.matrices = []

    def solve(self, matrix):
        self.matrices.append(np.array(matrix, copy=True))
        return self.results[min(len(self.matrices) - 1, len(self.results) - 1)]


def full_model(nodes=3):
    return BoltzmannMachine(
        nodes,
        linear_bias=torch.tensor([2., 6., 10.][:nodes]),
        quadratic_coef=torch.tensor(
            [[0., .5, -1.], [0., 0., 2.], [0., 0., 0.]]
        )[:nodes, :nodes].clone(),
        device="cpu",
    )


INVALID_RESULTS = [
    pytest.param(None, id="pending"),
    pytest.param(np.empty((0, 3)), id="empty"),
    pytest.param(np.ones(3), id="rank-one"),
    pytest.param(np.ones((1, 2)), id="missing-spin"),
    pytest.param(np.ones((1, 4)), id="extra-spin"),
    pytest.param(np.array([[0, 0, 1]]), id="binary-not-ising"),
    pytest.param(np.array([[1., -.5, 1.]]), id="fractional"),
    pytest.param(np.array([[1., 2., 1.]]), id="nonunit"),
    pytest.param(np.array([[1, -1, 1], [1, -1, 0]]), id="later-gauge"),
    pytest.param(np.array([[1., -1., 1.], [np.nan, -1., 1.]]), id="later-nan"),
    pytest.param(np.array([[1., -1., 1.], [1., np.inf, 1.]]), id="later-inf"),
    pytest.param(np.array([[1, -1, 1], [0, -1, 1]]), id="later-binary"),
    pytest.param(np.array([["1", "-1", "1"]]), id="text"),
]


@pytest.mark.parametrize("raw", INVALID_RESULTS)
@pytest.mark.parametrize("conditional", [False, True], ids=["sample", "condition"])
def test_invalid_backend_batches_are_rejected_before_decoding(raw, conditional):
    # Both calls submit a 3-spin matrix, but only one has two total model nodes.
    model = full_model(3 if conditional else 2)
    sampler = ResultSampler(raw)
    with pytest.raises(RuntimeError, match="sampler.solve"):
        if conditional:
            model.condition_sample(sampler, torch.tensor([[0.]]))
        else:
            model.sample(sampler)
    assert len(sampler.matrices) == 1
    assert sampler.matrices[0].shape == (3, 3)


@pytest.mark.parametrize("raw", [np.array([[0., 0., 1.]]),
                                np.array([[np.nan, 1., 1.]])],
                         ids=["invalid-binary", "nan"])
def test_bad_negative_phase_cannot_reach_a_real_sgd_step(raw):
    model = full_model(2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
    with pytest.raises(RuntimeError, match="sampler.solve"):
        negative = model.sample(ResultSampler(raw))
        model.objective(torch.zeros(1, 2), negative).backward()
        optimizer.step()
    for parameter, original in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, original)
        assert parameter.grad is None
    assert not optimizer.state


def test_empty_later_condition_cannot_silently_drop_observed_data():
    model = full_model()
    observed = torch.tensor([[0.], [1.], [0.]])
    original = observed.clone()
    sampler = ResultSampler(np.array([[1, -1, 1]]), np.empty((0, 3)),
                            np.array([[-1, 1, 1]]))
    with pytest.raises(RuntimeError, match="sampler.solve"):
        model.condition_sample(sampler, observed)
    assert len(sampler.matrices) == 2
    torch.testing.assert_close(observed, original)


def binary_energy(states, bias, edges):
    """Independent polynomial over binary units; no Ising conversion involved."""
    return np.array([
        -sum(float(bias[i]) * state[i] for i in range(len(state)))
        -sum(weight * state[i] * state[j] for i, j, weight in edges)
        for state in states
    ])


@pytest.mark.parametrize("restricted", [False, True], ids=["full-bm", "rbm"])
def test_binary_models_preserve_gauges_energies_objective_and_gradients(restricted):
    if restricted:
        model = RestrictedBoltzmannMachine(
            2, 1, quadratic_coef=torch.tensor([[.5], [-1.]]),
            linear_bias=torch.tensor([2., 6., 10.]), device="cpu",
        )
        edges = [(0, 2, .5), (1, 2, -1.)]
    else:
        model = full_model()
        edges = [(0, 1, .5), (0, 2, -1.), (1, 2, 2.)]
    # Rows one and two describe the same state with opposite global gauges.
    sampler = ResultSampler(np.array([[-1, 1, -1, 1], [1, -1, 1, -1],
                                      [1, -1, 1, 1]], dtype=np.int8))
    negative = model.sample(sampler)
    expected_states = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=float)
    np.testing.assert_array_equal(negative.numpy(), expected_states)
    assert sampler.matrices[0].shape == (4, 4)
    positive = torch.tensor([[1., 1., 0.], [0., 0., 1.]])
    bias = [2., 6., 10.]
    negative_energy = binary_energy(expected_states, bias, edges)
    positive_energy = binary_energy(positive.numpy(), bias, edges)
    torch.testing.assert_close(model(negative), torch.tensor(negative_energy).float())
    loss = model.objective(positive, negative)
    assert loss.item() == pytest.approx(positive_energy.mean() - negative_energy.mean())
    loss.backward()
    torch.testing.assert_close(model.linear_bias.grad,
                               negative.mean(0) - positive.mean(0))
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for i, j, _ in edges:
        expected_gradient = np.mean(expected_states[:, i] * expected_states[:, j])
        expected_gradient -= float((positive[:, i] * positive[:, j]).mean())
        actual = model.quadratic_coef.grad[i, j - 2] if restricted else (
            model.quadratic_coef.grad[i, j])
        assert actual.item() == pytest.approx(expected_gradient)


def gaussian_energy(states, mu, variance, weights, bias):
    """Scalar Gaussian/binary Hamiltonian evaluated independently in Python."""
    energies = []
    for state in states:
        gaussian, binary = state[:len(mu)], state[len(mu):]
        energy = sum(.5 * (g - m) ** 2 / v
                     for g, m, v in zip(gaussian, mu, variance))
        energy -= sum(gaussian[i] / variance[i] * weights[i, j] * binary[j]
                      for i, j in product(range(len(mu)), range(len(bias))))
        energy -= sum(b * h for b, h in zip(bias, binary))
        energies.append(energy)
    return np.array(energies)


@pytest.mark.parametrize("visible_gaussian", [True, False])
def test_gbrbm_validates_actual_bernoulli_matrix_not_total_nodes(visible_gaussian):
    model = GaussianBernoulliRestrictedBoltzmannMachine(
        2, 1, is_visible_gaussian=visible_gaussian, device="cpu",
    )
    gaussian, binary = model.num_gaussian, model.num_bernoulli
    mu = np.array([.25, -.5][:gaussian])
    variance = np.array([2., 4.][:gaussian])
    weights = np.array([[.5, -1.], [1., .25]])[:gaussian, :binary]
    bias = np.array([.25, -.5][:binary])
    with torch.no_grad():
        model.mu.copy_(torch.tensor(mu))
        model.log_var.copy_(torch.tensor(np.log(variance)))
        model.quadratic_coef.copy_(torch.tensor(weights))
        model.linear_bias.copy_(torch.tensor(bias))
    if binary == 1:
        raw = np.array([[-1, 1], [1, -1], [1, 1]], dtype=float)
        bits = np.array([[0], [0], [1]], dtype=float)
    else:
        raw = np.array([[-1, 1, 1], [1, -1, -1], [1, 1, 1]], dtype=float)
        bits = np.array([[0, 1], [0, 1], [1, 1]], dtype=float)
    sampler = ResultSampler(raw)
    negative = model.sample(sampler)
    assert sampler.matrices[0].shape == (binary + 1, binary + 1)
    assert sampler.matrices[0].shape[0] != model.num_nodes + 1
    expected_states = np.concatenate([bits @ weights.T + mu, bits], axis=1)
    np.testing.assert_allclose(negative.numpy(), expected_states, atol=1e-7)
    positive = torch.zeros(2, 3)
    expected_negative = gaussian_energy(expected_states, mu, variance, weights, bias)
    expected_positive = gaussian_energy(positive.numpy(), mu, variance, weights, bias)
    np.testing.assert_allclose(model(negative).detach().numpy(), expected_negative,
                               rtol=1e-6, atol=1e-7)
    loss = model.objective(positive, negative)
    assert loss.item() == pytest.approx(expected_positive.mean() - expected_negative.mean())
    loss.backward()
    torch.testing.assert_close(model.linear_bias.grad, negative[:, gaussian:].mean(0))
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    with pytest.raises(RuntimeError, match="sampler.solve"):
        model.sample(ResultSampler(np.ones((1, model.num_nodes + 1))))


class ParameterlessBM(AbstractBoltzmannMachine):
    """Minimal public base-class implementation without num_nodes or parameters."""

    def _to_ising_matrix(self):
        return np.zeros((3, 3))


@pytest.mark.parametrize("container", [
    lambda: np.array([[-1, 1, 1], [1, -1, -1]], dtype=np.float64),
    lambda: torch.tensor([[-1, 1, 1], [1, -1, -1]], dtype=torch.int64),
    lambda: np.ones((2, 3), dtype=bool),
], ids=["numpy-float64", "cpu-tensor", "all-positive-bool"])
def test_parameterless_base_preserves_accepted_numeric_containers(container):
    raw = container()
    model = ParameterlessBM(device="cpu")
    actual = model.sample(ResultSampler(raw))
    expected = [[1., 1.], [1., 1.]] if np.asarray(raw).dtype.kind == "b" else (
        [[0., 1.], [0., 1.]])
    torch.testing.assert_close(actual, torch.tensor(expected))
    assert actual.dtype == torch.float32
    with pytest.raises(RuntimeError, match="sampler.solve"):
        model.sample(ResultSampler(np.ones((1, 4))))


def test_conditional_results_keep_each_condition_all_rows_and_requested_dtype():
    model = full_model()
    observed = torch.tensor([[0.], [1.]])
    sampler = ResultSampler(np.array([[-1, 1, 1], [1, -1, -1]]),
                            np.array([[1, -1, 1]]))
    actual = model.condition_sample(sampler, observed, dtype=torch.float64)
    expected = torch.tensor([[0., 0., 1.], [0., 0., 1.], [1., 1., 0.]],
                            dtype=torch.float64)
    torch.testing.assert_close(actual, expected)
    assert [matrix.shape for matrix in sampler.matrices] == [(3, 3), (3, 3)]
    # Nothing remains to sample when all model units are observed; only the gauge remains.
    all_observed = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
    sampler = ResultSampler(np.array([[1], [-1]]), np.array([[-1]]))
    actual = model.condition_sample(sampler, all_observed)
    torch.testing.assert_close(actual, all_observed[[0, 0, 1]])
    assert [matrix.shape for matrix in sampler.matrices] == [(1, 1), (1, 1)]


@pytest.mark.parametrize("conditional", [False, True])
def test_solver_exceptions_propagate_without_retry(conditional):
    error = LookupError("backend failed")

    class FailedSampler:
        def solve(self, matrix):
            raise error

    with pytest.raises(LookupError) as caught:
        if conditional:
            full_model().condition_sample(FailedSampler(), torch.tensor([[0.]]))
        else:
            full_model().sample(FailedSampler())
    assert caught.value is error
