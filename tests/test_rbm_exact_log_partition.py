"""Exact small-RBM normalization against full joint-state references."""
from itertools import product
import math

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine as RBM


def _states(num_units, dtype=torch.float64):
    return torch.tensor(list(product((0, 1), repeat=num_units)), dtype=dtype).reshape(
        2 ** num_units, num_units
    )


def _model(num_visible, num_hidden, dtype=torch.float64):
    weights = torch.linspace(
        -0.7, 0.9, num_visible * num_hidden, dtype=dtype
    ).reshape(num_visible, num_hidden)
    bias = torch.linspace(-0.4, 0.6, num_visible + num_hidden, dtype=dtype)
    return RBM(num_visible, num_hidden, quadratic_coef=weights, linear_bias=bias)


def _joint_reference(model):
    """Evaluate a polynomial on every binary joint assignment, without softplus."""
    states = _states(model.num_nodes).numpy()
    weights = model.quadratic_coef.detach().double().numpy()
    bias = model.linear_bias.detach().double().numpy()
    visible = states[:, :model.num_visible]
    hidden = states[:, model.num_visible:]
    interactions = (visible[:, :, None] * hidden[:, None, :]).reshape(
        len(states), -1
    )
    features = np.concatenate((states, interactions), axis=1)
    parameters = np.concatenate((bias, weights.reshape(-1)))
    log_weights = features @ parameters
    log_partition = np.logaddexp.reduce(log_weights)
    probabilities = np.exp(log_weights - log_partition)
    return states, features, log_weights, probabilities, log_partition


@pytest.mark.parametrize("shape", [(1, 3), (3, 1)])
def test_joint_reference_agrees_with_actual_hamiltonian(shape):
    """Keep the independent oracle anchored to the public energy convention."""
    model = _model(*shape)
    states, _, log_weights, _, _ = _joint_reference(model)
    torch.testing.assert_close(
        -model(torch.from_numpy(states)), torch.from_numpy(log_weights)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (2, 2)])
def test_log_partition_matches_full_joint_enumeration(shape, dtype):
    model = _model(*shape, dtype=dtype)
    joint_states = _states(model.num_nodes, dtype=dtype)
    expected = torch.logsumexp(-model(joint_states), dim=0)
    actual = model.exact_log_partition()
    assert actual.shape == torch.Size([])
    assert actual.dtype == dtype
    assert actual.device == model.quadratic_coef.device
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(1, 3), (3, 1)])
def test_visible_probabilities_sum_to_one(shape):
    model = _model(*shape)
    visible = _states(model.num_visible)
    hidden = _states(model.num_hidden)
    joint = torch.cat((
        visible.repeat_interleave(len(hidden), dim=0),
        hidden.repeat(len(visible), 1),
    ), dim=1)
    free_energy = -torch.logsumexp(
        -model(joint).reshape(len(visible), len(hidden)), dim=1
    )
    probabilities = torch.exp(-free_energy - model.exact_log_partition())
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.all(probabilities > 0)


@pytest.mark.parametrize("shape", [(1, 3), (3, 1)])
def test_gradient_is_exact_model_sufficient_statistics(shape):
    model = _model(*shape)
    _, features, _, probabilities, _ = _joint_reference(model)
    expected = probabilities @ features
    gradients = torch.autograd.grad(
        model.exact_log_partition(enable_grad=True),
        (model.linear_bias, model.quadratic_coef),
    )
    actual = torch.cat([gradient.reshape(-1) for gradient in gradients])
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_hessian_is_exact_sufficient_statistics_covariance(shape):
    model = _model(*shape)
    _, features, _, probabilities, _ = _joint_reference(model)
    mean = probabilities @ features
    centered = features - mean
    covariance = (centered.T * probabilities) @ centered
    parameters = (model.linear_bias, model.quadratic_coef)
    gradients = torch.autograd.grad(
        model.exact_log_partition(enable_grad=True), parameters, create_graph=True
    )
    vector = torch.cat([gradient.reshape(-1) for gradient in gradients])
    rows = []
    for component in vector:
        derivatives = torch.autograd.grad(component, parameters, retain_graph=True)
        rows.append(torch.cat([derivative.reshape(-1) for derivative in derivatives]))
    torch.testing.assert_close(
        torch.stack(rows), torch.from_numpy(covariance), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_exact_likelihood_sgd_step_uses_data_minus_model_statistics(shape):
    model = _model(*shape)
    states, features, log_weights, probabilities, _ = _joint_reference(model)
    # Observed data repeat a visible state, so the empirical weights are unequal.
    visible_data = _states(model.num_visible)[[0, -1, -1]]
    positive = []
    for datum in visible_data.numpy():
        matching = np.all(states[:, :model.num_visible] == datum, axis=1)
        conditional_log_weights = log_weights[matching]
        conditional = np.exp(
            conditional_log_weights - np.logaddexp.reduce(conditional_log_weights)
        )
        positive.append(conditional @ features[matching])
    expected_gradient = probabilities @ features - np.mean(positive, axis=0)
    hidden = _states(model.num_hidden)
    joint = torch.cat((
        visible_data.repeat_interleave(len(hidden), dim=0),
        hidden.repeat(len(visible_data), 1),
    ), dim=1)

    def negative_log_likelihood():
        free_energy = -torch.logsumexp(
            -model(joint).reshape(len(visible_data), len(hidden)), dim=1
        )
        return free_energy.mean() + model.exact_log_partition(enable_grad=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    original = torch.cat((model.linear_bias.detach(), model.quadratic_coef.detach().flatten()))
    loss = negative_log_likelihood()
    before = loss.item()
    loss.backward()
    actual_gradient = torch.cat((model.linear_bias.grad, model.quadratic_coef.grad.flatten()))
    torch.testing.assert_close(
        actual_gradient, torch.from_numpy(expected_gradient), atol=1e-12, rtol=1e-12
    )
    optimizer.step()
    updated = torch.cat((model.linear_bias.detach(), model.quadratic_coef.detach().flatten()))
    torch.testing.assert_close(updated, original - 0.05 * torch.from_numpy(expected_gradient))
    assert negative_log_likelihood().item() < before


@pytest.mark.parametrize("shape", [(1, 21), (21, 1)])
def test_only_smaller_partition_is_enumerated(shape):
    model = RBM(*shape, quadratic_coef=torch.zeros(shape, dtype=torch.float64),
                linear_bias=torch.linspace(-1, 1, sum(shape), dtype=torch.float64))
    expected = np.logaddexp(0, model.linear_bias.detach().numpy()).sum()
    actual = model.exact_log_partition(max_enumerated_units=1)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize("shape", [(1, 4), (4, 1), (2, 2)])
def test_uniform_binary_model_has_known_partition(shape):
    model = RBM(*shape, quadratic_coef=torch.zeros(shape, dtype=torch.float64),
                linear_bias=torch.zeros(sum(shape), dtype=torch.float64))
    torch.testing.assert_close(
        model.exact_log_partition(), torch.tensor(sum(shape) * math.log(2), dtype=torch.float64)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_extreme_finite_coefficients_are_stable(shape, dtype):
    model = RBM(*shape, quadratic_coef=torch.tensor([[1000., -900.]], dtype=dtype).reshape(shape),
                linear_bias=torch.tensor([-1000., 1200., -1100.], dtype=dtype))
    _, _, _, _, expected = _joint_reference(model)
    actual = model.exact_log_partition(enable_grad=True)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual.double(), torch.tensor(expected, dtype=torch.float64))
    for gradient in torch.autograd.grad(actual, model.parameters()):
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1)])
@pytest.mark.parametrize("bias", [19.99, 20., 20.01, 21., 30., -20.01, 1000., -1000.])
def test_threshold_and_extreme_log_partition_derivatives_match_joint_energy(shape, bias):
    """A reference must preserve FP64 curvature above softplus's linear threshold."""
    linear_bias = torch.zeros(sum(shape), dtype=torch.float64)
    # Put the large bias in the analytically marginalized layer for either direction.
    marginalized_index = shape[0] if shape[0] <= shape[1] else 0
    linear_bias[marginalized_index] = bias
    model = RBM(*shape, quadratic_coef=torch.zeros(shape, dtype=torch.float64),
                linear_bias=linear_bias)
    parameters = (model.linear_bias, model.quadratic_coef)

    def derivatives(log_partition):
        gradients = torch.autograd.grad(log_partition, parameters, create_graph=True)
        vector = torch.cat([gradient.reshape(-1) for gradient in gradients])
        rows = []
        for component in vector:
            row = torch.autograd.grad(component, parameters, retain_graph=True)
            rows.append(torch.cat([derivative.reshape(-1) for derivative in row]))
        return vector, torch.stack(rows)

    expected = torch.logsumexp(-model(_states(sum(shape))), dim=0)
    expected_gradient, expected_hessian = derivatives(expected)
    actual = model.exact_log_partition(enable_grad=True)
    actual_gradient, actual_hessian = derivatives(actual)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-15)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(actual_hessian, expected_hessian, atol=1e-13, rtol=1e-13)
    assert torch.isfinite(actual_gradient).all() and torch.isfinite(actual_hessian).all()


def test_gradient_context_is_explicit_and_restored():
    model = _model(2, 1)
    with torch.enable_grad():
        assert not model.exact_log_partition().requires_grad
        assert torch.is_grad_enabled()
    with torch.no_grad():
        assert model.exact_log_partition(enable_grad=True).requires_grad
        assert not torch.is_grad_enabled()


def test_deterministic_normalization_needs_no_sampler_or_random_draws(monkeypatch):
    model = _model(2, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("exact normalization must not use a sampler")

    monkeypatch.setattr(model, "sample", forbidden)
    monkeypatch.setattr(model, "get_ising_matrix", forbidden)
    rng_state = torch.random.get_rng_state().clone()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    first = model.exact_log_partition()
    torch.testing.assert_close(first, model.exact_log_partition(), rtol=0, atol=0)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_current_parameters_and_noncontiguous_weights_are_used():
    weights = torch.tensor([[0.3, -0.6], [0.2, 0.8], [-0.5, 0.1]]).t()
    assert not weights.is_contiguous()
    model = RBM(2, 3, quadratic_coef=weights, linear_bias=torch.zeros(5)).double()
    # The inherited cached construction device is irrelevant to parameter placement.
    model.device = torch.device("meta")
    first = model.exact_log_partition()
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    with torch.no_grad():
        model.linear_bias.add_(0.4)
    expected = torch.logsumexp(-model(_states(5)), dim=0)
    updated = model.exact_log_partition()
    torch.testing.assert_close(updated, expected)
    assert updated > first


@pytest.mark.parametrize("dtype,tolerance", [(torch.float16, 0.005), (torch.bfloat16, 0.04)])
def test_low_precision_small_model_keeps_native_dtype(dtype, tolerance):
    model = _model(2, 1, dtype=dtype)
    _, _, _, _, expected = _joint_reference(model)
    actual = model.exact_log_partition(enable_grad=True)
    assert actual.dtype == dtype
    assert abs(actual.item() - expected) < tolerance
    gradients = torch.autograd.grad(actual, model.parameters())
    assert all(gradient.dtype == dtype and torch.isfinite(gradient).all()
               for gradient in gradients)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_partition_follows_existing_constructor_contract(shape):
    model = _model(*shape)
    expected = torch.logsumexp(-model(_states(sum(shape))), dim=0)
    actual = model.exact_log_partition(max_enumerated_units=0)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.mark.parametrize("limit", [True, False, 1.0, 1.5, "2", None])
def test_enumeration_limit_rejects_non_integer_values(limit):
    with pytest.raises(TypeError, match="max_enumerated_units"):
        _model(1, 1).exact_log_partition(max_enumerated_units=limit)


def test_enumeration_limit_rejects_negative_values():
    with pytest.raises(ValueError, match="max_enumerated_units"):
        _model(1, 1).exact_log_partition(max_enumerated_units=-1)


@pytest.mark.parametrize("shape,limit", [((21, 21), 20), ((2, 3), 1), ((1, 1), 0)])
def test_enumeration_guard_precedes_state_allocation(shape, limit, monkeypatch):
    model = _model(*shape)

    def forbidden(*args, **kwargs):
        raise AssertionError("over-budget enumeration was allocated")

    monkeypatch.setattr(torch, "arange", forbidden)
    with pytest.raises(ValueError, match="max_enumerated_units"):
        model.exact_log_partition(max_enumerated_units=limit)


def test_numpy_integer_budget_is_accepted():
    model = _model(2, 1)
    torch.testing.assert_close(
        model.exact_log_partition(max_enumerated_units=np.int64(1)),
        torch.logsumexp(-model(_states(3)), dim=0),
    )
