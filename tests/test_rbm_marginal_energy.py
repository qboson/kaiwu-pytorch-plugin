"""Acceptance tests for the binary RBM's visible-state free energy."""

import itertools
import math

import pytest
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def _model(dtype):
    """Supply parameters in the requested precision without depending on .to()."""
    return RestrictedBoltzmannMachine(
        3, 2,
        quadratic_coef=torch.tensor([[0.2, -0.6], [0.3, 0.9], [-0.8, 0.4]], dtype=dtype),
        linear_bias=torch.tensor([-0.1, 0.2, 0.25, -0.4, 0.35], dtype=dtype),
        device="cpu",
    )


def _joint_energies(model, visible):
    """Enumerate hidden states and evaluate only the existing joint forward API."""
    hidden = torch.tensor(
        list(itertools.product((0.0, 1.0), repeat=model.num_hidden)),
        dtype=visible.dtype, device=visible.device,
    ).reshape(-1, model.num_hidden)
    full_states = torch.cat(
        [visible[:, None, :].expand(-1, len(hidden), -1),
         hidden[None, :, :].expand(len(visible), -1, -1)],
        dim=-1,
    )
    return model(full_states.reshape(-1, model.num_nodes)).reshape(len(visible), len(hidden))


def _enumerated_free_energy(model, visible):
    return -torch.logsumexp(-_joint_energies(model, visible), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_marginal_energy_matches_hidden_enumeration_for_every_visible_state(dtype):
    """The closed form must equal the hidden-state partition sum for all binary inputs."""
    model = _model(dtype)
    visible = torch.tensor(list(itertools.product((0.0, 1.0), repeat=3)), dtype=dtype)

    actual = model.marginal_energy(visible)

    assert actual.shape == (8,)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, _enumerated_free_energy(model, visible))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_free_energy_difference_equals_visible_log_probability_ratio(dtype):
    """A tiny enumerated joint distribution independently verifies the ratio convention."""
    model = _model(dtype)
    visible = torch.tensor(list(itertools.product((0.0, 1.0), repeat=3)), dtype=dtype)
    joint = _joint_energies(model, visible)
    normalized_joint = torch.softmax(-joint.flatten(), dim=0).reshape_as(joint)
    marginal_probability = normalized_joint.sum(dim=-1)

    free_energy = model.marginal_energy(visible)
    log_ratio = free_energy[5] - free_energy[2]

    expected = marginal_probability[2].log() - marginal_probability[5].log()
    torch.testing.assert_close(log_ratio, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("zero_hidden_bias", [False, True])
def test_zero_weights_retain_hidden_partition_constant(dtype, zero_hidden_bias):
    """Free energy retains hidden entropy even when visible and hidden states decouple."""
    model = _model(dtype)
    with torch.no_grad():
        model.quadratic_coef.zero_()
        if zero_hidden_bias:
            model.hidden_bias.zero_()
    visible = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=dtype)
    hidden_constant = (
        2 * math.log(2) if zero_hidden_bias
        else sum(math.log1p(math.exp(value)) for value in model.hidden_bias.tolist())
    )
    expected = torch.tensor(
        [-sum(value * bias for value, bias in zip(row, model.visible_bias.tolist())) - hidden_constant
         for row in visible.tolist()], dtype=dtype,
    )

    torch.testing.assert_close(model.marginal_energy(visible), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_hidden_logits_keep_free_energy_and_gradients_finite(dtype):
    """Stable log-sums handle positive and negative logits beyond exp's finite range."""
    model = _model(dtype)
    with torch.no_grad():
        model.quadratic_coef.copy_(torch.tensor([[1000.0, -1000.0]] * 3, dtype=dtype))
        model.hidden_bias.copy_(torch.tensor([1000.0, -1000.0], dtype=dtype))
    visible = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)

    free_energy = model.marginal_energy(visible, enable_grad=True)
    gradients = torch.autograd.grad(free_energy.sum(), (visible, *model.parameters()))

    torch.testing.assert_close(free_energy, _enumerated_free_energy(model, visible))
    assert torch.isfinite(free_energy).all()
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_input_and_parameter_gradients_match_hidden_enumeration(dtype):
    """Differentiable free energy produces the same gradients as explicit marginalization."""
    model = _model(dtype)
    visible = torch.tensor([[0.13, 0.27, 0.98], [0.35, 0.9, -0.2]], dtype=dtype, requires_grad=True)
    loss_weights = torch.tensor([0.5, -1.2], dtype=dtype)
    variables = (visible, *model.parameters())

    actual = torch.autograd.grad(
        (model.marginal_energy(visible, enable_grad=True) * loss_weights).sum(), variables
    )
    expected = torch.autograd.grad(
        (_enumerated_free_energy(model, visible) * loss_weights).sum(), variables
    )

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_visible_gradient_passes_finite_difference_gradcheck():
    """An independent numerical derivative checks the continuous input extension."""
    model = _model(torch.float64)
    visible = torch.tensor([[0.13, 0.27, 0.98], [0.35, 0.9, -0.2]], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda value: model.marginal_energy(value, enable_grad=True),
        (visible,), eps=1e-6, atol=1e-6, rtol=1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ambient_grad", [False, True])
@pytest.mark.parametrize("enable_grad", [False, True])
def test_gradient_option_restores_caller_context(dtype, ambient_grad, enable_grad):
    """The explicit option follows energy() conventions without changing caller context."""
    model = _model(dtype)
    visible = torch.ones((2, 3), dtype=dtype, requires_grad=True)

    with torch.set_grad_enabled(ambient_grad):
        result = model.marginal_energy(visible, enable_grad=enable_grad)
        assert result.requires_grad == enable_grad
        assert torch.is_grad_enabled() == ambient_grad
        default_result = model.marginal_energy(visible)
        assert not default_result.requires_grad
        assert torch.is_grad_enabled() == ambient_grad
    assert visible.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("empty_batch", [False, True])
def test_noncontiguous_and_empty_visible_batches(dtype, empty_batch):
    """Matrix operations naturally support strided input and zero-length batches."""
    model = _model(dtype)
    storage = torch.tensor([[0.0, 9.0, 1.0, 8.0, 0.0, 7.0], [1.0, 6.0, 0.0, 5.0, 1.0, 4.0]], dtype=dtype)
    visible = storage[:, ::2]
    if empty_batch:
        visible = visible[:0]
    else:
        assert not visible.is_contiguous()

    result = model.marginal_energy(visible, enable_grad=True)

    assert result.shape == (len(visible),)
    assert result.dtype == dtype
    torch.testing.assert_close(result, _enumerated_free_energy(model, visible))
