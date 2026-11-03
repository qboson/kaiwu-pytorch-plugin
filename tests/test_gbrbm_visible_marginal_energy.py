"""Acceptance tests for scoring logical visible states in both GBRBM orientations."""
from itertools import product
from math import log, pi
from pathlib import Path

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import gbrbm as module

assert Path(module.__file__).resolve() == (
    Path(__file__).resolve().parents[1] / 'src/kaiwu/torch_plugin/gbrbm.py'
)


def model(gaussian_visible, dtype=torch.float64, eps=1e-8):
    result = module.GaussianBernoulliRestrictedBoltzmannMachine(
        2 if gaussian_visible else 3, 3 if gaussian_visible else 2,
        is_visible_gaussian=gaussian_visible, device=torch.device('cpu'), eps=eps,
    )
    # Convert already-built parameters, without relying on unrelated constructor
    # dtype or initialization changes in other contributions.
    {torch.float64: result.double, torch.float32: result.float,
     torch.float16: result.half, torch.bfloat16: result.bfloat16}[dtype]()
    with torch.no_grad():
        result.mu.copy_(torch.tensor([.35, -.25], dtype=dtype))
        result.log_var.copy_(torch.tensor([.6, 1.7], dtype=dtype).log())
        result.quadratic_coef.copy_(torch.tensor([[.3, -.5, .7], [-.4, .2, .6]], dtype=dtype))
        result.linear_bias.copy_(torch.tensor([-.1, .25, -.3], dtype=dtype))
    return result


def observed(model):
    if model.is_visible_gaussian:
        return model.mu.new_tensor([[.2, -.6], [1.1, .8], [-.7, .5]])
    return model.mu.new_tensor(list(product((0., 1.), repeat=model.num_visible)))


def binary_sum(model, visible):
    """Enumerate hidden Bernoulli states through the existing joint energy API."""
    hidden = visible.new_tensor(list(product((0., 1.), repeat=model.num_bernoulli)))
    joint = torch.cat((visible[:, None, :].expand(-1, len(hidden), -1),
                       hidden[None, :, :].expand(len(visible), -1, -1)), dim=-1)
    energies = model.energy(joint.reshape(-1, model.num_nodes), enable_grad=True)
    return -torch.logsumexp(-energies.reshape(len(visible), len(hidden)), dim=1)


def gaussian_integral(model, visible):
    """Integrate actual joint energy with an independent Hermite rule.

    Nodes are centered at the prior Gaussian mean, not the conditional mean.
    The Jacobian and Hermite weight correction retain parameter gradients.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    indices = np.array(list(product(range(len(nodes)), repeat=model.num_gaussian)))
    x = visible.new_tensor(nodes[indices])
    gaussian = model.mu + (2 * model.var).sqrt() * x
    joint = torch.cat((gaussian[None, :, :].expand(len(visible), -1, -1),
                       visible[:, None, :].expand(-1, len(x), -1)), dim=-1)
    energies = model.energy(joint.reshape(-1, model.num_nodes), enable_grad=True)
    log_weights = visible.new_tensor(np.log(weights[indices]).sum(axis=1))
    log_jacobian = .5 * torch.log(2 * model.var).sum()
    return -torch.logsumexp(-energies.reshape(len(visible), len(x))
                            + x.square().sum(dim=1) + log_weights + log_jacobian, dim=1)


def joint_marginal(model, visible):
    return binary_sum(model, visible) if model.is_visible_gaussian else gaussian_integral(model, visible)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gaussian_visible_matches_exhaustive_binary_hidden_sum(dtype):
    rbm = model(True, dtype)
    visible = observed(rbm)
    actual = rbm.visible_marginal_energy(visible)
    assert actual.shape == (len(visible),) and actual.dtype == dtype
    torch.testing.assert_close(actual, binary_sum(rbm, visible))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_bernoulli_visible_matches_gaussian_integral_for_all_observations(dtype):
    rbm = model(False, dtype)
    visible = observed(rbm)
    actual = rbm.visible_marginal_energy(visible)
    assert actual.shape == (8,) and actual.dtype == dtype
    tolerance = 2e-5 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual, gaussian_integral(rbm, visible), rtol=0, atol=tolerance)


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_scores_give_fixed_model_visible_log_probability_ratios(gaussian_visible):
    rbm = model(gaussian_visible)
    visible = observed(rbm)
    actual = rbm.visible_marginal_energy(visible)
    integrated_log_mass = -joint_marginal(rbm, visible)
    torch.testing.assert_close(actual[1:] - actual[0],
                               integrated_log_mass[0] - integrated_log_mass[1:])
    assert torch.argmax(actual) == torch.argmin(integrated_log_mass)


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_input_all_parameter_gradients_and_input_hessian_match_joint_marginal(gaussian_visible):
    rbm = model(gaussian_visible)
    visible = observed(rbm)[:1].clone().requires_grad_(True)
    # Choose a nonzero Bernoulli state so the coupling and bias derivatives
    # exercise the observed data rather than vanish at the all-zero state.
    if not gaussian_visible:
        visible = rbm.mu.new_tensor([[1., 0., 1.]]).requires_grad_(True)
    inputs = (visible, *rbm.parameters())
    actual = rbm.visible_marginal_energy(visible, enable_grad=True)
    expected = joint_marginal(rbm, visible)
    actual_gradients = torch.autograd.grad(actual.sum(), inputs, create_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), inputs, create_graph=True)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-11, atol=2e-12)
    actual_hessian = torch.stack([
        torch.autograd.grad(actual_gradients[0][:, index].sum(), visible, retain_graph=True)[0]
        for index in range(rbm.num_visible)
    ])
    expected_hessian = torch.stack([
        torch.autograd.grad(expected_gradients[0][:, index].sum(), visible, retain_graph=True)[0]
        for index in range(rbm.num_visible)
    ])
    torch.testing.assert_close(actual_hessian, expected_hessian, rtol=2e-11, atol=2e-12)


def test_gaussian_integration_keeps_variance_dependent_normalization_and_gradient():
    rbm = model(False)
    with torch.no_grad():
        rbm.quadratic_coef.zero_()
        rbm.linear_bias.zero_()
    actual = rbm.visible_marginal_energy(observed(rbm), enable_grad=True)
    expected = -.5 * (rbm.var.log() + log(2 * pi)).sum()
    torch.testing.assert_close(actual, expected.expand_as(actual))
    gradient = torch.autograd.grad(actual.mean(), rbm.log_var)[0]
    torch.testing.assert_close(gradient, gradient.new_full(gradient.shape, -.5))


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_variance_floor_matches_joint_energy_and_stops_clipped_variance_gradient(gaussian_visible):
    rbm = model(gaussian_visible, eps=.001)
    with torch.no_grad():
        rbm.log_var[0] = -20
        rbm.quadratic_coef[0].zero_()
    visible = observed(rbm)
    actual = rbm.visible_marginal_energy(visible, enable_grad=True)
    torch.testing.assert_close(actual, joint_marginal(rbm, visible), rtol=2e-11, atol=2e-11)
    gradient = torch.autograd.grad(actual.sum(), rbm.log_var)[0]
    assert gradient[0].item() == 0 and gradient[1].item() != 0


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_explicit_gradient_option_matches_energy_evaluation_convention(gaussian_visible):
    rbm = model(gaussian_visible)
    visible = observed(rbm).requires_grad_(True)
    assert not rbm.visible_marginal_energy(visible).requires_grad
    assert visible.grad is None and all(parameter.grad is None for parameter in rbm.parameters())
    with torch.no_grad():
        assert not rbm.visible_marginal_energy(visible).requires_grad
        score = rbm.visible_marginal_energy(visible, enable_grad=True)
    assert score.requires_grad
    score.sum().backward()
    assert visible.grad is not None
    assert all(parameter.grad is not None for parameter in rbm.parameters())


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_inference_mode_prevents_autograd_recording(gaussian_visible):
    rbm = model(gaussian_visible)
    visible = observed(rbm)
    with torch.inference_mode():
        for enable_grad in (False, True):
            assert not rbm.visible_marginal_energy(visible, enable_grad=enable_grad).requires_grad


@pytest.mark.parametrize('gaussian_visible', [True, False])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_scoring_uses_current_parameters_and_retains_dtype(gaussian_visible, dtype):
    rbm = model(gaussian_visible, dtype)
    visible = observed(rbm)
    actual = rbm.visible_marginal_energy(visible)
    assert actual.dtype == dtype and actual.device == rbm.mu.device
    # Match the effective quantized parameters/variance in a float64 joint
    # energy oracle, without depending on low-precision quadrature arithmetic.
    reference = model(gaussian_visible)
    with torch.no_grad():
        for destination, source in zip(reference.parameters(), rbm.parameters()):
            destination.copy_(source.double())
        reference.log_var.copy_(rbm.var.double().log())
    expected = joint_marginal(reference, visible.double())
    tolerance = 8 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_stable_tails_without_exponentiating_logits_or_variance_products(gaussian_visible):
    rbm = model(gaussian_visible)
    with torch.no_grad():
        rbm.linear_bias.copy_(rbm.linear_bias.new_tensor([-1000., 1000., 800.]))
        rbm.quadratic_coef.mul_(100)
    visible = observed(rbm)
    if gaussian_visible:
        visible = visible * 1000
    score = rbm.visible_marginal_energy(visible, enable_grad=True)
    assert torch.isfinite(score).all()
    gradients = torch.autograd.grad(score.sum(), tuple(rbm.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    if gaussian_visible:
        torch.testing.assert_close(score, binary_sum(rbm, visible), rtol=2e-14, atol=2e-9)
    # A large but finite half-precision variance must not overflow merely
    # because a normalizing constant multiplies it by 2*pi before taking log.
    half = model(False, torch.float16)
    with torch.no_grad():
        half.log_var.fill_(np.log(20000))
    assert torch.isfinite(half.var).all()
    assert torch.isfinite(half.visible_marginal_energy(observed(half))).all()


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_empty_observation_batch_returns_empty_energy_vector(gaussian_visible):
    rbm = model(gaussian_visible)
    result = rbm.visible_marginal_energy(rbm.mu.new_empty((0, rbm.num_visible)))
    assert result.shape == (0,) and result.dtype == rbm.mu.dtype


@pytest.mark.parametrize('gaussian_visible', [True, False])
def test_legacy_marginal_remains_gaussian_partition_scoring_without_gradients(gaussian_visible):
    rbm = model(gaussian_visible)
    gaussian = rbm.mu.new_tensor([[.2, -.6], [1.1, .8]]).requires_grad_(True)
    result = rbm.marginal_energy(gaussian)
    torch.testing.assert_close(result, binary_sum(rbm, gaussian))
    assert not result.requires_grad


@pytest.mark.parametrize('gaussian_visible', [True, False])
@pytest.mark.parametrize('wrong_shape', ['vector', 'width'])
def test_visible_interface_rejects_shapes_that_do_not_identify_observation_rows(gaussian_visible, wrong_shape):
    rbm = model(gaussian_visible)
    shape = (rbm.num_visible,) if wrong_shape == 'vector' else (2, rbm.num_visible + 1)
    with pytest.raises(ValueError, match='num_visible'):
        rbm.visible_marginal_energy(rbm.mu.new_zeros(shape))
