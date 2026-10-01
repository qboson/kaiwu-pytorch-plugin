"""Numerical accuracy tests for the distributions used by QVAE."""

import math

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import (
    FactorialBernoulliUtil,
    MixtureGeneric,
    sigmoid_cross_entropy_with_logits,
)


TAIL_CASES = [
    (torch.float16, 11.0, 5e-3),
    (torch.bfloat16, 15.0, 2e-2),
    (torch.float32, 20.0, 1e-6),
    (torch.float64, 40.0, 1e-12),
]


@pytest.mark.parametrize("dtype,magnitude,rtol", TAIL_CASES)
def test_log_prob_retains_mode_probability_in_both_tails(dtype, magnitude, rtol):
    """Finite confident predictions still have a negative log probability."""
    logits = torch.tensor([-magnitude, magnitude], dtype=dtype)
    samples = torch.tensor([0.0, 1.0], dtype=dtype)

    actual = FactorialBernoulliUtil(logits).log_prob_per_var(samples)

    # log p(mode) = -log(1 + exp(-|logit|)), evaluated independently.
    expected = torch.full_like(logits, -math.log1p(math.exp(-magnitude)))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0)
    assert torch.all(actual < 0.0)


@pytest.mark.parametrize("dtype,magnitude,rtol", TAIL_CASES)
def test_log_prob_retains_mode_gradient_in_both_tails(dtype, magnitude, rtol):
    """Tail gradients do not vanish merely because sigmoid rounds to one."""
    logits = torch.tensor([-magnitude, magnitude], dtype=dtype, requires_grad=True)
    samples = torch.tensor([0.0, 1.0], dtype=dtype)

    log_prob = FactorialBernoulliUtil(logits).log_prob_per_var(samples)
    actual = torch.autograd.grad(log_prob.sum(), logits)[0]

    tail_probability = 1.0 / (1.0 + math.exp(magnitude))
    expected = torch.tensor([-tail_probability, tail_probability], dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0)


@pytest.mark.parametrize("dtype,magnitude,rtol", TAIL_CASES)
def test_entropy_is_symmetric_and_accurate_in_both_tails(dtype, magnitude, rtol):
    """Entropy is invariant under exchanging the zero and one outcomes."""
    logits = torch.tensor([-magnitude, magnitude], dtype=dtype)

    actual = FactorialBernoulliUtil(logits).entropy()

    expected_value = math.log1p(math.exp(-magnitude))
    expected_value += magnitude / (1.0 + math.exp(magnitude))
    expected = torch.full_like(logits, expected_value)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0)
    torch.testing.assert_close(actual[0], actual[1], rtol=rtol, atol=0.0)


@pytest.mark.parametrize("dtype,magnitude,rtol", TAIL_CASES)
def test_entropy_retains_analytic_tail_gradient(dtype, magnitude, rtol):
    """The derivative of entropy is -logit * p(0) * p(1)."""
    logits = torch.tensor([-magnitude, magnitude], dtype=dtype, requires_grad=True)

    entropy = FactorialBernoulliUtil(logits).entropy()
    actual = torch.autograd.grad(entropy.sum(), logits)[0]

    tail_probability = 1.0 / (1.0 + math.exp(magnitude))
    derivative = magnitude * tail_probability * (1.0 - tail_probability)
    expected = torch.tensor([derivative, -derivative], dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0)


def test_log_prob_matches_torch_distribution_for_broadcast_batches():
    """The stable formula retains Bernoulli's sample/batch broadcasting."""
    logits = torch.tensor([[-4.0, 0.0, 4.0], [-2.0, 1.0, 3.0]], dtype=torch.float64)
    samples = torch.tensor([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0]]], dtype=logits.dtype)

    actual = FactorialBernoulliUtil(logits).log_prob_per_var(samples)
    expected = torch.distributions.Bernoulli(logits=logits).log_prob(samples)

    assert actual.shape == (2, 2, 3)
    torch.testing.assert_close(actual, expected)


def test_cross_entropy_matches_torch_for_soft_labels_and_gradients():
    """QVAE reconstruction targets can be probabilities rather than hard bits."""
    logits = torch.tensor([[-4.0, 0.0, 4.0]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[0.25, 0.5, 0.75]], dtype=logits.dtype, requires_grad=True)

    actual = sigmoid_cross_entropy_with_logits(logits, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )

    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (logits, labels), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (logits, labels))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_log_prob_preserves_boolean_samples():
    """Boolean observations remain usable as binary Bernoulli samples."""
    logits = torch.tensor([-2.0, 2.0], dtype=torch.float64)
    samples = torch.tensor([False, True])

    actual = FactorialBernoulliUtil(logits).log_prob_per_var(samples)
    expected = torch.distributions.Bernoulli(logits=logits).log_prob(samples.to(logits.dtype))

    torch.testing.assert_close(actual, expected)


def test_entropy_matches_torch_distribution_and_has_smooth_second_derivative():
    """Entropy remains smooth at zero for differentiable KL computations."""
    logits = torch.tensor([-4.0, 0.0, 4.0], dtype=torch.float64, requires_grad=True)

    actual = FactorialBernoulliUtil(logits).entropy()
    expected = torch.distributions.Bernoulli(logits=logits).entropy()

    torch.testing.assert_close(actual, expected)
    first_derivative = torch.autograd.grad(actual.sum(), logits, create_graph=True)[0]
    second_derivative = torch.autograd.grad(first_derivative.sum(), logits)[0]
    torch.testing.assert_close(second_derivative[1], torch.tensor(-0.25, dtype=logits.dtype))


def test_smoothed_posterior_inherits_accurate_bernoulli_entropy():
    """QVAE's mixture posterior uses the corrected latent Bernoulli entropy."""
    logits = torch.tensor([[-20.0, 20.0]], dtype=torch.float32)

    actual = MixtureGeneric(logits, smoothing_dist_beta=1.0).entropy().sum(dim=1)
    per_variable = math.log1p(math.exp(-20.0)) + 20.0 / (1.0 + math.exp(20.0))
    expected = torch.tensor([2.0 * per_variable], dtype=logits.dtype)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=0.0)
