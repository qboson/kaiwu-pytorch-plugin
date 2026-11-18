"""Conditional variance and likelihood-gradient controls for sampler output."""

import itertools

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.gbrbm import GaussianBernoulliRestrictedBoltzmannMachine


class BalancedSampler:
    """Enumerate legal Bernoulli assignments at the external solver boundary."""

    def __init__(self, repeats=1):
        self.repeats = repeats
        self.last_matrix = None

    def solve(self, matrix):
        self.last_matrix = matrix.copy()
        states = np.array(list(itertools.product((-1., 1.), repeat=matrix.shape[0] - 1)))
        states = np.repeat(states, self.repeats, axis=0)
        return np.c_[states, np.ones(len(states))]


@pytest.fixture(params=[(torch.float32, True), (torch.float64, True),
                        (torch.float32, False), (torch.float64, False)])
def model(request):
    dtype, gaussian_visible = request.param
    with torch.random.fork_rng():
        torch.manual_seed(19)
        visible, hidden = (2, 3) if gaussian_visible else (3, 2)
        result = GaussianBernoulliRestrictedBoltzmannMachine(
            visible, hidden, is_visible_gaussian=gaussian_visible, dtype=dtype, device="cpu",
            quadratic_coef=torch.zeros(2, 3, dtype=dtype),
            linear_bias=torch.zeros(3, dtype=dtype),
        )
        with torch.no_grad():
            result.mu.copy_(torch.tensor([0.25, -0.5], dtype=dtype))
            result.log_var.copy_(torch.log(torch.tensor([1., 4.], dtype=dtype)))
            result.quadratic_coef.copy_(torch.tensor([[0.2, -0.1, 0.3], [0.1, 0.3, -0.2]], dtype=dtype))
            result.linear_bias.zero_()
    return result


def test_default_sampling_retains_conditional_means_and_rng(model):
    sampler = BalancedSampler()
    state = torch.random.get_rng_state().clone()
    result = model.sample(sampler)
    expected = model.infer_from_bernoulli(result[:, model.num_gaussian:], no_random=True)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    torch.testing.assert_close(model.sample(BalancedSampler(), no_random=True), expected)
    assert not result.requires_grad
    assert result.dtype == model.dtype and result.device.type == "cpu"
    np.testing.assert_array_equal(sampler.last_matrix, model.get_ising_matrix())


def test_random_sampling_uses_the_conditional_standard_deviation(model, monkeypatch):
    noise = torch.tensor([[-1., 1.], [1., -1.]] * 4, dtype=model.dtype)
    monkeypatch.setattr(torch, "randn_like", lambda means: noise.to(means))
    result = model.sample(BalancedSampler(), no_random=False)
    means = model.infer_from_bernoulli(result[:, 2:], no_random=True)
    torch.testing.assert_close(result[:, :2], means[:, :2] + noise * model.std)
    torch.testing.assert_close(result[:, 2:], means[:, 2:], rtol=0, atol=0)
    assert not result.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())


def test_seeded_random_samples_preserve_gaussian_moments(model):
    with torch.random.fork_rng():
        torch.manual_seed(12345)
        first = model.sample(BalancedSampler(repeats=2500), no_random=False)
        torch.manual_seed(12345)
        second = model.sample(BalancedSampler(repeats=2500), no_random=False)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    means = model.infer_from_bernoulli(first[:, 2:], no_random=True)
    residual = first[:, :2] - means[:, :2]
    torch.testing.assert_close(residual.mean(dim=0), torch.zeros(2, dtype=model.dtype),
                               rtol=0, atol=0.05)
    torch.testing.assert_close(residual.var(dim=0, unbiased=False), model.var,
                               rtol=0.05, atol=0.01)


def test_zero_coupling_matched_moments_have_stationary_variance_gradient(model, monkeypatch):
    with torch.no_grad():
        model.mu.zero_()
        model.log_var.zero_()
        model.quadratic_coef.zero_()
    # Antithetic pairs for each Bernoulli assignment give exact first/second moments.
    noise = torch.tensor([[-1., -1.], [1., 1.]] * 8, dtype=model.dtype)
    monkeypatch.setattr(torch, "randn_like", lambda means: noise.to(means))
    bernoulli = torch.tensor(list(itertools.product((0., 1.), repeat=3)), dtype=model.dtype)
    gaussian = torch.tensor(list(itertools.product((-1., 1.), repeat=2)), dtype=model.dtype)
    positive = torch.cat((gaussian.repeat_interleave(8, dim=0), bernoulli.repeat(4, 1)), dim=1)
    means_only = model.sample(BalancedSampler(repeats=2))
    model.objective(positive, means_only).backward()
    torch.testing.assert_close(model.log_var.grad, torch.full((2,), -0.5, dtype=model.dtype))
    model.zero_grad()
    negatives = model.sample(BalancedSampler(repeats=2), no_random=False)
    model.objective(positive, negatives).backward()
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter), rtol=0, atol=1e-7)
