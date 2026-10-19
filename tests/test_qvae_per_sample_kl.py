"""The QVAE KL surrogate must retain each observation's positive energy."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae_dist_util import FactorialBernoulliUtil


class FixedSampler:
    """Return legal spins that decode to the all-zero negative state."""

    def solve(self, matrix):
        return np.array([[-1., -1., 1.]], dtype=np.float32)


class SuppliedQVAE(QVAE):
    """Use real RBM energies without invoking a licensed backend."""

    def _create_encoder(self):
        raise AssertionError("An encoder is supplied")

    def _create_decoder(self):
        raise AssertionError("A decoder is supplied")

    def _create_bm(self):
        raise AssertionError("A BM is supplied")

    def _create_sampler(self, sampler_type):
        raise AssertionError("A sampler is supplied")


def _model():
    config = SimpleNamespace(
        num_latent_units=2, loss_type="mse", dist_beta=1., kl_beta=1., weight_decay=0.
    )
    bm = RestrictedBoltzmannMachine(
        1, 1, quadratic_coef=torch.tensor([[2.]]),
        linear_bias=torch.tensor([1., -3.]), device="cpu",
    )
    return SuppliedQVAE(2, None, config, nn.Identity(), nn.Identity(), bm, FixedSampler())


def _oracle(logits):
    probabilities = logits.sigmoid()
    energy = -probabilities[:, 0] + 3 * probabilities[:, 1]
    energy = energy - 2 * probabilities[:, 0] * probabilities[:, 1]
    entropy = torch.distributions.Bernoulli(logits=logits).entropy().sum(-1)
    return energy - entropy


def test_kl_surrogate_matches_each_observations_energy():
    model = _model()
    logits = torch.tensor([[1., -2.], [-0.5, 0.25]])
    actual = model._kl_dist_from(FactorialBernoulliUtil(logits))
    torch.testing.assert_close(actual, _oracle(logits))
    singleton = torch.cat([
        model._kl_dist_from(FactorialBernoulliUtil(row[None])) for row in logits
    ])
    torch.testing.assert_close(actual, singleton)


def test_weighted_kl_gradients_do_not_mix_observations():
    model = _model()
    logits = torch.tensor([[1., -2.], [-0.5, 0.25]], requires_grad=True)
    weights = torch.tensor([0.25, 2.])
    actual = model._kl_dist_from(FactorialBernoulliUtil(logits))
    expected = _oracle(logits)
    actual_gradient = torch.autograd.grad((actual * weights).sum(), logits)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_mean_training_objective_and_gradient_are_preserved():
    model = _model()
    logits = torch.tensor([[1., -2.], [-0.5, 0.25]], requires_grad=True)
    actual = model._kl_dist_from(FactorialBernoulliUtil(logits)).mean()
    expected = _oracle(logits).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits)[0],
        torch.autograd.grad(expected, logits)[0],
    )
