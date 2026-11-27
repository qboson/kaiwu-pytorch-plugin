"""Regression coverage for representable implicit mixture-logit gradients."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from kaiwu.classical import BruteForceOptimizer
from kaiwu.torch_plugin import QVAE, RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae_dist_util import (
    Exponential, FactorialBernoulliUtil, MixtureGeneric,
)


def implicit_cdf_gradient(logits, samples):
    """Differentiate H(z, l)=constant in float64, independently of the carrier."""
    logits, samples = logits.double(), samples.double()
    beta = 10.0
    normalizer = -torch.expm1(torch.tensor(-beta, dtype=torch.float64))
    pdf_zero = beta * torch.exp(-beta * samples) / normalizer
    pdf_one = beta * torch.exp(-beta * (1 - samples)) / normalizer
    cdf_zero = -torch.expm1(-beta * samples) / normalizer
    cdf_one = 1 + torch.expm1(-beta * (1 - samples)) / normalizer
    probability = torch.sigmoid(logits)
    sigmoid_slope = torch.exp(
        -torch.nn.functional.softplus(logits)
        - torch.nn.functional.softplus(-logits)
    )
    return ((cdf_zero - cdf_one) * sigmoid_slope
            / ((1 - probability) * pdf_zero + probability * pdf_one))


@pytest.mark.parametrize("magnitude", [-20.0, 0.0, 5.0, 20.0, 30.0])
@pytest.mark.parametrize("binary_state", [0.0, 1.0])
def test_float32_gradient_matches_implicit_cdf(magnitude, binary_state):
    """Interior fixed samples isolate the logit derivative without sampling error."""
    logits = torch.full((3,), magnitude, requires_grad=True)
    smoothed_zero = torch.tensor([0.1, 0.2, 0.3])
    with patch.object(Exponential, "sample", return_value=smoothed_zero), \
            patch.object(FactorialBernoulliUtil, "reparameterize",
                         return_value=torch.full_like(logits, binary_state)):
        samples = MixtureGeneric(logits, 10.0).reparameterize(True)
    samples.sum().backward()
    expected = implicit_cdf_gradient(logits.detach(), samples.detach())
    torch.testing.assert_close(logits.grad.double(), expected, rtol=0.003, atol=1e-28)


def test_bfloat16_confident_logits_retain_the_representable_derivative():
    """A confident, finite BF16 posterior must not lose its entire gradient."""
    logits = torch.full((3,), 7.0, dtype=torch.bfloat16, requires_grad=True)
    # These interior points avoid claiming a repair of existing BF16 tail arithmetic.
    with patch.object(Exponential, "sample", return_value=torch.tensor([0.1, 0.2, 0.3])), \
            patch.object(FactorialBernoulliUtil, "reparameterize",
                         return_value=torch.ones_like(logits)):
        samples = MixtureGeneric(logits, 10.0).reparameterize(True)
    samples.sum().backward()
    assert torch.isfinite(logits.grad).all()
    expected = implicit_cdf_gradient(logits.detach(), samples.detach())
    torch.testing.assert_close(logits.grad.double(), expected, rtol=0.01, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("is_training", [False, True])
def test_forward_samples_and_rng_preserve_ancestral_sampling(dtype, is_training):
    """The zero-forward carrier must consume no randomness or change samples."""
    logits = torch.tensor(
        [-20.0, -7.0, 0.0, 5.0, 7.0, 20.0], dtype=dtype, requires_grad=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        bits = FactorialBernoulliUtil(logits).reparameterize(False)
        base_sample = Exponential(10.0).sample(bits.shape).to(bits.device)
        expected = torch.where(bits == 0, base_sample, 1 - base_sample)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(71)
        actual = MixtureGeneric(logits, 10.0).reparameterize(is_training)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.get_rng_state(), expected_rng)


def test_real_cpu_autocast_qvae_loss_updates_encoder():
    """Exercise the public QVAE loss and an actual local SDK optimizer."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        encoder, decoder = nn.Linear(2, 2), nn.Linear(2, 2)
        with torch.no_grad():
            encoder.weight.fill_(3.5)
            encoder.bias.zero_()
            decoder.weight.copy_(torch.eye(2))
            decoder.bias.zero_()
        config = SimpleNamespace(
            num_latent_units=2, loss_type="mse", dist_beta=10.0,
            kl_beta=0.0, weight_decay=0.0,
        )
        model = QVAE(
            2, None, config, encoder=encoder, decoder=decoder,
            bm=RestrictedBoltzmannMachine(1, 1), sampler=BruteForceOptimizer(),
        )
        inputs = torch.ones(32, 2)
        initial = encoder.weight.detach().clone()
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.manual_seed(19)
            reconstruction, posterior, logits, _ = model(inputs)
            loss = model.loss(inputs, reconstruction, posterior)
        assert logits.dtype == torch.bfloat16
        assert torch.all(logits == 7.0)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(encoder.weight.grad).all()
        assert encoder.weight.grad.abs().sum() > 0
        optimizer.step()
        assert (encoder.weight - initial).abs().sum() > 0
