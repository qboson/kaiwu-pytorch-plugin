"""Stable smoothing-mixture log densities and differentiable likelihoods."""

from decimal import Decimal, localcontext

import pytest
import torch

from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


def _reference(rate, logit, value):
    """Evaluate the density and its two derivatives in decimal probability space."""
    with localcontext() as context:
        context.prec = 90
        beta = Decimal.from_float(float(rate))
        parameter = Decimal.from_float(float(logit))
        sample = Decimal.from_float(float(value))
        one = Decimal(1)
        weight_zero = one / (one + parameter.exp())
        weight_one = one / (one + (-parameter).exp())
        normalizer = one - (-beta).exp()
        component_zero = beta * (-beta * sample).exp() / normalizer
        component_one = beta * (-beta * (one - sample)).exp() / normalizer
        density = weight_zero * component_zero + weight_one * component_one
        parameter_derivative = (
            weight_zero * weight_one * (component_one - component_zero) / density
        )
        sample_derivative = (
            beta * (weight_one * component_one - weight_zero * component_zero) / density
        )
        return float(density.ln()), float(parameter_derivative), float(sample_derivative)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("rate", [20.0, 200.0, 1000.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_finite_tail_log_densities_and_gradients_match_decimal_reference(dtype, rate, sign):
    """Representable log densities survive both saturated weights and tiny component PDFs."""
    logits = torch.tensor([sign * 40.0, sign * 1000.0, 0.0, sign * 5.0], dtype=dtype)
    logits.requires_grad_(True)
    values = torch.tensor([0.0 if sign > 0 else 1.0, 0.5, 0.5, 0.125], dtype=dtype)
    values.requires_grad_(True)
    distribution = MixtureGeneric(logits, rate)
    references = [_reference(rate, logit, value) for logit, value in zip(logits.tolist(), values.tolist())]
    expected = torch.tensor(references, dtype=dtype)

    log_density = distribution.log_prob_per_var(values)
    parameter_gradient, value_gradient = torch.autograd.grad(log_density.sum(), (logits, values))

    assert torch.isfinite(log_density).all()
    assert torch.isfinite(parameter_gradient).all()
    assert torch.isfinite(value_gradient).all()
    tolerance = {
        torch.float16: {"rtol": 0.005, "atol": 0.005},
        torch.bfloat16: {"rtol": 0.04, "atol": 0.02},
        torch.float32: {"rtol": 2e-6, "atol": 2e-5},
        torch.float64: {"rtol": 2e-12, "atol": 2e-12},
    }[dtype]
    # The existing smoothing helper stores its normalization constants in
    # float32 even for double samples; this change preserves that convention.
    density_tolerance = {"rtol": 2e-7, "atol": 2e-7} if dtype == torch.float64 else tolerance
    torch.testing.assert_close(log_density, expected[:, 0], **density_tolerance)
    torch.testing.assert_close(parameter_gradient, expected[:, 1], **tolerance)
    torch.testing.assert_close(value_gradient, expected[:, 2], **tolerance)


@pytest.mark.parametrize("rate", [1.0, 2.0, 10.0])
def test_moderate_log_densities_preserve_broadcasting(rate):
    """Shared sample values broadcast over each latent variable's mixture weights."""
    logits = torch.tensor([[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]], dtype=torch.float64)
    values = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    expected = torch.tensor([
        [_reference(rate, logit, value)[0] for logit in row]
        for row, value in zip(logits.tolist(), values.flatten().tolist())
    ], dtype=torch.float64)

    actual = MixtureGeneric(logits, rate).log_prob_per_var(values)

    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-7)


def test_reflecting_sample_and_logit_preserves_log_density():
    """Exchanging the two binary mixture components preserves density values."""
    logits = torch.tensor([40.0, -40.0, 100.0, 0.0])
    values = torch.tensor([0.0, 1.0, 0.125, 0.5])

    original = MixtureGeneric(logits, 200.0).log_prob_per_var(values)
    reflected = MixtureGeneric(-logits, 200.0).log_prob_per_var(1.0 - values)

    assert torch.isfinite(original).all()
    torch.testing.assert_close(original, reflected)


def test_log_density_likelihood_can_train_a_real_encoder():
    """A finite smoothing likelihood produces a useful encoder update at a saturated logit."""
    encoder = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        encoder.weight.fill_(40.0)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
    inputs = torch.ones(3, 1)
    values = torch.zeros(3, 1)
    loss = -MixtureGeneric(encoder(inputs), 200.0).log_prob_per_var(values).mean()

    loss.backward()

    assert torch.isfinite(loss)
    torch.testing.assert_close(encoder.weight.grad, torch.ones_like(encoder.weight))
    optimizer.step()
    updated_loss = -MixtureGeneric(encoder(inputs), 200.0).log_prob_per_var(values).mean()
    assert updated_loss.item() < loss.item()
