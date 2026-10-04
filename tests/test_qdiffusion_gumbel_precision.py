"""Positive-noise sampling must preserve masked support in half precision."""

import pytest
import torch

from kaiwu.torch_plugin import _qdiffusion_sampling as sampling


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_positive_noise_preserves_single_token_support(
    monkeypatch, multiple, temperature, dtype
):
    """A legal zero uniform draw cannot invalidate the only allowed token."""
    logits = torch.tensor([[[-torch.inf, -torch.inf, 0.0]]], dtype=dtype)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.zeros_like(tensor))
    if multiple:
        tokens, scores = sampling.stochastic_sample_from_categorical_n(
            logits, temperature=temperature, noise_scale=1.0, n=3
        )
    else:
        tokens, scores = sampling.stochastic_sample_from_categorical(
            logits, temperature=temperature, noise_scale=1.0
        )
    assert tokens.eq(2).all()
    assert torch.isfinite(scores).all()
    assert scores.dtype == dtype
    torch.testing.assert_close(scores, torch.zeros_like(scores))


def test_positive_noise_real_half_precision_random_draws():
    """Actual seeded uniform draws reproduce the forbidden-token regression."""
    with torch.random.fork_rng():
        torch.manual_seed(0)
        logits = torch.full((20000, 1, 4), -torch.inf, dtype=torch.float16)
        logits[..., 2] = 0.0
        tokens, scores = sampling.stochastic_sample_from_categorical(
            logits, temperature=0.0, noise_scale=1.0
        )
    assert tokens.eq(2).all()
    assert torch.isfinite(scores).all()


def test_stochastic_remasking_keeps_protected_positions(monkeypatch):
    """Zero draws cannot turn protected +inf scores into NaNs."""
    scores = torch.tensor([[torch.inf, 1.0, 2.0]], dtype=torch.float16)
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.zeros_like(tensor))
    mask = sampling.topk_masking(scores, torch.tensor([[1]]), stochastic=True)
    assert torch.equal(mask, torch.tensor([[False, True, False]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_precision_sampling_keeps_existing_results(dtype):
    """Promotion of half precision must leave existing full-precision draws intact."""
    logits = torch.tensor([[[-1.0, 0.0, 2.0]]], dtype=dtype, requires_grad=True)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        uniform = torch.rand_like(logits)
        noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
        expected_scores, expected_tokens = (logits + noise).log_softmax(-1).max(-1)
        torch.manual_seed(42)
        tokens, scores = sampling.stochastic_sample_from_categorical(
            logits, temperature=0.0, noise_scale=1.0
        )
    torch.testing.assert_close(tokens, expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(scores.sum(), logits)[0]
    expected_gradient = torch.autograd.grad(expected_scores.sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
