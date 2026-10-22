"""Zero-noise categorical sampling and mixed-precision training regressions."""

import math

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin import _qdiffusion_sampling as sampling
from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


def sample_candidates(logits, temperature, noise_scale, multiple):
    """Exercise both single and multiple noisy categorical entry points."""
    if multiple:
        return sampling.stochastic_sample_from_categorical_n(
            logits, temperature=temperature, noise_scale=noise_scale, n=3
        )
    return sampling.stochastic_sample_from_categorical(
        logits, temperature=temperature, noise_scale=noise_scale
    )


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_zero_noise_matches_unperturbed_distribution(monkeypatch, multiple, temperature, dtype):
    """A legal uniform endpoint cannot affect disabled-noise sampling."""
    logits = torch.tensor([[[-math.inf, 0.0, 5.0, -2.0]]], dtype=dtype)
    original = logits.clone()
    monkeypatch.setattr(sampling.torch, "rand_like", lambda data: torch.zeros_like(data))
    reference_logits = logits.unsqueeze(0).expand(3, *logits.shape) if multiple else logits
    torch.manual_seed(17)
    if temperature:
        distribution = torch.distributions.Categorical(logits=reference_logits / temperature)
        expected_tokens = distribution.sample()
        expected_scores = distribution.log_prob(expected_tokens)
    else:
        expected_scores, expected_tokens = reference_logits.log_softmax(-1).max(-1)
    torch.manual_seed(17)

    tokens, scores = sample_candidates(logits, temperature, 0.0, multiple)

    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(logits, original)
    assert scores.dtype == dtype
    assert scores.device == logits.device
    assert torch.isfinite(scores).all()
    assert tokens.ne(0).all()


@pytest.mark.parametrize("multiple", [False, True])
def test_zero_noise_scores_keep_logit_gradients(multiple):
    """Bypassing disabled noise preserves differentiable log-probability scores."""
    logits = torch.tensor([[[-1.0, 0.0, 2.0]]], dtype=torch.float64, requires_grad=True)

    tokens, scores = sample_candidates(logits, 0.0, 0.0, multiple)
    actual_gradient = torch.autograd.grad(scores.sum(), logits)[0]

    expected_gradient = -logits.detach().softmax(-1)
    expected_gradient[..., 2] += 1.0
    if multiple:
        expected_gradient *= 3.0
    assert tokens.eq(2).all()
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("multiple", [False, True])
def test_positive_noise_still_perturbs_candidates(monkeypatch, multiple):
    """The nonzero-noise branch retains its Gumbel perturbation semantics."""
    logits = torch.tensor([[[0.2, 0.1, 0.0]]])
    uniform = torch.tensor([0.1, 0.9, 0.2])
    monkeypatch.setattr(
        sampling.torch, "rand_like", lambda data: uniform.expand_as(data)
    )
    quantiles = torch.tensor([-math.log(-math.log(value)) for value in uniform.tolist()])
    reference_logits = logits + quantiles
    if multiple:
        reference_logits = reference_logits.unsqueeze(0).expand(3, *logits.shape)
    expected_scores, expected_tokens = reference_logits.log_softmax(-1).max(-1)

    tokens, scores = sample_candidates(logits, 0.0, 1.0, multiple)

    assert tokens.eq(1).all()
    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(scores, expected_scores)


class HalfProposal(nn.Module):
    """Return half-precision logits with one clearly preferred normal token."""

    def __init__(self):
        super().__init__()
        logits = torch.zeros(128, dtype=torch.float16)
        logits[5] = 10.0
        self.register_buffer("logits", logits)

    def forward(self, tokens):
        return self.logits.expand(*tokens.shape, -1)


class RecordingEnergy(EnergyModel):
    """Trainable offline energy model recording positive and negative candidates."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.02))
        self.candidates = []

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del noisy_tokens, attention_mask
        self.candidates.append(candidate_tokens.detach().clone())
        return candidate_tokens.float().sum(-1, keepdim=True) * self.weight


def test_half_precision_objective_zero_noise_keeps_forbidden_tokens_excluded():
    """Real CPU RNG and the public objective retain valid negative training samples."""
    energy = RecordingEnergy()
    model = QDiffusion(
        HalfProposal(),
        energy,
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        QDiffusionConfig(
            num_candidates=2,
            proposal_temperature=0.0,
            proposal_noise_scale=0.0,
            disable_resample=True,
        ),
    )
    targets = torch.full((4, 8), 5, dtype=torch.long)
    torch.manual_seed(0)

    outputs = model.objective({"targets": targets})

    assert outputs["logits"].dtype == torch.float16
    assert outputs["logits"].device.type == "cpu"
    assert energy.candidates[1].eq(5).all()
    torch.testing.assert_close(outputs["negative_energy_mean"], torch.tensor(0.8))
    outputs["energy_objective"].mean().backward()
    expected_gradient = torch.tensor(40.0 * math.tanh(0.4))
    torch.testing.assert_close(energy.weight.grad, expected_gradient)
