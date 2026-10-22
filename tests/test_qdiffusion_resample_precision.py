"""Real CPU generation must preserve the score dtype through resampling."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]


class ConfidentProposal(nn.Module):
    """A real embedding and dense head with a known ordinary-token preference."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)
        self.calls = []
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[5] = 12.0

    def forward(self, tokens):
        logits = self.head(self.embedding(tokens))
        self.calls.append((tokens.detach().clone(), logits.dtype))
        return logits


class PooledEnergy(EnergyModel):
    """A dense energy scorer operating at the same precision as the proposal."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 1)

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del noisy_tokens
        pooled = (self.embedding(candidate_tokens) * attention_mask.unsqueeze(-1)).sum(1)
        return self.head(pooled)


def make_generator(dtype, **config_kwargs):
    """Construct actual CPU proposal and energy parameters in the requested dtype."""
    torch.manual_seed(0)
    proposal = ConfidentProposal()
    model = QDiffusion(
        proposal,
        PooledEnergy(),
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2, x_id=4),
        QDiffusionConfig(**config_kwargs),
        device="cpu",
        dtype=dtype,
    )
    return model, proposal


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("disable_resample", [False, True])
def test_public_generate_keeps_score_precision_with_real_proposal(dtype, disable_resample):
    """Default repetition resampling must work for every supported proposal dtype."""
    model, proposal = make_generator(dtype, disable_resample=disable_resample)
    targets = torch.full((1, 4), 5, dtype=torch.long)
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    state = model.generate(targets, max_steps=1, return_state=True)

    torch.testing.assert_close(state["output_tokens"], targets)
    assert state["output_scores"].dtype == torch.float32
    assert torch.isfinite(state["output_scores"]).all()
    assert state["output_scores"].shape == targets.shape
    assert len(proposal.calls) == (1 if disable_resample else 2)
    for tokens, logits_dtype in proposal.calls:
        torch.testing.assert_close(tokens, torch.full_like(targets, 3))
        assert logits_dtype == dtype
    for parameter, original in zip(model.parameters(), original_parameters):
        assert parameter.dtype == dtype
        torch.testing.assert_close(parameter, original)
    if not disable_resample:
        # The configured nucleus keeps only token 5, whose log probability is zero.
        torch.testing.assert_close(state["output_scores"], torch.zeros_like(state["output_scores"]))


@pytest.mark.parametrize("dtype", DTYPES)
def test_repetition_threshold_still_controls_second_proposal_call(dtype):
    """A frequency equal to the threshold does not trigger repetition resampling."""
    model, proposal = make_generator(dtype, resample_ratio=1.0)
    targets = torch.full((1, 4), 5, dtype=torch.long)

    state = model.generate(targets, max_steps=1, return_state=True)

    torch.testing.assert_close(state["output_tokens"], targets)
    assert len(proposal.calls) == 1
    assert state["output_scores"].dtype == torch.float32
    assert torch.isfinite(state["output_scores"]).all()
