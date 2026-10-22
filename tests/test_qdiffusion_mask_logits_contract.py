"""Pin the special-token masking applied before sampling."""

import math

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class _Tokenizer:  # pylint: disable=too-few-public-methods
    def batch_decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(str(int(t)) for t in row) for row in tokens]


class _Proposal(nn.Module):
    def __init__(self, vocab=8, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return self.head(self.embedding(input_ids))


class _Energy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def _model() -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(
            pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4, tokenizer=_Tokenizer()
        ),
        config=QDiffusionConfig(num_diffusion_timesteps=4, disable_resample=True),
        freeze_proposal=False,
    )


def test_special_token_logits_are_masked_to_negative_infinity() -> None:
    model = _model()
    logits = torch.zeros(2, 5, 8)
    masked = model._mask_logits(logits)  # pylint: disable=protected-access

    for token_id in (0, 1, 2, 3, 4):
        assert torch.all(masked[..., token_id] == -math.inf)
    others = [index for index in range(8) if index not in (0, 1, 2, 3, 4)]
    assert torch.all(masked[..., others] == 0.0)
    assert torch.all(logits == 0.0)


def test_masked_copy_leaves_the_input_untouched() -> None:
    model = _model()
    logits = torch.randn(1, 3, 8)
    original = logits.clone()
    model._mask_logits(logits)  # pylint: disable=protected-access
    torch.testing.assert_close(logits, original)
