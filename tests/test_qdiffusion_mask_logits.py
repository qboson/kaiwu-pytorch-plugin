"""_mask_logits must suppress every special token and keep the input intact."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Minimal proposal backbone returning logits over eight tokens."""

    def __init__(self, vocab_size=8, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Returns per-token logits."""
        del kwargs
        return self.lm_head(self.embedding(input_ids))


class Energy(EnergyModel):
    """Energy model stub used only for construction."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns a constant score."""
        del noisy_tokens, candidate_tokens, attention_mask
        return torch.zeros(1, 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=Energy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=False,
    )


def test_every_special_token_is_masked():
    model = _model()
    logits = torch.zeros(2, 5, 8)

    masked = model._mask_logits(logits)

    for token_id in (0, 1, 2, 3, 4):
        assert torch.isinf(masked[..., token_id]).all()
        assert (masked[..., token_id] < 0).all()
    torch.testing.assert_close(masked[..., 5:], logits[..., 5:])


def test_masking_does_not_mutate_the_input():
    model = _model()
    logits = torch.zeros(2, 5, 8)

    model._mask_logits(logits)

    assert torch.count_nonzero(logits) == 0
