"""Repetition resampling must rewrite only the over-represented tokens."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Proposal backbone whose favourite token is seven."""

    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        with torch.no_grad():
            self.lm_head.bias.zero_()
            self.lm_head.bias[7] = 10.0

    def forward(self, input_ids, **kwargs):
        """Returns logits favouring token seven."""
        del kwargs
        return self.lm_head(self.embedding(input_ids))


class ConstantEnergy(EnergyModel):
    """Energy stub returning zeros."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns one zero energy per candidate."""
        del noisy_tokens, attention_mask
        return candidate_tokens.to(torch.float32).mean(dim=1, keepdim=True)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ConstantEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(
            num_candidates=2,
            resample_ratio=0.5,
            resample_top_p=1.0,
        ),
        freeze_proposal=False,
    )


def test_only_over_represented_tokens_are_resampled():
    model = _model()
    tokens = torch.tensor([[5, 5, 5, 5, 6, 6]])
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

    model._resample(tokens, scores)

    torch.testing.assert_close(tokens, torch.tensor([[7, 7, 7, 7, 6, 6]]))
    torch.testing.assert_close(scores[0, 4:], torch.tensor([0.5, 0.6]))
    assert not torch.allclose(scores[0, :4], torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_a_sequence_without_repetition_is_untouched():
    model = _model()
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10]])
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

    model._resample(tokens, scores)

    torch.testing.assert_close(tokens, torch.tensor([[5, 6, 7, 8, 9, 10]]))
    torch.testing.assert_close(scores, torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))
