"""A token repeated exactly ratio*seq_len times is not resampled."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Proposal backbone favouring token seven."""

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
    """Energy stub."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns zeros."""
        del noisy_tokens, attention_mask
        return torch.zeros(candidate_tokens.size(0), 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ConstantEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(
            num_candidates=2,
            resample_ratio=0.25,
            resample_top_p=1.0,
        ),
        freeze_proposal=False,
    )


def test_top_token_is_resampled_but_exact_threshold_token_is_not():
    model = _model()
    tokens = torch.tensor([[5, 5, 5, 5, 6, 6, 7, 8]])
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

    model._resample(tokens, scores)

    assert tokens.tolist() == [[7, 7, 7, 7, 6, 6, 7, 8]]
    torch.testing.assert_close(scores[0, 4:], torch.tensor([0.5, 0.6, 0.7, 0.8]))
