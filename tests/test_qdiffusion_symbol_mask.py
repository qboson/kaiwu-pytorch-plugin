"""Only pad/bos/eos positions are non-editable; partial masks stay fixed."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Minimal proposal backbone."""

    def __init__(self, vocab_size=16, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Returns per-token logits."""
        del kwargs
        return self.lm_head(self.embedding(input_ids))


class ConstantEnergy(EnergyModel):
    """Energy stub."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns zeros."""
        del noisy_tokens, candidate_tokens, attention_mask
        return torch.zeros(1, 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ConstantEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=False,
    )


def test_pad_bos_eos_are_not_editable():
    model = _model()
    tokens = torch.tensor([[0, 1, 2, 3, 5, 4]])

    mask = model.get_non_special_symbol_mask(tokens)

    assert mask.tolist() == [[False, False, False, True, True, True]]


def test_partial_masks_are_removed_from_the_editable_set():
    model = _model()
    tokens = torch.tensor([[0, 1, 2, 3, 5, 4]])
    partial = torch.tensor([[False, True, False, False, True, False]])

    mask = model.get_non_special_symbol_mask(tokens, partial_masks=partial)

    assert mask.tolist() == [[False, False, False, True, False, True]]
