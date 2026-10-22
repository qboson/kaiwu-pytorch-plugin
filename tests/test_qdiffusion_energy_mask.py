"""QDiffusion.energy must derive and forward the attention mask."""

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


class MaskEnergy(EnergyModel):
    """Records the attention mask and returns zeros."""

    def __init__(self):
        super().__init__()
        self.masks = []

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Stores the mask and returns one zero per candidate."""
        del noisy_tokens
        self.masks.append(attention_mask.clone())
        return torch.zeros(candidate_tokens.size(0), 1)


def _model(energy):
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=energy,
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=False,
    )


def test_default_mask_marks_non_pad_tokens():
    energy = MaskEnergy()
    model = _model(energy)
    noisy = torch.tensor([[5, 6, 7, 8]])
    candidate = torch.tensor([[0, 5, 3, 7]])

    model.energy(noisy, candidate)

    assert energy.masks[-1].tolist() == [[False, True, True, True]]


def test_an_explicit_mask_is_forwarded_unchanged():
    energy = MaskEnergy()
    model = _model(energy)
    noisy = torch.tensor([[5, 6, 7, 8]])
    candidate = torch.tensor([[0, 5, 3, 7]])
    explicit = torch.tensor([[True, False, False, True]])

    model.energy(noisy, candidate, attention_mask=explicit)

    assert energy.masks[-1].tolist() == explicit.tolist()
