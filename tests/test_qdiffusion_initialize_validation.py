"""initialize_state must reject malformed inputs and negative budgets."""

import pytest
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

    def __init__(self, vocab_size=8, hidden_size=8):
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
        del noisy_tokens, attention_mask
        return torch.zeros(candidate_tokens.size(0), 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ConstantEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=False,
    )


def test_a_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        _model().initialize_state(torch.tensor([1, 5, 6]))


def test_a_negative_step_budget_is_rejected():
    with pytest.raises(ValueError, match="max_steps"):
        _model().initialize_state(torch.tensor([[1, 5, 6]]), max_steps=-1)


def test_a_zero_step_budget_is_allowed():
    state = _model().initialize_state(torch.tensor([[1, 5, 6]]), max_steps=0)

    assert state["max_steps"] == 0
    assert state["step"] == 0
