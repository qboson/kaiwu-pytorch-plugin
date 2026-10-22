"""initialize_state must expose the documented state payload."""

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


def test_state_fields_and_history_snapshot():
    model = _model()
    tokens = torch.tensor([[1, 5, 6, 2, 0]])

    state = model.initialize_state(tokens, max_steps=7, temperature=0.3)

    assert state["step"] == 0
    assert state["max_steps"] == 7
    assert state["temperature"] == 0.3
    assert state["output_masks"].dtype == torch.bool
    assert len(state["history"]) == 1
    snapshot = state["history"][0].clone()
    state["output_tokens"][0, 1] = 7
    torch.testing.assert_close(state["history"][0], snapshot)
