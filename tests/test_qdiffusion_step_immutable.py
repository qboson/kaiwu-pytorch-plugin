"""step must return a new state and leave the input untouched."""

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
        config=QDiffusionConfig(num_candidates=2, disable_resample=True),
        freeze_proposal=False,
    )


def test_step_does_not_mutate_the_input_state():
    torch.manual_seed(0)
    model = _model()
    state = model.initialize_state(torch.tensor([[1, 5, 6, 2, 0]]), max_steps=4)
    tokens_before = state["output_tokens"].clone()

    next_state = model.step(state)

    assert next_state is not state
    assert state["step"] == 0
    assert next_state["step"] == 1
    torch.testing.assert_close(state["output_tokens"], tokens_before)
    assert len(state["history"]) == 1
    assert len(next_state["history"]) == 2
