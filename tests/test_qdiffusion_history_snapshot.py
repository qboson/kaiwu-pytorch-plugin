"""The decode history must store snapshots, not the live token tensor."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class Proposal(nn.Module):
    """Minimal proposal backbone favouring token seven."""

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
        config=QDiffusionConfig(num_candidates=2, disable_resample=True),
        freeze_proposal=False,
    )


def test_history_entry_is_independent_of_later_mutations():
    model = _model()
    tokens = torch.tensor([[0, 3, 5, 2]])
    state = {
        "output_tokens": tokens,
        "output_scores": torch.zeros_like(tokens, dtype=torch.float32),
        "history": [],
    }

    out = model._decode_step(state)
    snapshot = out["history"][-1].clone()

    out["output_tokens"][0, 0] = 9

    torch.testing.assert_close(out["history"][-1], snapshot)
    assert out["history"][-1][0, 0].item() == 0
