"""generate must keep partial-mask positions fixed for the whole loop."""

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
            num_diffusion_timesteps=8,
            num_candidates=2,
            proposal_temperature=0.0,
            disable_resample=True,
        ),
        freeze_proposal=False,
    )


def test_fixed_positions_survive_the_full_loop():
    torch.manual_seed(0)
    model = _model()
    tokens = torch.tensor([[1, 5, 9, 2, 0]])
    partial = torch.tensor([[False, False, True, False, False]])

    out = model.generate(tokens, max_steps=2, partial_masks=partial)

    assert out.tolist() == [[1, 7, 9, 2, 0]]
