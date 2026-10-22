"""Coupled corruption must keep the second mask inside the first one."""

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
        config=QDiffusionConfig(num_candidates=2, num_diffusion_timesteps=8),
        freeze_proposal=False,
    )


def test_the_second_corruption_is_a_subset_of_the_first():
    model = _model()
    clean = torch.tensor([[5, 6, 7, 8]])
    maskable = torch.ones(1, 4, dtype=torch.bool)

    out = model._sample_coupled(
        clean,
        torch.tensor([8.0]),
        torch.tensor([0.0]),
        maskable,
    )

    assert out["x_t"].tolist() == [[3, 3, 3, 3], [5, 6, 7, 8]]
    assert out["t"].tolist() == [8.0, 0.0]
    assert out["loss_mask"].tolist() == [
        [True, True, True, True],
        [False, False, False, False],
    ]
