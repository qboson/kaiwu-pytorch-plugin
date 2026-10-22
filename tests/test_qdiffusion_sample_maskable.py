"""Training corruption must only mask positions inside maskable_mask."""

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


def test_full_timestep_masks_only_the_maskable_positions():
    model = _model()
    clean = torch.tensor([[5, 6, 7, 8, 9]])
    maskable = torch.tensor([[True, True, False, False, False]])

    out = model._sample(clean, torch.tensor([8]), maskable)

    assert out["x_t"].tolist() == [[3, 3, 7, 8, 9]]
    assert out["loss_mask"].tolist() == [[True, True, False, False, False]]


def test_zeroth_timestep_keeps_everything_clean():
    model = _model()
    clean = torch.tensor([[5, 6, 7, 8, 9]])
    maskable = torch.tensor([[True, True, True, True, True]])

    out = model._sample(clean, torch.tensor([0]), maskable)

    assert out["x_t"].tolist() == clean.tolist()
    assert not bool(out["loss_mask"].any())
