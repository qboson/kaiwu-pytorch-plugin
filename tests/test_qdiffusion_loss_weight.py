"""The timestep weighting must follow the documented linear/constant formulas."""

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


class Energy(EnergyModel):
    """Energy model stub used only for construction."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns a constant score."""
        del noisy_tokens, candidate_tokens, attention_mask
        return torch.zeros(1, 1)


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=Energy(),
        token_spec=spec,
        config=QDiffusionConfig(num_diffusion_timesteps=8, num_candidates=2),
        freeze_proposal=False,
    )


def test_linear_weight_counts_down_to_one_over_timesteps():
    model = _model()

    weight = model._compute_loss_weight(torch.tensor([1, 4]), "linear")

    assert weight.shape == (2, 1)
    torch.testing.assert_close(weight[:, 0], torch.tensor([1.0, 0.625]))


def test_constant_weight_is_one_for_every_timestep():
    model = _model()

    weight = model._compute_loss_weight(torch.tensor([1, 4]), "constant")

    torch.testing.assert_close(weight[:, 0], torch.tensor([1.0, 1.0]))
