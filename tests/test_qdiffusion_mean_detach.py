"""Objective energy means must be detached summaries."""

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

    def __init__(self, vocab_size=8, hidden_size=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        """Returns per-token logits."""
        del kwargs
        return self.lm_head(self.embedding(input_ids))


class ScalableEnergy(EnergyModel):
    """Energy model with a trainable scale so scores carry gradients."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns the candidate mean times a trainable scale."""
        del noisy_tokens, attention_mask
        return candidate_tokens.to(torch.float32).mean(dim=1, keepdim=True) * self.scale


def _model():
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=ScalableEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(num_diffusion_timesteps=8, num_candidates=2),
        freeze_proposal=False,
    )


def test_energy_means_are_detached_but_the_objective_keeps_grad():
    torch.manual_seed(0)
    model = _model()
    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])

    outputs = model.objective({"targets": targets})

    for key in ("positive_energy_mean", "negative_energy_mean"):
        value = outputs[key]
        assert value.ndim == 0
        assert value.requires_grad is False
    assert outputs["energy_objective"].requires_grad is True
