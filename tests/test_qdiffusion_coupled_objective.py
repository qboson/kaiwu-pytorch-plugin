"""Coupled sampling must double the training batch consistently."""

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


class DummyEnergy(EnergyModel):
    """Tiny energy model."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        """Returns one score per candidate."""
        del attention_mask
        return candidate_tokens.to(torch.float32).mean(dim=1, keepdim=True) - noisy_tokens.to(
            torch.float32
        ).mean(dim=1, keepdim=True)


def _model(use_coupled):
    spec = SequenceTokenSpec(pad_id=0, bos_id=1, eos_id=2, mask_id=3, x_id=4)
    return QDiffusion(
        proposal_model=Proposal(),
        energy_model=DummyEnergy(),
        token_spec=spec,
        config=QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=2,
            use_coupled_sampling=use_coupled,
        ),
        freeze_proposal=False,
    )


def test_coupled_objective_repeats_targets_for_both_steps():
    torch.manual_seed(0)
    model = _model(True)
    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])

    outputs = model.objective({"targets": targets})

    assert outputs["logits"].shape == (4, 5, 8)
    assert outputs["targets"].shape == (4, 5)
    torch.testing.assert_close(outputs["targets"], targets.repeat(2, 1))
    assert outputs["weight"].shape == (4, 1)
    assert outputs["loss_mask"].shape == (4, 5)


def test_single_step_objective_keeps_the_batch():
    torch.manual_seed(0)
    model = _model(False)
    targets = torch.tensor([[1, 5, 6, 2, 0], [1, 6, 5, 2, 0]])

    outputs = model.objective({"targets": targets})

    assert outputs["targets"].shape == (2, 5)
    assert outputs["logits"].shape == (2, 5, 8)
