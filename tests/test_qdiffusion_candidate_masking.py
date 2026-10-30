"""QDiffusion candidate sampling must exclude reserved tokens."""

import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)

SPECIAL_IDS = {0, 1, 2, 3, 4}


class SpecialTokenProposal(nn.Module):
    """Proposal that ranks every reserved token above the real vocabulary."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, **kwargs):
        batch, length = tokens.shape
        logits = torch.full((batch, length, 8), -100.0)
        logits[..., 0] = 100.0
        logits[..., 1] = 90.0
        logits[..., 2] = 80.0
        logits[..., 3] = 70.0
        logits[..., 4] = 60.0
        logits[..., 5:] = 0.0
        return logits


class RecordingEnergy(EnergyModel):
    """Records every candidate batch and attention mask it receives."""

    def __init__(self):
        super().__init__()
        self.candidate_batches = []
        self.attention_masks = []

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        self.candidate_batches.append(candidate_tokens.clone())
        self.attention_masks.append(attention_mask.clone())
        return candidate_tokens.float().sum(dim=1, keepdim=True)


def make_model():
    spec = SequenceTokenSpec(
        mask_id=3,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        x_id=4,
    )
    config = QDiffusionConfig(
        num_diffusion_timesteps=8,
        num_candidates=1,
        proposal_temperature=0.0,
        proposal_noise_scale=0.0,
        disable_resample=True,
    )
    energy = RecordingEnergy()
    model = QDiffusion(
        SpecialTokenProposal(),
        energy,
        spec,
        config,
        freeze_proposal=False,
    )
    return model, energy


def test_sample_candidates_excludes_reserved_tokens():
    """Direct candidate sampling must mask pad/bos/eos/mask/x logits."""
    model, _ = make_model()
    batch, length = 1, 5
    logits = torch.full((batch, length, 8), -100.0)
    logits[..., 0] = 100.0
    logits[..., 1] = 90.0
    logits[..., 2] = 80.0
    logits[..., 3] = 70.0
    logits[..., 4] = 60.0
    logits[..., 5:] = 0.0

    tokens, _ = model._sample_candidates(logits, num_candidates=1)

    generated = set(tokens.flatten().tolist())
    assert generated.isdisjoint(SPECIAL_IDS)


def test_objective_negative_candidates_exclude_reserved_tokens():
    """Training negatives must use the same protected vocabulary as decoding."""
    model, energy = make_model()
    targets = torch.tensor([[1, 5, 6, 7, 2]])

    model.objective({"targets": targets})

    negative_candidates = energy.candidate_batches[-1]
    negative_mask = energy.attention_masks[-1]
    generated = set(negative_candidates.flatten().tolist())
    assert generated.isdisjoint(SPECIAL_IDS)
    assert bool(negative_mask.any())