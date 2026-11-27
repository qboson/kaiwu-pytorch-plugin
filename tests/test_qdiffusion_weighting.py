# -*- coding: utf-8 -*-
"""QDiffusion objective rejects unknown timestep weighting strategies."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class _Proposal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 8)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return self.embedding(input_ids)


class _Energy(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        del noisy_tokens, attention_mask
        return candidate_tokens.to(torch.float32).sum(dim=1, keepdim=True)


def _model() -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2),
        config=QDiffusionConfig(
            num_diffusion_timesteps=8,
            num_candidates=1,
            proposal_temperature=0.0,
            disable_resample=True,
        ),
        freeze_proposal=False,
    )


def test_unknown_weighting_raises_value_error() -> None:
    targets = torch.tensor([[1, 5, 6, 2], [1, 6, 5, 2]])
    model = _model()

    with pytest.raises(ValueError, match="Unsupported timestep weighting"):
        model.objective({"targets": targets}, weighting="quadratic")


@pytest.mark.parametrize("weighting", ["linear", "constant"])
def test_supported_weighting_runs(weighting: str) -> None:
    targets = torch.tensor([[1, 5, 6, 2], [1, 6, 5, 2]])
    model = _model()

    outputs = model.objective({"targets": targets}, weighting=weighting)

    assert outputs["weight"].shape == (2, 1)
    assert torch.isfinite(outputs["weight"]).all()
