# -*- coding: utf-8 -*-
"""QDiffusion rejects invalid generation settings at construction time."""

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


def _build(config: QDiffusionConfig) -> QDiffusion:
    return QDiffusion(
        proposal_model=_Proposal(),
        energy_model=_Energy(),
        token_spec=SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2),
        config=config,
        freeze_proposal=False,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"resample_ratio": -0.1},
        {"resample_ratio": 1.1},
        {"resample_top_p": 0.0},
        {"resample_top_p": 1.1},
        {"proposal_noise_scale": -0.1},
        {"decoding_strategy": "bad"},
        {"decoding_strategy": "reparam-cond-deterministic"},
        {"decoding_strategy": "reparam-other-deterministic-linear"},
        {"decoding_strategy": "reparam-cond-unknown-linear"},
        {"decoding_strategy": "reparam-cond-deterministic-cubic"},
    ],
)
def test_invalid_generation_config_is_rejected(override) -> None:
    """Invalid thresholds or strategy fields fail before decoding starts."""
    config = QDiffusionConfig(**override)

    with pytest.raises(ValueError):
        _build(config)


def test_valid_generation_config_is_accepted() -> None:
    """The default and a stochastic/cosine strategy remain valid."""
    _build(QDiffusionConfig())
    _build(
        QDiffusionConfig(
            resample_ratio=0.0,
            resample_top_p=1.0,
            proposal_noise_scale=0.5,
            decoding_strategy="reparam-cond-stochastic0.5-cosine",
        )
    )
