"""Regression tests for QDiffusion energy-temperature handling."""

import pytest
import torch
from torch import nn

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel,
    QDiffusion,
    QDiffusionConfig,
    SequenceTokenSpec,
)


class _DummyProposal(nn.Module):
    def __init__(self, vocab=8, dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


class _TokenEnergyModel(EnergyModel):
    """Scores a candidate by the value of its first token."""

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        return candidate_tokens[:, 0].float().unsqueeze(-1)


def _build_model(energy_temperature):
    config = QDiffusionConfig(
        num_candidates=3,
        proposal_temperature=1.0,
        energy_temperature=energy_temperature,
    )
    spec = SequenceTokenSpec(mask_id=1, pad_id=0, bos_id=2, eos_id=3)
    return QDiffusion(_DummyProposal(), _TokenEnergyModel(), spec, config=config)


def test_zero_energy_temperature_selects_lowest_energy_candidate():
    model = _build_model(energy_temperature=0.0)
    noisy_tokens = torch.tensor([[1, 4]])
    candidate_tokens = torch.tensor([[[5, 5], [7, 7], [3, 3]]])
    candidate_scores = torch.zeros(1, 3, 2)

    tokens, _ = model._select_candidates(  # pylint: disable=protected-access
        noisy_tokens, candidate_tokens, candidate_scores
    )

    assert tokens.tolist() == [[3, 3]]


def test_negative_energy_temperature_is_rejected():
    model = _build_model(energy_temperature=-1.0)
    with pytest.raises(ValueError, match="energy_temperature"):
        model._select_candidates(  # pylint: disable=protected-access
            torch.tensor([[1, 4]]),
            torch.tensor([[[5, 5], [7, 7], [3, 3]]]),
            torch.zeros(1, 3, 2),
        )


def test_zero_energy_temperature_generate_does_not_crash():
    model = _build_model(energy_temperature=0.0)
    input_tokens = torch.full((2, 4), 1, dtype=torch.long)

    output = model.generate(input_tokens, max_steps=3)

    assert output.shape == input_tokens.shape
