"""Regression tests for QDiffusion.objective proposal gradients."""

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


class _ZeroEnergyModel(EnergyModel):
    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        return torch.zeros(candidate_tokens.size(0), 1)


def _build_model(freeze_proposal):
    spec = SequenceTokenSpec(mask_id=1, pad_id=0, bos_id=2, eos_id=3)
    return QDiffusion(
        _DummyProposal(),
        _ZeroEnergyModel(),
        spec,
        config=QDiffusionConfig(num_candidates=2),
        freeze_proposal=freeze_proposal,
    )


def _batch():
    return {"targets": torch.randint(4, 8, (2, 4))}


def test_objective_exposes_trainable_logits_for_unfrozen_proposal():
    model = _build_model(freeze_proposal=False)

    outputs = model.objective(_batch())

    assert outputs["logits"].requires_grad
    outputs["logits"].sum().backward()
    assert model.proposal_model.head.weight.grad is not None


def test_objective_keeps_frozen_proposal_detached():
    model = _build_model(freeze_proposal=True)

    outputs = model.objective(_batch())

    assert not outputs["logits"].requires_grad
