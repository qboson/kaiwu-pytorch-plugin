"""A unique proposal needs no energy reranking or external sampling."""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from kaiwu.torch_plugin.qdiffusion import (
    EnergyModel, QDiffusion, QDiffusionConfig, SequenceTokenSpec,
)


class Proposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, tokens):
        logits = torch.zeros(*tokens.shape, 8, device=tokens.device)
        logits[..., 5] = 100.
        return logits


class Energy(EnergyModel):
    def __init__(self, forbid=False):
        super().__init__()
        self.forbid = forbid
        self.calls = 0

    def score_conditioned(self, noisy_tokens, candidate_tokens, attention_mask):
        self.calls += 1
        if self.forbid:
            raise RuntimeError("external energy scoring must not be needed")
        return (candidate_tokens[:, :1].float() - 5.) * 1000.


def make_model(energy, count=1):
    return QDiffusion(
        Proposal(), energy,
        SequenceTokenSpec(mask_id=3, pad_id=0, bos_id=1, eos_id=2),
        QDiffusionConfig(num_candidates=count, disable_resample=True, proposal_noise_scale=0.),
    )


def test_single_candidate_generation_needs_no_energy_backend():
    energy = Energy(forbid=True)
    model = make_model(energy)
    result = model.generate(torch.tensor([[1, 3, 3, 2]]), max_steps=3)
    assert result.tolist() == [[1, 5, 5, 2]]
    assert energy.calls == 0


def test_unique_candidate_keeps_scores_and_score_gradients():
    model = make_model(Energy(forbid=True))
    tokens = torch.tensor([[[5, 6]], [[6, 5]]])
    scores = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]]], requires_grad=True)
    selected, selected_scores = model._select_candidates(tokens[:, 0], tokens, scores)
    assert selected.tolist() == [[5, 6], [6, 5]]
    torch.testing.assert_close(selected_scores, scores[:, 0])
    selected_scores.sum().backward()
    torch.testing.assert_close(scores.grad, torch.ones_like(scores))


def test_multiple_candidates_still_use_energy_reranking():
    energy = Energy()
    model = make_model(energy, count=2)
    tokens = torch.tensor([[[6, 6], [5, 7]], [[5, 6], [6, 7]]])
    scores = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    selected, selected_scores = model._select_candidates(tokens[:, 0], tokens, scores)
    assert energy.calls == 1
    assert selected.tolist() == [[5, 7], [5, 6]]
    assert selected_scores.tolist() == [[3., 4.], [5., 6.]]
