"""Conditioned-feature contract for the DPLM BM reranker."""

import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qdiffusion", "dplm"))

from models.bm import BMConditionedEnergyModel  # noqa: E402


class FakeEncoder:
    """Expands each token into a constant hidden vector."""

    hidden_size = 4

    def __init__(self):
        self.calls = []

    def encode_tokens(self, tokens, attention_mask=None):
        self.calls.append((tokens, attention_mask))
        return tokens.float().unsqueeze(-1).expand(-1, -1, 4)


class DummySampler:
    """Placeholder sampler; the tested methods never sample."""

    def solve(self, ising_mat):
        raise AssertionError("sampler must not run in these tests")


def build_model():
    """Builds a 2-visible/1-hidden reranker over the fake encoder."""
    encoder = FakeEncoder()
    model = BMConditionedEnergyModel(
        encoder=encoder,
        bm_num_visible=2,
        bm_num_hidden=1,
        sampler=DummySampler(),
    )
    return model, encoder


def test_features_concatenate_noisy_before_candidate():
    """The pooled noisy feature is the first half of the concatenation."""
    model, encoder = build_model()
    noisy = torch.tensor([[1, 2, 0]])
    candidate = torch.tensor([[3, 4, 0]])
    mask = torch.tensor([[1, 1, 0]])
    features = model.build_conditioned_features(noisy, candidate, mask)
    assert features.shape == (1, 8)
    assert features.tolist() == [[1.5, 1.5, 1.5, 1.5, 3.5, 3.5, 3.5, 3.5]]
    assert [call[0] for call in encoder.calls] == [noisy, candidate]
    assert all(call[1] is mask for call in encoder.calls)


def test_visible_logits_are_the_projected_conditioned_features():
    """``build_visible_logits`` only adds the linear projection."""
    model, _encoder = build_model()
    noisy = torch.tensor([[1, 2, 0]])
    candidate = torch.tensor([[3, 4, 0]])
    mask = torch.tensor([[1, 1, 0]])
    features = model.build_conditioned_features(noisy, candidate, mask)
    expected = model.feature_projector(features)
    logits = model.build_visible_logits(noisy, candidate, mask)
    assert logits.shape == (1, 2)
    assert torch.equal(logits, expected)


def test_score_conditioned_delegates_to_the_visible_scorer(monkeypatch):
    """Scoring hands the projected logits to ``score_visible_logits``."""
    model, _encoder = build_model()
    recorded = {}

    def fake_score(visible_logits):
        recorded["logits"] = visible_logits
        return torch.tensor([[0.25]])

    monkeypatch.setattr(model, "score_visible_logits", fake_score)
    noisy = torch.tensor([[1, 2, 0]])
    candidate = torch.tensor([[3, 4, 0]])
    mask = torch.tensor([[1, 1, 0]])
    scores = model.score_conditioned(noisy, candidate, mask)
    expected = model.build_visible_logits(noisy, candidate, mask)
    assert torch.equal(recorded["logits"], expected)
    assert scores.tolist() == [[0.25]]
