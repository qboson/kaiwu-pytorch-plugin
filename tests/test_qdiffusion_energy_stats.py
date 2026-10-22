"""EnergyModel diagnostics must be a copy of the latest score call."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin.qdiffusion import EnergyModel


class DummySampler:
    """Returns one all-plus spin row for any Ising matrix."""

    def solve(self, ising_mat):
        """Returns a single solution row."""
        return np.ones((1, ising_mat.shape[0]))


def _model():
    return EnergyModel(bm_num_visible=2, bm_num_hidden=3, sampler=DummySampler())


def test_stats_are_empty_before_the_first_score():
    assert _model().get_last_stats() == {}


def test_stats_describe_the_latest_score_call():
    model = _model()
    model.score_visible_logits(torch.zeros(1, 2))
    first = model.get_last_stats()

    model.score_visible_logits(torch.full((1, 2), 100.0))
    second = model.get_last_stats()

    assert set(first) == {"sampling_mode", "visible_on_ratio", "hidden_on_ratio"}
    assert float(first["visible_on_ratio"]) == pytest.approx(0.5, abs=1e-6)
    assert float(first["hidden_on_ratio"]) == pytest.approx(1.0, abs=1e-6)
    assert float(second["visible_on_ratio"]) > 0.99


def test_stats_are_returned_as_a_copy():
    model = _model()
    model.score_visible_logits(torch.zeros(1, 2))

    stats = model.get_last_stats()
    stats.clear()

    assert set(model.get_last_stats()) == {"sampling_mode", "visible_on_ratio", "hidden_on_ratio"}
