"""EnergyModel must accept an empty batch instead of failing inside torch.cat."""

import numpy as np
import torch

from kaiwu.torch_plugin import EnergyModel


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_mat):
        return np.ones((2, ising_mat.shape[0]), dtype=np.float32)


def _model() -> EnergyModel:
    return EnergyModel(bm_num_visible=2, bm_num_hidden=2, sampler=_Sampler())


def test_score_visible_logits_empty_batch_returns_empty_scores() -> None:
    model = _model()
    energy = model.score_visible_logits(torch.zeros(0, 2))
    assert energy.shape == (0, 1)
    assert model.get_last_stats() == {}


def test_sample_hidden_state_empty_batch_returns_empty_states() -> None:
    model = _model()
    full_states, split_sizes = model.sample_hidden_state(torch.zeros(0, 2))
    assert full_states.shape == (0, 4)
    assert split_sizes.shape == (0,)
    assert split_sizes.dtype == torch.long


def test_non_empty_batch_still_scores() -> None:
    model = _model()
    energy = model.score_visible_logits(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]))
    assert energy.shape == (2, 1)
    assert set(model.get_last_stats()) == {
        "sampling_mode",
        "visible_on_ratio",
        "hidden_on_ratio",
    }
