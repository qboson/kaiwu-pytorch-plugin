"""EnergyModel must reject a sampler that returns no solutions."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import EnergyModel


class _EmptySampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((0, ising_matrix.shape[0]), dtype=np.float32)


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


def _model(sampler) -> EnergyModel:
    return EnergyModel(bm_num_visible=2, bm_num_hidden=2, sampler=sampler)


def test_empty_sampler_is_rejected_by_hidden_state_sampling() -> None:
    with pytest.raises(RuntimeError, match="sampler returned no solutions"):
        _model(_EmptySampler()).sample_hidden_state(torch.zeros(2, 2))


def test_empty_sampler_is_rejected_by_scoring() -> None:
    with pytest.raises(RuntimeError, match="sampler returned no solutions"):
        _model(_EmptySampler()).score_visible_logits(torch.zeros(2, 2))


def test_normal_sampler_still_scores() -> None:
    scores = _model(_Sampler()).score_visible_logits(torch.zeros(2, 2))
    assert scores.shape == (2, 1)
    assert torch.isfinite(scores).all()
