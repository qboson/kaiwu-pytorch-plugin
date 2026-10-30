"""sample_hidden_state must receive exactly bm_num_visible visible units."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import EnergyModel

NUM_VISIBLE = 3
NUM_HIDDEN = 2


class _Sampler:  # pylint: disable=too-few-public-methods
    def solve(self, ising_matrix):
        return np.ones((1, ising_matrix.shape[0]), dtype=np.float32)


def _model() -> EnergyModel:
    return EnergyModel(
        bm_num_visible=NUM_VISIBLE, bm_num_hidden=NUM_HIDDEN, sampler=_Sampler()
    )


@pytest.mark.parametrize("width", [1, NUM_VISIBLE - 1, NUM_VISIBLE + 1])
def test_wrong_visible_width_is_rejected(width: int) -> None:
    with pytest.raises(ValueError, match="visible_state must have shape"):
        _model().sample_hidden_state(torch.rand(2, width))


def test_one_dimensional_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="visible_state must have shape"):
        _model().sample_hidden_state(torch.rand(NUM_VISIBLE))


def test_correct_width_returns_full_states() -> None:
    states, sizes = _model().sample_hidden_state(torch.rand(2, NUM_VISIBLE))
    assert states.shape == (2, NUM_VISIBLE + NUM_HIDDEN)
    assert sizes.tolist() == [1, 1]
