"""condition_sample must validate the visible state shape."""

import numpy as np
import pytest
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class TwoRowSampler:
    """Returns two all-plus spin rows."""

    def solve(self, ising_mat):
        """Returns two solution rows."""
        return np.ones((2, ising_mat.shape[0]))


def _model():
    return BoltzmannMachine(5, device=torch.device("cpu"))


def test_one_dimensional_visible_state_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        _model().condition_sample(TwoRowSampler(), torch.ones(2))


def test_width_beyond_the_node_count_is_rejected():
    with pytest.raises(ValueError, match="num_nodes"):
        _model().condition_sample(TwoRowSampler(), torch.ones(1, 6))


def test_a_valid_visible_state_still_samples():
    out = _model().condition_sample(TwoRowSampler(), torch.tensor([[1.0, 0.0]]))

    assert out.shape == (2, 5)
