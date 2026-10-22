"""condition_sample must honour the requested output dtype."""

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class TwoRowSampler:
    """Returns two all-plus spin rows."""

    def solve(self, ising_mat):
        """Returns two solution rows."""
        return np.ones((2, ising_mat.shape[0]))


def _model():
    return BoltzmannMachine(5, device=torch.device("cpu"))


def test_float64_is_honoured():
    model = _model()
    visible = torch.tensor([[1.0, 0.0]])

    out = model.condition_sample(TwoRowSampler(), visible, dtype=torch.float64)

    assert out.shape == (2, 5)
    assert out.dtype == torch.float64


def test_default_dtype_is_float32():
    model = _model()
    visible = torch.tensor([[1.0, 0.0]])

    out = model.condition_sample(TwoRowSampler(), visible)

    assert out.dtype == torch.float32
