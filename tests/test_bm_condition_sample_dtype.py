# -*- coding: utf-8 -*-
"""condition_sample must accept state tensors in a different dtype or as NumPy."""

from __future__ import annotations

import numpy as np
import torch

from kaiwu.torch_plugin import BoltzmannMachine


class _DummySampler:
    """Return two deterministic spin solutions for each Ising matrix."""

    def solve(self, ising_matrix):
        return np.ones((2, ising_matrix.shape[0]), dtype=np.float32)


def test_double_model_accepts_float32_visible() -> None:
    """A float64 model must not fail on default float32 state tensors."""
    model = BoltzmannMachine(4).double()
    visible = torch.ones(2, 2, dtype=torch.float32)

    samples = model.condition_sample(_DummySampler(), visible)

    assert samples.shape == (4, 4)
    assert samples.dtype == torch.float64


def test_float_model_accepts_float64_visible() -> None:
    """A float32 model must not fail on float64 state tensors either."""
    model = BoltzmannMachine(4)
    visible = torch.ones(2, 2, dtype=torch.float64)

    samples = model.condition_sample(_DummySampler(), visible)

    assert samples.shape == (4, 4)
    assert samples.dtype == torch.float32


def test_numpy_visible_is_converted_to_a_tensor() -> None:
    """NumPy condition arrays are a natural input form for the public API."""
    model = BoltzmannMachine(4)
    visible = np.ones((2, 2), dtype=np.float32)

    samples = model.condition_sample(_DummySampler(), visible)

    assert samples.shape == (4, 4)
    assert samples.dtype == torch.float32
