# -*- coding: utf-8 -*-
"""RBM state propagation must accept mixed-dtype and NumPy inputs."""

from __future__ import annotations

import numpy as np
import torch

from kaiwu.torch_plugin import RestrictedBoltzmannMachine


def test_get_hidden_double_model_accepts_float32_visible() -> None:
    """A float64 model must not fail on default float32 visible states."""
    rbm = RestrictedBoltzmannMachine(4, 3).double()
    visible = torch.randn(2, 4, dtype=torch.float32)

    result = rbm.get_hidden(visible)

    assert result.shape == (2, 7)
    assert torch.isfinite(result).all()


def test_get_hidden_float_model_accepts_float64_visible() -> None:
    """A float32 model must not fail on float64 visible states either."""
    rbm = RestrictedBoltzmannMachine(4, 3)
    visible = torch.randn(2, 4, dtype=torch.float64)

    result = rbm.get_hidden(visible)

    assert result.shape == (2, 7)
    assert torch.isfinite(result).all()


def test_get_visible_double_model_accepts_float32_hidden() -> None:
    """The reverse propagation has the same input contract."""
    rbm = RestrictedBoltzmannMachine(4, 3).double()
    hidden = torch.randn(2, 3, dtype=torch.float32)

    result = rbm.get_visible(hidden)

    assert result.shape == (2, 7)
    assert torch.isfinite(result).all()


def test_get_hidden_accepts_numpy_visible() -> None:
    """NumPy condition arrays are converted instead of failing at .size()."""
    rbm = RestrictedBoltzmannMachine(4, 3)
    visible = np.ones((2, 4), dtype=np.float32)

    result = rbm.get_hidden(visible)

    assert result.shape == (2, 7)
    assert torch.isfinite(result).all()
