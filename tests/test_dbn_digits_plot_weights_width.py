"""Regression test for ``RBMVisualizer.plot_weights``.

The helper reshaped each component with ``sqrt(n_visible)`` where ``n_visible``
defaulted to the 8x8 digit width (64).  For any other input width the reshape
failed, so the documented visualiser only worked for one dataset.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from supervised_dbn_digits import RBMVisualizer  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


@pytest.mark.parametrize("num_visible", [16, 64])
def test_plot_weights_infers_visible_width(tmp_path, num_visible):
    torch.manual_seed(0)
    rbm = RestrictedBoltzmannMachine(num_visible=num_visible, num_hidden=4)

    visualizer = RBMVisualizer(result_dir=str(tmp_path))
    # Must not raise: the component layout is derived from the model.
    visualizer.plot_weights(rbm, grid_shape=(2, 2))


def test_plot_weights_still_accepts_an_explicit_width(tmp_path):
    torch.manual_seed(0)
    rbm = RestrictedBoltzmannMachine(num_visible=16, num_hidden=4)

    visualizer = RBMVisualizer(result_dir=str(tmp_path))
    visualizer.plot_weights(rbm, n_visible=16, grid_shape=(2, 2))
