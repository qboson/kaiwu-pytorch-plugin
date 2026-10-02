"""Regression test for the ``img_shape`` auto-inference in the visualiser.

``RBMVisualizer.plot_reconstructed_images`` documents image-shape
auto-inference (``img_shape`` defaults to ``None``), but only filled the shape
in for perfectly square widths.  Any other width crashed on the first
``img_shape[0]`` access:

    TypeError: 'NoneType' object is not subscriptable

Reachable with the example's own ``[128, 256]`` architecture when visualising a
layer with 128 visible units.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from kaiwu.torch_plugin import UnsupervisedDBN  # noqa: E402
from supervised_dbn_digits import RBMVisualizer  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


def _dbn(num_visible):
    torch.manual_seed(0)
    dbn = UnsupervisedDBN([6])
    dbn.create_rbm_layer(num_visible)
    return dbn


@pytest.mark.parametrize("num_visible", [12, 16, 128])
def test_img_shape_is_inferred_for_any_width(tmp_path, num_visible):
    rng = np.random.RandomState(0)
    x = rng.rand(3, num_visible).astype(np.float32)
    y = np.array([0, 1, 2])

    visualizer = RBMVisualizer(result_dir=str(tmp_path))
    errors = visualizer.plot_reconstructed_images(
        _dbn(num_visible), x, y, n_images=3
    )

    assert errors.shape == (3,)
