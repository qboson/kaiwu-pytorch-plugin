"""Regression test for RBMVisualizer.plot_reconstructed_images layout inference.

The helper only inferred an image shape for perfect-square feature counts. For
a 12-feature layer, ``img_shape`` stayed None and the plotting code raised
``TypeError: 'NoneType' object is not subscriptable`` instead of drawing a
rectangular image.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

pytest.importorskip("kaiwu")

from supervised_dbn_digits import RBMVisualizer  # noqa: E402


class _FakeRBM:
    """Minimal reconstruction source for the visualizer."""

    def reconstruct(self, X, layer_index):
        return X.copy(), np.zeros(X.shape[0], dtype=np.float32)


def test_non_square_feature_count_infers_a_rectangular_layout(tmp_path):
    visualizer = RBMVisualizer(result_dir=str(tmp_path / "results"))
    X = np.arange(24, dtype=np.float32).reshape(2, 12)
    y = np.array([0, 1])

    # Must not raise: base main leaves img_shape=None for 12 features.
    visualizer.plot_reconstructed_images(
        _FakeRBM(), X, y, n_images=2, save_pdf=False
    )


def test_square_feature_count_still_works(tmp_path):
    visualizer = RBMVisualizer(result_dir=str(tmp_path / "results"))
    X = np.arange(128, dtype=np.float32).reshape(2, 64)
    y = np.array([0, 1])

    visualizer.plot_reconstructed_images(
        _FakeRBM(), X, y, n_images=2, save_pdf=False
    )
