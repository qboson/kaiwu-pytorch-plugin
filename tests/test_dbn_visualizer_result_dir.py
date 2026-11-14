"""Regression test for RBMVisualizer reconstructed-image output paths.

``plot_reconstructed_images(save_pdf=True)`` hardcoded ``results/...`` instead
of using the configured ``result_dir``, so a custom directory received no PDF
and a fresh working directory crashed with FileNotFoundError.
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
    def reconstruct(self, X, layer_index):
        return X.copy(), np.zeros(X.shape[0], dtype=np.float32)


def test_reconstructed_pdf_uses_the_configured_result_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_dir = tmp_path / "custom-results"
    visualizer = RBMVisualizer(result_dir=str(result_dir))
    X = np.arange(128, dtype=np.float32).reshape(2, 64)
    y = np.array([0, 1])

    visualizer.plot_reconstructed_images(
        _FakeRBM(),
        X,
        y,
        n_images=2,
        title_suffix="unit",
        save_pdf=True,
    )

    saved = result_dir / "reconstructed_images_unit.pdf"
    assert saved.is_file()
    assert not (tmp_path / "results").exists()
