"""Regression test for the visualizer's ``save_pdf`` output directory.

``RBMVisualizer`` stores ``result_dir`` (created in ``__init__``) and every
sibling method writes through it.  ``plot_reconstructed_images`` used to write
to a hardcoded relative ``results/`` path instead, so a custom ``result_dir``
raised ``FileNotFoundError`` (or silently wrote to the wrong directory).
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

from supervised_dbn_digits import RBMVisualizer  # noqa: E402


class _ReconstructableModel:
    """Stand-in exposing the ``reconstruct`` interface the visualizer calls."""

    def reconstruct(self, X, layer_index=0):
        recon = np.clip(X, 0.0, 1.0)
        errors = np.mean((X - recon) ** 2, axis=1)
        return recon, errors


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


def test_save_pdf_uses_the_configured_result_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    x = np.random.RandomState(0).rand(2, 4).astype(np.float32)
    y = np.array([0, 1])

    result_dir = tmp_path / "custom_results"
    visualizer = RBMVisualizer(result_dir=str(result_dir))
    visualizer.plot_reconstructed_images(
        _ReconstructableModel(),
        x,
        y,
        n_images=2,
        img_shape=(2, 2),
        title_suffix="unit",
        save_pdf=True,
    )

    expected = result_dir / "reconstructed_images_unit.pdf"
    assert expected.exists(), "PDF must be written into the configured result_dir"
    assert not (tmp_path / "results").exists(), (
        "the hardcoded 'results/' directory must not be used"
    )
