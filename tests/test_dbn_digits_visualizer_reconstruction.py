"""Regression test for ``RBMVisualizer.plot_reconstructed_images``.

The visualizer is documented as the RBM result-visualisation helper and its
sibling ``plot_weights`` works on a ``RestrictedBoltzmannMachine`` (it reads
``quadratic_coef`` / ``num_hidden``).  ``plot_reconstructed_images`` used to
call ``rbm.reconstruct(...)``, a method that only exists on
``UnsupervisedDBN``, so the documented flow

    rbm = DBNPretrainer(...).get_rbm_layer(0)
    RBMVisualizer().plot_reconstructed_images(rbm, X, y)

raised ``AttributeError``.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DBN_DIR = os.path.join(REPO_ROOT, "example", "dbn_digits")
sys.path.insert(0, DBN_DIR)

sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from supervised_dbn_digits import RBMVisualizer  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


def _small_rbm():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return RestrictedBoltzmannMachine(num_visible=4, num_hidden=2)


def test_plot_reconstructed_images_accepts_an_rbm(tmp_path):
    rbm = _small_rbm()
    x = np.random.RandomState(0).rand(3, 4).astype(np.float32)
    y = np.array([0, 1, 2])

    visualizer = RBMVisualizer(result_dir=str(tmp_path))
    errors = visualizer.plot_reconstructed_images(
        rbm,
        x,
        y,
        n_images=3,
        img_shape=(2, 2),
        save_pdf=False,
    )

    assert errors.shape == (3,)
    assert np.isfinite(errors).all()
