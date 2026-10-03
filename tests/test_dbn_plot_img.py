"""Regression tests for DBN pretraining visualisation shapes.

The pretraining visualiser reshaped each sample with the hardcoded 8x8 digits
layout, so recording samples for any layer whose width is not 64 crashed with
`ValueError: cannot reshape array of size 320 into shape (20,8,8)`.

The tests drive the real `DBNTrainer`/`UnsupervisedDBN` code; only the
licence-gated Kaiwu solver is replaced by a stub that honours the documented
`solve(ising_matrix) -> (n_solutions, num_nodes + 1)` contract.
"""

import os
import sys

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")
pytest.importorskip("kaiwu")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../example/dbn_digits"))
)
from dbn_trainer import DBNTrainer  # noqa: E402  pylint: disable=wrong-import-position
from kaiwu.torch_plugin.dbn import UnsupervisedDBN  # noqa: E402  pylint: disable=wrong-import-position


class _StubSampler:
    """Legal Ising solutions without the license-gated SDK solver."""

    def __init__(self, num_samples=20):
        self.num_samples = num_samples

    def solve(self, ising_mat):
        num_spins = ising_mat.shape[0] - 1
        rng = np.random.default_rng(0)
        return rng.choice([-1.0, 1.0], size=(self.num_samples, num_spins + 1))


@pytest.mark.parametrize("num_features", [16, 64, 128, 784])
def test_generated_images_use_the_layer_width(num_features):
    """The stacked image must tile the layer's full width for every layout."""
    trainer = DBNTrainer(
        n_epochs_rbm=1, batch_size=4, verbose=False, plot_img=False, random_state=0
    )

    image = trainer._gen_digits_image(  # pylint: disable=protected-access
        np.zeros((5, num_features))
    )

    digits, remainder = divmod(image.shape[1], 5)
    assert remainder == 0, "expected five horizontally stacked samples"
    assert image.shape[0] * digits == num_features
    assert image.shape[0] > 0


def test_plot_img_does_not_crash_for_non_square_layers():
    """plot_img=True must work for layers that are not 8x8 images."""
    rng = np.random.default_rng(0)
    data = rng.random((8, 16)).astype(np.float32)
    trainer = DBNTrainer(
        n_epochs_rbm=1, batch_size=4, verbose=True, plot_img=True, random_state=0
    )
    trainer.sampler = _StubSampler()
    dbn = UnsupervisedDBN([8, 4])
    dbn.create_rbm_layer(data.shape[1])

    trainer.train(dbn, data)

    assert dbn.num_layers == 2
