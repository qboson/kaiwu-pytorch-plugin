"""Regression test for ``RBMRunner.gen_digits_image``.

``fit(plot_img=True)`` displays ``rbm.sample(sampler)[:20]``.  A legal sampler
may return fewer than 20 rows (the Kaiwu simulated annealer merges and
deduplicates its solutions and *may* return fewer than ``size_limit``), so the
image helper must not assume a fixed batch size.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "rbm_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

from rbm_digits import RBMRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture()
def runner():
    return RBMRunner(n_components=8, n_iter=1)


@pytest.mark.parametrize("n_samples", [1, 2, 7, 20])
def test_gen_digits_image_accepts_any_sample_count(runner, n_samples):
    data = np.random.RandomState(0).rand(n_samples, 64).astype(np.float32)
    image = runner.gen_digits_image(data, 8)
    assert image.shape == (8, n_samples * 8)


def test_gen_digits_image_preserves_sample_order(runner):
    data = np.zeros((2, 64), dtype=np.float32)
    data[1] = 1.0
    image = runner.gen_digits_image(data, 8)
    np.testing.assert_allclose(image[:, :8], 0.0)
    np.testing.assert_allclose(image[:, 8:], 1.0)
