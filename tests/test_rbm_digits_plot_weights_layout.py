"""Regression test for ``RBMRunner.plot_weights`` component layout.

The helper reshaped each component with a hardcoded 8x8 layout, so
``plot_weights`` only worked for the bundled 8x8 digits dataset.  Any other
visible width crashed:

    ValueError: cannot reshape array of size 16 into shape (8,8)

The layout must follow the model's visible width.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "rbm_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402
from rbm_digits import RBMRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


@pytest.fixture()
def runner():
    instance = RBMRunner(n_components=8, n_iter=1, verbose=False)
    return instance


@pytest.mark.parametrize("num_visible", [16, 64])
def test_plot_weights_follows_the_visible_width(runner, num_visible):
    torch.manual_seed(0)
    runner.rbm = RestrictedBoltzmannMachine(num_visible, 8)

    runner.plot_weights(save_pdf=False)  # must not raise


def test_non_square_width_reports_a_clear_error(runner):
    torch.manual_seed(0)
    runner.rbm = RestrictedBoltzmannMachine(20, 8)

    with pytest.raises(ValueError) as excinfo:
        runner.plot_weights(save_pdf=False)

    message = str(excinfo.value)
    assert "20" in message and "square" in message
