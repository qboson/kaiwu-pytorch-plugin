"""Regression tests for plot_MNIST_output figure ownership.

The helper lives in the qvae_mnist example, whose module imports optional
example-only dependencies.  Stub those when unavailable so the core test
environment can still exercise the figure lifecycle.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


if importlib.util.find_spec("gif") is None:
    _stub("gif", frame=lambda function: function)
if importlib.util.find_spec("imageio") is None:
    _stub("imageio")
if importlib.util.find_spec("torchvision") is None:
    transforms = _stub("torchvision.transforms")
    _stub("torchvision", transforms=transforms)
if importlib.util.find_spec("torchmetrics") is None:
    fid = _stub("torchmetrics.image.fid", FrechetInceptionDistance=object)
    image = _stub("torchmetrics.image", fid=fid)
    _stub("torchmetrics", image=image)

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example" / "qvae_mnist"
sys.path.insert(0, str(EXAMPLE_DIR))

from utils.helpers import plot_MNIST_output  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def test_plot_mnist_output_closes_its_figure_and_keeps_caller_figure():
    caller = plt.figure()
    caller_number = caller.number
    x_true = np.zeros((2, 28 * 28), dtype=np.float32)
    x_recon = np.ones((2, 28 * 28), dtype=np.float32)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "reconstruction.png"
        plot_MNIST_output(x_true, x_recon, n_samples=2, output=str(output))

        assert output.is_file()
        assert output.stat().st_size > 0
        pixels = np.asarray(Image.open(output).convert("RGB"))
        assert np.array_equal(pixels[..., 0], pixels[..., 1])
        assert np.array_equal(pixels[..., 1], pixels[..., 2])

    assert plt.get_fignums() == [caller_number]


def test_plot_mnist_output_closes_its_figure_when_saving_fails():
    caller = plt.figure()
    caller_number = caller.number
    x_true = np.zeros((2, 28 * 28), dtype=np.float32)
    x_recon = np.zeros((2, 28 * 27), dtype=np.float32)

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "reconstruction.png"
        with pytest.raises(ValueError):
            plot_MNIST_output(x_true, x_recon, n_samples=2, output=str(output))

    assert plt.get_fignums() == [caller_number]
