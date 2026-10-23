"""Image-grid rendering contract for the QVAE-MNIST helpers."""

import importlib.util
import os

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location(
    "qvae_mnist_helpers_grid",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "utils", "helpers.py"),
)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)


def features(rows, width=784):
    """Builds a feature matrix with the requested shape."""
    return torch.rand(rows, width)


def test_grid_requires_flat_28x28_features():
    """Wrong width or too few rows fail before any rendering."""
    with pytest.raises(AssertionError):
        helpers.plot_flattened_images_grid(features(4, width=100), grid_size=2)
    with pytest.raises(AssertionError):
        helpers.plot_flattened_images_grid(features(3), grid_size=2)


def test_grid_is_saved_into_the_requested_nested_path(tmp_path):
    """A valid feature matrix writes the requested PNG, creating parents."""
    target = tmp_path / "nested" / "grid.png"
    helpers.plot_flattened_images_grid(features(4), grid_size=2, save_path=str(target))
    assert target.is_file()
    assert target.stat().st_size > 0
