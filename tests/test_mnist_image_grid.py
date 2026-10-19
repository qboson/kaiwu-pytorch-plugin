"""Image grids remain two-dimensional even when they contain one image."""

import importlib.util
from pathlib import Path

import pytest
import torch

# The MNIST helper module imports these optional example dependencies eagerly.
for dependency in ("matplotlib", "pandas", "sklearn", "torchvision", "torchmetrics", "gif", "imageio"):
    pytest.importorskip(dependency)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@pytest.fixture
def helpers():
    path = Path(__file__).resolve().parents[1] / "example/qvae_mnist/utils/helpers.py"
    spec = importlib.util.spec_from_file_location("mnist_grid_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    plt.close("all")


@pytest.mark.parametrize("grid_size", [1, 2])
def test_image_grid_saves_each_image_in_row_major_order(helpers, tmp_path, monkeypatch, grid_size):
    features = torch.arange(grid_size ** 2 * 784, dtype=torch.float32).reshape(-1, 784)
    features.requires_grad_()
    figures = []
    subplots = plt.subplots

    def record_figure(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", record_figure)
    output = tmp_path / "images" / "grid.png"
    helpers.plot_flattened_images_grid(features, grid_size=grid_size, save_path=str(output))

    assert len(figures[0].axes) == grid_size ** 2
    for index, axis in enumerate(figures[0].axes):
        np.testing.assert_array_equal(axis.images[0].get_array(), features[index].detach().reshape(28, 28))
    with Image.open(output) as image:
        assert image.format == "PNG"
        image.verify()
    assert figures[0].number not in plt.get_fignums()
