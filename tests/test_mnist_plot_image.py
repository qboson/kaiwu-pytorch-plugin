"""The single-image plotting helper creates its own mappable and coordinates."""
import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np
import pytest


def plotter():
    path = Path(__file__).resolve().parents[1] / 'example/qvae_mnist/utils/helpers.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plot_image')
    namespace = {'plt': plt}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['plot_image']


@pytest.mark.parametrize('layer,shape', [(0, (3, 24)), (1, (5, 4))])
def test_image_colorbar_and_shape_ticks(layer, shape, tmp_path):
    plt.close('all')
    pixels = np.arange(np.prod(shape), dtype=float).reshape(shape)
    before = pixels.copy()
    try:
        im = plotter()(pixels, layer, vmin=-1, vmax=100)
        assert isinstance(im, AxesImage)
        np.testing.assert_array_equal(im.get_array(), pixels)
        np.testing.assert_array_equal(pixels, before)
        assert im.get_clim() == (-1, 100)
        np.testing.assert_array_equal(im.axes.get_xticks(), np.arange(0, shape[1], 10 if layer == 0 else 1))
        np.testing.assert_array_equal(im.axes.get_yticks(), np.arange(shape[0]))
        assert im.colorbar.mappable is im
        assert im.colorbar.ax.get_ylabel() == 'Energy (MeV)'
        # Render a real artifact without unrelated optional model/FID imports.
        im.figure.set_size_inches(4, 3)
        target = tmp_path / 'image.png'
        im.figure.savefig(target)
        assert target.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    finally:
        plt.close('all')


def test_default_color_limits_come_from_image():
    try:
        image = np.array([[-3., 0.], [1., 7.]])
        im = plotter()(image, 2)
        assert im.get_clim() == (-3., 7.)
    finally:
        plt.close('all')
