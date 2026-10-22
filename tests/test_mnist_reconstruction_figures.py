"""Reconstruction PNG generation does not retain pyplot figures."""
import ast
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest


def plotter():
    path = Path(__file__).resolve().parents[1] / 'example/qvae_mnist/utils/helpers.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plot_MNIST_output')
    namespace = {'plt': plt}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['plot_MNIST_output']


def test_repeated_saves_keep_only_callers_figures(tmp_path):
    plt.close('all')
    caller = plt.figure()
    images = np.arange(784, dtype=float).reshape(1, 784) / 784
    seen = []
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        seen.append(figure)
        return original(figure, *args, **kwargs)

    try:
        with patch.object(Figure, 'savefig', save):
            for index in range(3):
                path = tmp_path / f'recon-{index}.png'
                plotter()(images, 1 - images, n_samples=1, output=path)
                assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
                assert plt.get_fignums() == [caller.number]
        for figure in seen:
            assert len(figure.axes) == 2
            np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), images.reshape(28, 28))
            np.testing.assert_array_equal(figure.axes[1].images[0].get_array(), (1 - images).reshape(28, 28))
    finally:
        plt.close('all')


@pytest.mark.parametrize('failure', ['save', 'image'])
def test_error_releases_only_owned_figure(tmp_path, failure):
    plt.close('all')
    caller = plt.figure()
    images = np.zeros((1, 784))
    try:
        if failure == 'save':
            with patch.object(Figure, 'savefig', side_effect=OSError('disk full')):
                with pytest.raises(OSError, match='disk full'):
                    plotter()(images, images, n_samples=1, output=tmp_path / 'unused.png')
        else:
            with pytest.raises(ValueError):
                plotter()(images[:, :3], images, n_samples=1, output=tmp_path / 'unused.png')
        assert plt.get_fignums() == [caller.number]
    finally:
        plt.close('all')
