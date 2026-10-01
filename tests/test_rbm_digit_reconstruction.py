"""Digit reconstruction plots must condition on each supplied image."""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest
import torch


class RecordedSpinSampler:
    """Provide legal negative states for actual training; no equilibrium claim."""

    def __init__(self, **kwargs):
        self.calls = 0

    def solve(self, matrix):
        self.calls += 1
        bits = np.array([[0] * (len(matrix) - 1), [1] * (len(matrix) - 1)])
        return np.column_stack((2 * bits - 1, np.ones(2, dtype=int)))


@pytest.fixture
def fitted_runner(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "example/rbm_digits/rbm_digits.py"
    spec = importlib.util.spec_from_file_location("actual_rbm_reconstruction_example", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == source
    monkeypatch.setattr(module, "SimulatedAnnealingOptimizer", RecordedSpinSampler)
    monkeypatch.setattr(plt, "show", lambda: None)
    runner = module.RBMRunner(n_components=2, n_iter=1, batch_size=8,
                              learning_rate=0, verbose=False, plot_img=False)
    images = torch.stack((torch.zeros(64), torch.ones(64),
                          torch.arange(64).remainder(2).float(),
                          torch.arange(64).lt(32).float(),
                          torch.arange(64).ge(32).float()))
    runner.fit(images.numpy())
    assert runner.sampler.calls == 1
    assert runner.rbm.quadratic_coef.grad is not None
    with torch.no_grad():
        runner.rbm.quadratic_coef.copy_(torch.stack((
            torch.linspace(-.12, .18, 64), torch.linspace(.16, -.09, 64)), dim=1))
        runner.rbm.linear_bias.copy_(torch.cat((torch.linspace(-.4, .3, 64),
                                               torch.tensor([.2, -.35]))))
    yield runner, images
    plt.close("all")


def independent_mean_field(images, rbm):
    """NumPy arithmetic for visible->hidden->visible conditional probabilities."""
    binary = images.double().numpy() > .5
    weights = rbm.quadratic_coef.detach().cpu().double().numpy()
    biases = rbm.linear_bias.detach().cpu().double().numpy()
    hidden = 1 / (1 + np.exp(-(binary @ weights + biases[64:])))
    return 1 / (1 + np.exp(-(hidden @ weights.T + biases[:64])))


def plotted_rows():
    axes = plt.gcf().axes
    return [np.asarray(axis.images[0].get_array()).reshape(-1) for axis in axes]


@pytest.mark.parametrize("count", [1, 2, 5])
def test_actual_plot_matches_each_input_conditional_reconstruction(fitted_runner, tmp_path, count):
    runner, images = fitted_runner
    images = images[:count]
    before = {name: value.clone() for name, value in runner.rbm.state_dict().items()}
    before_calls = runner.sampler.calls

    runner.plot_images(images, torch.arange(count), save_pdf=False)

    rows = plotted_rows()
    assert len(rows) == 2 * count
    np.testing.assert_array_equal(np.stack(rows[:count]), images.numpy())
    np.testing.assert_allclose(np.stack(rows[count:]), independent_mean_field(images, runner.rbm),
                               atol=1e-7, rtol=1e-6)
    assert runner.sampler.calls == before_calls
    assert all(torch.equal(value, before[name]) for name, value in runner.rbm.state_dict().items())
    assert [axis.get_title() for axis in plt.gcf().axes[:count]] == [
        f"Label: {label}" for label in range(count)]
    path = tmp_path / f"conditional_reconstruction_{count}.png"
    plt.gcf().savefig(path)
    with Image.open(path) as image:
        image.verify()


def test_plot_reconstruction_follows_input_permutation_and_differs_between_images(fitted_runner):
    runner, images = fitted_runner
    selected = images[:2]
    runner.plot_images(selected, [10, 20])
    original = np.stack(plotted_rows()[2:])
    runner.plot_images(selected.flip(0), [20, 10])
    permuted = np.stack(plotted_rows()[2:])
    assert not np.allclose(original[0], original[1])
    np.testing.assert_array_equal(permuted, original[::-1])


def test_unfitted_plot_keeps_clear_failure(fitted_runner):
    runner, images = fitted_runner
    runner.rbm = None
    with pytest.raises(ValueError, match="fit first"):
        runner.plot_images(images[:1], [0])
