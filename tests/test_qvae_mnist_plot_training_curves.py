"""Training-curve plotting must create its directory and honour show=False."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("gif")
pytest.importorskip("imageio")
pytest.importorskip("torchvision")
pytest.importorskip("torchmetrics")


ROOT = Path(__file__).resolve().parents[1]


def load_helpers():
    spec = importlib.util.spec_from_file_location(
        "qvae_mnist_helpers_curves",
        ROOT / "example" / "qvae_mnist" / "utils" / "helpers.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakePyplot:
    def __init__(self):
        self.show_calls = 0

    def figure(self, **kwargs):
        return object()

    def subplot(self, *args, **kwargs):
        return object()

    def plot(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def xlabel(self, *args, **kwargs):
        return None

    def ylabel(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def tight_layout(self, *args, **kwargs):
        return None

    def savefig(self, path, **kwargs):
        assert Path(path).parent.exists(), path

    def show(self):
        self.show_calls += 1

    def close(self):
        return None


def _call_plot(helpers, fake_plt, save_path, show):
    helpers.plot_training_curves(
        [1.0, 0.5],
        [1.2, 0.6],
        [50.0, 80.0],
        [45.0, 75.0],
        save_path=str(save_path),
        show=show,
    )


def test_plot_training_curves_honours_show_false(monkeypatch, tmp_path):
    helpers = load_helpers()
    fake_plt = _FakePyplot()
    monkeypatch.setattr(helpers, "plt", fake_plt)

    _call_plot(helpers, fake_plt, tmp_path / "curves.png", show=False)

    assert fake_plt.show_calls == 0


def test_plot_training_curves_creates_missing_directory(monkeypatch, tmp_path):
    helpers = load_helpers()
    fake_plt = _FakePyplot()
    monkeypatch.setattr(helpers, "plt", fake_plt)
    save_path = tmp_path / "nested" / "curves.png"

    _call_plot(helpers, fake_plt, save_path, show=False)

    assert save_path.parent.exists()
