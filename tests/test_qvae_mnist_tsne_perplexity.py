"""Tiny test sets must not crash t-SNE on the fixed perplexity of 30."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("gif")
pytest.importorskip("imageio")
pytest.importorskip("torchvision")
pytest.importorskip("torchmetrics")


ROOT = Path(__file__).resolve().parents[1]


def load_helpers():
    spec = importlib.util.spec_from_file_location(
        "qvae_mnist_helpers", ROOT / "example" / "qvae_mnist" / "utils" / "helpers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "n_samples, expected",
    [(2, 1.0), (10, 9.0), (31, 30.0), (100, 30.0)],
)
def test_perplexity_stays_below_sample_count(n_samples, expected):
    assert load_helpers()._tsne_perplexity(n_samples) == expected


@pytest.mark.parametrize("n_samples", [0, 1])
def test_too_few_samples_are_rejected(n_samples):
    with pytest.raises(ValueError, match="at least two samples"):
        load_helpers()._tsne_perplexity(n_samples)


class _FakeTSNE:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, values):
        return np.zeros((len(values), 2))


class _FakeAxis:
    def scatter(self, *args, **kwargs):
        return object()


class _FakePyplot:
    def __init__(self):
        self.show_calls = 0

    def subplots(self, **kwargs):
        return object(), _FakeAxis()

    def colorbar(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        pass

    def xlabel(self, *args, **kwargs):
        pass

    def ylabel(self, *args, **kwargs):
        pass

    def tight_layout(self, *args, **kwargs):
        pass

    def savefig(self, path, **kwargs):
        assert Path(path).parent.exists(), path

    def show(self):
        self.show_calls += 1

    def close(self):
        pass


class _FakeModel:
    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, x):
        return None, None, None, torch.zeros(x.size(0), 3)


def _loader():
    return [(torch.zeros(4, 2), torch.tensor([0, 1, 0, 1]))]


def test_t_sne_honours_show_false(monkeypatch, tmp_path):
    helpers = load_helpers()
    fake_plt = _FakePyplot()
    monkeypatch.setattr(helpers, "TSNE", _FakeTSNE)
    monkeypatch.setattr(helpers, "plt", fake_plt)

    helpers.t_SNE(
        _loader(),
        _FakeModel(),
        epochs=3,
        save_path=str(tmp_path / "plot.png"),
        show=False,
    )

    assert fake_plt.show_calls == 0


def test_t_sne_creates_missing_output_directory(monkeypatch, tmp_path):
    helpers = load_helpers()
    fake_plt = _FakePyplot()
    monkeypatch.setattr(helpers, "TSNE", _FakeTSNE)
    monkeypatch.setattr(helpers, "plt", fake_plt)
    save_path = tmp_path / "nested" / "plot.png"

    helpers.t_SNE(
        _loader(),
        _FakeModel(),
        epochs=3,
        save_path=str(save_path),
        show=False,
    )

    assert save_path.parent.exists()
