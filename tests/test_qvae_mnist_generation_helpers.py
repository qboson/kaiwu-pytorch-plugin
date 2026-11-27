"""The QVAE-MNIST generation helpers must use the model smoothing distribution."""

import importlib.util
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = REPO_ROOT / "example" / "qvae_mnist" / "utils" / "helpers.py"


def _load_helpers():
    spec = importlib.util.spec_from_file_location("qvae_mnist_helpers", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubExponential:  # pylint: disable=too-few-public-methods
    def __init__(self, beta):
        del beta

    def sample(self, shape):
        return torch.full(shape, 0.25)


class _RaisingExponential:  # pylint: disable=too-few-public-methods
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("torch Exponential must not be used for smoothing")


class _Progress:  # pylint: disable=too-few-public-methods
    def update(self, count):
        del count

    def close(self):
        return None


class _BM:  # pylint: disable=too-few-public-methods
    def sample(self, sampler):
        del sampler
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]])


class _Decoder:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.inputs = []

    def __call__(self, zeta):
        self.inputs.append(zeta.clone())
        return torch.zeros(zeta.size(0), 4)


class _Model:  # pylint: disable=too-few-public-methods
    def __init__(self):
        self.bm = _BM()
        self.decoder = _Decoder()
        self.dist_beta = 1.0
        self._train_bias = torch.zeros(4)

    def eval(self):
        return self


def test_generate_qvae_samples_uses_the_truncated_smoothing(monkeypatch) -> None:
    helpers = _load_helpers()
    monkeypatch.setattr(helpers, "Exponential", _StubExponential)
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda model: None)
    monkeypatch.setattr(helpers, "tqdm", lambda *args, **kwargs: _Progress())

    model = _Model()
    images = helpers.generate_qvae_samples(model, 1.0, n_images=3)

    assert images.shape[0] == 3
    first = model.decoder.inputs[0]
    expected = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    torch.testing.assert_close(first, expected)


def test_generate_qvae_images_uses_the_truncated_smoothing(monkeypatch) -> None:
    helpers = _load_helpers()
    monkeypatch.setattr(helpers, "Exponential", _StubExponential)
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda model: None)
    monkeypatch.setattr(helpers, "plot_flattened_images_grid", lambda *a, **k: None)
    monkeypatch.setattr(
        torch.distributions, "Exponential", _RaisingExponential, raising=False
    )

    model = _Model()
    images, _ = helpers.generate_qvae_images(model, save_path=".", grid_size=1)

    assert images.shape[0] == 2
    expected = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    torch.testing.assert_close(model.decoder.inputs[0], expected)
