"""Generation must add the Bernoulli bias only for Bernoulli models."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = REPO_ROOT / "example" / "qvae_mnist" / "utils" / "helpers.py"


def _load_helpers():
    spec = importlib.util.spec_from_file_location("qvae_mnist_helpers_bias", HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubExponential:  # pylint: disable=too-few-public-methods
    def __init__(self, beta):
        del beta

    def sample(self, shape):
        return torch.full(shape, 0.25)


class _BM:  # pylint: disable=too-few-public-methods
    def sample(self, sampler):
        del sampler
        return torch.tensor([[0.0, 1.0], [1.0, 0.0]])


class _Decoder:  # pylint: disable=too-few-public-methods
    def __call__(self, zeta):
        return torch.zeros(zeta.size(0), 4)


class _Model:  # pylint: disable=too-few-public-methods
    def __init__(self, loss_type):
        self.bm = _BM()
        self.decoder = _Decoder()
        self.dist_beta = 1.0
        self.config = SimpleNamespace(loss_type=loss_type)
        self._train_bias = torch.ones(4)

    def eval(self):
        return self


def test_train_bias_is_applied_only_for_bernoulli_models() -> None:
    helpers = _load_helpers()
    zeros = torch.zeros(2, 4)

    torch.testing.assert_close(
        helpers._apply_train_bias(_Model("mse"), zeros), zeros
    )
    torch.testing.assert_close(
        helpers._apply_train_bias(_Model("bernoulli"), zeros), torch.ones(2, 4)
    )


def test_generated_images_skip_the_bias_for_mse_models(monkeypatch) -> None:
    helpers = _load_helpers()
    monkeypatch.setattr(helpers, "Exponential", _StubExponential)
    monkeypatch.setattr(helpers, "init_qvae_sampler", lambda model: None)
    monkeypatch.setattr(helpers, "plot_flattened_images_grid", lambda *a, **k: None)

    images, _ = helpers.generate_qvae_images(_Model("mse"), save_path=".", grid_size=1)
    torch.testing.assert_close(images, torch.full((2, 4), 0.5))

    images, _ = helpers.generate_qvae_images(
        _Model("bernoulli"), save_path=".", grid_size=1
    )
    torch.testing.assert_close(images, torch.sigmoid(torch.ones(2, 4)))
