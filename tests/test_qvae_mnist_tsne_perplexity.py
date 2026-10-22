"""Regression test for ``helpers.t_SNE`` on small evaluation sets.

``t_SNE`` hardcoded ``perplexity=30``.  scikit-learn requires
``perplexity < n_samples``, so the documented quick-validation workflow
(``run_pipeline.py --num-test-samples 20`` / a small ``--num-train-samples``
subset) aborted after training with:

    ValueError: perplexity (30) must be less than n_samples (20)

The helper must scale perplexity down for small sets instead of failing.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HELPERS_DIR = os.path.join(REPO_ROOT, "example", "qvae_mnist")
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import importlib.util  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def _load_helpers():
    """Load the example helpers by path (its ``utils`` package is a namespace
    package that other examples' regular ``utils`` packages can shadow)."""
    path = os.path.join(HELPERS_DIR, "utils", "helpers.py")
    spec = importlib.util.spec_from_file_location("qvae_mnist_helpers_ppl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


class _StubQVAE(torch.nn.Module):
    def __init__(self, latent=4):
        super().__init__()
        self.lin = torch.nn.Linear(latent, latent)

    def forward(self, x):
        zeta = self.lin(x)
        return x, x, x, zeta


def _loader(n, d=4):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=generator)
    y = torch.randint(0, 10, (n,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


class _RecordingTSNE:
    """Records constructor kwargs and returns a fixed 2-D embedding."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def fit_transform(self, values):
        return np.zeros((len(values), 2))


def test_small_evaluation_set_is_supported(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "TSNE", _RecordingTSNE)

    _, save_path, _ = helpers.t_SNE(
        test_loader=_loader(12),
        qvae_model=_StubQVAE(),
        epochs=1,
        save_path=str(tmp_path / "small.png"),
        show=False,
    )

    assert os.path.exists(save_path)
    assert _RecordingTSNE.last_kwargs["perplexity"] < 12


def test_perplexity_stays_at_the_default_for_large_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "TSNE", _RecordingTSNE)

    helpers.t_SNE(
        test_loader=_loader(100),
        qvae_model=_StubQVAE(),
        epochs=1,
        save_path=str(tmp_path / "large.png"),
        show=False,
    )

    assert _RecordingTSNE.last_kwargs["perplexity"] == 30
