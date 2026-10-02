"""Regression test for the ``show`` flag of ``helpers.t_SNE``.

``Trainer._save_tsne_frame`` calls ``t_SNE(..., show=False)`` to render a frame
without displaying it.  ``t_SNE`` nevertheless called ``plt.show()``
unconditionally before honouring the flag, so frame rendering opened/blocked a
figure on interactive backends and displayed a figure that the caller asked not
to display.  The sibling ``plot_training_curves`` helper had the same defect.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import utils.helpers as helpers  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


class _StubQVAE(torch.nn.Module):
    """Minimal QVAE stand-in: forward returns ``(x, x, x, zeta)``."""

    def __init__(self, latent=4):
        super().__init__()
        self.lin = torch.nn.Linear(latent, latent)

    def forward(self, x):
        zeta = self.lin(x)
        return x, x, x, zeta


def _loader(n=40, d=4):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=generator)
    y = torch.randint(0, 10, (n,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_tsne_honours_show_false(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    show_calls = []
    close_calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: show_calls.append(1))
    monkeypatch.setattr(plt, "close", lambda *a, **k: close_calls.append(1))

    helpers.t_SNE(
        test_loader=_loader(),
        qvae_model=_StubQVAE(),
        epochs=1,
        save_path=str(tmp_path / "tsne.png"),
        show=False,
    )

    assert show_calls == [], "t_SNE(show=False) must not call plt.show()"
    assert len(close_calls) == 1, "t_SNE(show=False) must close the figure"


def test_tsne_still_shows_when_requested(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    show_calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: show_calls.append(1))
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    helpers.t_SNE(
        test_loader=_loader(),
        qvae_model=_StubQVAE(),
        epochs=1,
        save_path=str(tmp_path / "tsne_show.png"),
        show=True,
    )

    assert len(show_calls) == 1, "t_SNE(show=True) must display exactly one figure"
