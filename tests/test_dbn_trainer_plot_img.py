"""Regression test for the ``plot_img`` option of ``DBNTrainer``.

``plot_img=True`` promises training-progress plots, but the visualisation calls
were nested inside the ``verbose`` branch, so with the default ``verbose`` the
option produced nothing at all.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "dbn_digits"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

from dbn_trainer import DBNTrainer  # noqa: E402
from kaiwu.torch_plugin import RestrictedBoltzmannMachine  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


def _run_layer(monkeypatch, plot_img, verbose):
    calls = {"gen": 0, "weights": 0, "recon": 0}
    monkeypatch.setattr(
        DBNTrainer,
        "_train_batch",
        lambda self, rbm, optimizer, batch_x: torch.zeros((), requires_grad=True),
    )
    for name, key in (
        ("_visualize_generated_samples", "gen"),
        ("_visualize_weights_gradients", "weights"),
        ("_visualize_training_progress", None),
    ):
        if key is None:
            continue
        monkeypatch.setattr(
            DBNTrainer, name, lambda self, *a, _k=key, **kw: calls.__setitem__(_k, calls[_k] + 1)
        )
    monkeypatch.setattr(
        DBNTrainer,
        "_visualize_training_progress",
        lambda self, *a, **kw: (
            DBNTrainer._visualize_generated_samples(self, *a, **kw),
            DBNTrainer._visualize_weights_gradients(self, *a, **kw),
        ),
    )

    torch.manual_seed(0)
    rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3)
    data = np.random.RandomState(0).rand(4, 4).astype(np.float32)
    trainer = DBNTrainer(
        n_epochs_rbm=1, batch_size=4, plot_img=plot_img, verbose=verbose
    )
    trainer._train_rbm_layer(rbm, data, 0)
    return calls


def test_plot_img_works_without_verbose(monkeypatch):
    calls = _run_layer(monkeypatch, plot_img=True, verbose=False)
    assert calls["gen"] > 0 and calls["weights"] > 0, (
        "plot_img=True must visualise training progress even when verbose=False"
    )


def test_plot_img_false_does_not_visualise(monkeypatch):
    calls = _run_layer(monkeypatch, plot_img=False, verbose=False)
    assert calls == {"gen": 0, "weights": 0, "recon": 0}
