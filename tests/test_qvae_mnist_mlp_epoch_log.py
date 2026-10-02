"""Regression test for the MLP progress log of ``MLPClassifier``.

The periodic progress line printed the current epoch as both the numerator and
the denominator (``Epoch {epoch}/{epoch}``), so a 20-epoch run logged
``Epoch 10/10`` instead of ``Epoch 10/20``.
"""

import logging
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

import downstream.classifier as classifier_module  # noqa: E402
from downstream.classifier import MLPClassifier  # noqa: E402


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")


def test_progress_log_reports_the_total_epochs(caplog, monkeypatch):
    monkeypatch.setattr(
        classifier_module.MLPClassifier,
        "_train_mlp_epoch",
        lambda self, **kwargs: (90.0, 1.0),
    )
    monkeypatch.setattr(
        classifier_module.MLPClassifier,
        "_eval_mlp_epoch",
        lambda self, **kwargs: (80.0, 1.2),
    )

    rng = np.random.RandomState(0)
    x = rng.rand(40, 8).astype(np.float32)
    y = rng.randint(0, 3, 40)

    classifier = MLPClassifier(
        input_dim=8,
        hidden_dims=[4],
        output_dim=3,
        epochs_mlp=20,
        batch_size_mlp=8,
        device=torch.device("cpu"),
        save_path=None,
    )
    with caplog.at_level(logging.INFO):
        classifier.fit(x, y)

    assert "Epoch 10/20" in caplog.text
