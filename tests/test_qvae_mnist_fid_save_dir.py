"""Regression test for ``helpers.evaluate_qvae_fid`` and its save_path.

The helper writes ``fid_results.txt`` into ``save_path`` after computing the
(expensive) FID score, but never created the directory, so any save_path that
does not exist yet raised FileNotFoundError and the score was lost.
"""

import os
import sys
import types

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, "src")))

import torch  # noqa: E402

import utils.helpers as helpers  # noqa: E402


def _trainer():
    return types.SimpleNamespace(
        config=types.SimpleNamespace(num_epochs=3),
        model=types.SimpleNamespace(_latent_dimensions=8),
    )


def test_fid_save_path_is_created(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "compute_fid_in_batches", lambda *a, **k: 1.25)

    save_path = tmp_path / "new" / "fid_dir"
    assert not save_path.exists()

    score = helpers.evaluate_qvae_fid(
        trainer=_trainer(),
        fake_imgs=torch.zeros(4, 784),
        real_imgs=torch.zeros(4, 784),
        device="cpu",
        save_path=str(save_path),
    )

    assert score == 1.25
    written = save_path / "fid_results.txt"
    assert written.exists(), "FID result must be written into the requested directory"
    assert "1.25" in written.read_text(encoding="utf-8")


def test_fid_without_save_path_still_returns_the_score(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "compute_fid_in_batches", lambda *a, **k: 0.5)

    score = helpers.evaluate_qvae_fid(
        trainer=_trainer(),
        fake_imgs=torch.zeros(2, 784),
        real_imgs=torch.zeros(2, 784),
        device="cpu",
        save_path=None,
    )

    assert score == 0.5
