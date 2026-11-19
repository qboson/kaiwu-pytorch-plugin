"""Validation outliers should not change training-cell Bernoulli scaling."""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("tqdm")
sklearn = pytest.importorskip("sklearn.model_selection")


def test_bernoulli_scaling_only_uses_training_cells_when_enabled(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "example" / "qvae_cell"
    monkeypatch.syspath_prepend(str(root))
    for name in ("trainer", "models"):
        monkeypatch.setitem(sys.modules, name, None)
        sys.modules.pop(name)
    trainer_module = importlib.import_module("trainer")
    args = SimpleNamespace(
        loss_type="bernoulli", fit_bernoulli_on_train=True, val_percentage=0.25,
        seed=17, batch_size=2,
    )
    trainer = trainer_module.Trainer(args, torch.device("cpu"))
    train, held_out = sklearn.train_test_split(np.arange(8), test_size=0.25, random_state=17)
    data = np.tile(np.linspace(0, 2, 8, dtype=np.float32).reshape(-1, 1), (1, 2))
    data[held_out] = [[-2000, 1000], [2000, -1000]]
    fit = data[train]
    expected = (fit - fit.min()) / (fit.max() - fit.min())
    _, _, evaluation = trainer.make_loaders(
        trainer.adata_to_array(SimpleNamespace(X=data)), np.zeros(8, dtype=int)
    )
    observed = evaluation.dataset.tensors[0].numpy()

    np.testing.assert_allclose(observed[train], expected)
    np.testing.assert_array_equal(observed[held_out], [[0, 1], [1, 0]])
