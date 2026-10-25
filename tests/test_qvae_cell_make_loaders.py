"""Loader-split contract for the single-cell trainer."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))
for _name in [m for m in list(sys.modules) if m == "models" or m.startswith("models.")]:
    del sys.modules[_name]
pre = importlib.import_module("kaiwu.preprocess")
if not hasattr(pre, "PrecisionReducer"):
    pre.PrecisionReducer = type(
        "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
    )

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_trainer_loaders", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


def collect_ids(loader):
    """Reads the batch-index column from every batch."""
    ids = []
    for _x, batch_idx in loader:
        ids.extend(batch_idx.tolist())
    return ids


def test_split_sizes_and_disjointness():
    """Ten cells split 8/2 with no overlap and a full evaluation loader."""
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(val_percentage=0.2, seed=42, batch_size=4)
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    ids = np.arange(10)
    train_loader, val_loader, eval_loader = trainer.make_loaders(x, ids)

    train_ids = collect_ids(train_loader)
    val_ids = collect_ids(val_loader)
    assert len(train_ids) == 8
    assert len(val_ids) == 2
    assert set(train_ids).isdisjoint(val_ids)
    assert sorted(train_ids + val_ids) == list(range(10))
    assert collect_ids(eval_loader) == list(range(10))
