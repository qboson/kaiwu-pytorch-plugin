"""Batch-index encoding contract for the qvae_cell trainer."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import pandas as pd

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
    "qvae_cell_trainer_batch", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


def test_categories_are_encoded_and_counted():
    """Category codes are returned in row order and n_batches is the category count."""
    adata = SimpleNamespace(obs=pd.DataFrame({"batch": ["b", "a", "b", "a"]}))
    trainer = Trainer.__new__(Trainer)
    trainer.n_batches = 0
    codes = trainer.batch_indices(adata, "batch")
    assert codes.tolist() == [1, 0, 1, 0]
    assert trainer.n_batches == 2
