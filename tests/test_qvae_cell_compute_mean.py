"""Dataset-mean contract for the qvae_cell trainer."""

import importlib
import importlib.util
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))
pre = importlib.import_module("kaiwu.preprocess")
if not hasattr(pre, "PrecisionReducer"):
    pre.PrecisionReducer = type(
        "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
    )

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_trainer_mean", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


def test_mean_is_weighted_by_samples_per_batch():
    """Batches contribute in proportion to their row counts."""
    trainer = Trainer.__new__(Trainer)
    loader = [
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.zeros(2, dtype=torch.long)),
        (torch.tensor([[5.0, 6.0]]), torch.zeros(1, dtype=torch.long)),
    ]
    mean = trainer.compute_mean_x(loader)
    assert isinstance(mean, np.ndarray)
    assert mean.tolist() == [3.0, 4.0]
