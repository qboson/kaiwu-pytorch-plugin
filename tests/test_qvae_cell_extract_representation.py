"""Representation-extraction contract for the qvae_cell trainer."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

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
    "qvae_cell_trainer_repr", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


class StubModel:
    """Returns constant q logits and zeta samples."""

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, x, batch_idx):
        rows = x.shape[0]
        return None, None, torch.full((rows, 2), 1.0), torch.full((rows, 2), 5.0)


def build_loader():
    """Two batches of 2 and 1 rows with matching batch indices."""
    return [
        (torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(1, 3), torch.ones(1, dtype=torch.long)),
    ]


def build_trainer(representation):
    """Builds only the attributes extraction reads."""
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(representation=representation)
    trainer.device = torch.device("cpu")
    return trainer


def test_q_logits_are_concatenated_across_batches():
    """``representation='q'`` returns the encoder logits."""
    reps = build_trainer("q").extract_representation(StubModel(), build_loader())
    assert isinstance(reps, np.ndarray)
    assert reps.shape == (3, 2)
    assert np.all(reps == 1.0)


def test_zeta_samples_are_concatenated_across_batches():
    """``representation='zeta'`` returns the posterior samples."""
    reps = build_trainer("zeta").extract_representation(StubModel(), build_loader())
    assert reps.shape == (3, 2)
    assert np.all(reps == 5.0)
