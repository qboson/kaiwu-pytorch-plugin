"""Energy-computation contract for the qvae_cell trainer."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_cell"))
pre = importlib.import_module("kaiwu.preprocess")
if not hasattr(pre, "PrecisionReducer"):
    pre.PrecisionReducer = type(
        "PrecisionReducer", (), {"__init__": lambda self, *args, **kwargs: None}
    )

_spec = importlib.util.spec_from_file_location(
    "qvae_cell_trainer_energy", os.path.join(REPO_ROOT, "example", "qvae_cell", "trainer.py")
)
trainer_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer_module)
Trainer = trainer_module.Trainer


class StubModel:
    """Returns one fixed energy per input row and records the loss type."""

    def __init__(self):
        self.loss_types = []
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self

    def energy(self, x, loss_type):
        self.loss_types.append(loss_type)
        return torch.arange(x.shape[0], dtype=torch.float32) + 1.0


def test_energies_are_flattened_in_loader_order_with_the_configured_loss():
    """Per-batch energies are concatenated and the loss type is forwarded."""
    model = StubModel()
    loader = [
        (torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(1, 3), torch.zeros(1, dtype=torch.long)),
    ]
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(loss_type="bernoulli")
    trainer.device = torch.device("cpu")
    energies = trainer.compute_energy(model, loader)
    assert energies == [1.0, 2.0, 1.0]
    assert model.loss_types == ["bernoulli", "bernoulli"]
    assert model.evaluated is True
