"""Energy-collection contract for the MNIST trainer."""

import importlib
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [
    entry
    for entry in ORIGINAL_PATH
    if not entry.endswith("qdiffusion") and not entry.endswith("dplm")
]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
for _name in [m for m in list(sys.modules) if m == "trainer" or m.startswith("trainer.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

trainer_module = importlib.import_module("trainer.trainer")
sys.path[:] = ORIGINAL_PATH
Trainer = trainer_module.Trainer


class StubModel(torch.nn.Module):
    """Returns a fixed energy vector per batch and records the loss type."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.loss_types = []
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self

    def energy(self, x, loss_type):
        self.loss_types.append(loss_type)
        return torch.arange(x.shape[0], dtype=torch.float32) + 1.0


def test_energies_are_flattened_in_loader_order_with_the_configured_loss():
    """Per-batch energies are concatenated and the configured loss type is used."""
    model = StubModel()
    loader = [
        (torch.zeros(2, 3), torch.zeros(2, dtype=torch.long)),
        (torch.zeros(1, 3), torch.zeros(1, dtype=torch.long)),
    ]
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(loss_type="bernoulli")
    energies = trainer.compute_energy(model, loader)
    assert energies == [1.0, 2.0, 1.0]
    assert model.loss_types == ["bernoulli", "bernoulli"]
    assert model.evaluated is True
