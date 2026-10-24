"""Tuner-wiring contract for the MNIST QVAE trainer."""

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
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]

trainer_module = importlib.import_module("trainer.trainer")
sys.path[:] = ORIGINAL_PATH
Trainer = trainer_module.Trainer


class StubModel(torch.nn.Module):
    """Exposes encoder/decoder/bm modules and records the mean/bias setters."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)
        self.decoder = torch.nn.Linear(1, 1)
        self.bm = torch.nn.Linear(1, 1)
        self.dataset_means = []
        self.train_biases = []

    def set_dataset_mean(self, mean):
        self.dataset_means.append(mean)

    def set_train_bias(self, mean):
        self.train_biases.append(mean)


def build_trainer():
    """Builds only the attributes ``_setup_tuner`` reads."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(type="QVAE", lr=1e-3, bm_lr=5e-4)
    trainer.model = StubModel()
    trainer.train_loader = object()
    trainer.test_loader = object()
    trainer.output_dir = "outputs_smoke"
    trainer.dataset_mean = [0.5, 1.5]
    return trainer


def test_two_optimisers_and_mean_are_registered_for_qvae():
    """The QVAE path registers separate VAE/BM optimizers and the dataset mean."""
    trainer = build_trainer()
    tuner = trainer._setup_tuner()
    assert tuner._use_two_optimisers is True
    assert tuner.outpath == "outputs_smoke"
    assert tuner._model is trainer.model
    assert tuner.train_loader is trainer.train_loader
    assert trainer.model.dataset_means == [[0.5, 1.5]]
    assert trainer.model.train_biases == [[0.5, 1.5]]
    assert len(tuner._vae_optimiser.param_groups[0]["params"]) == 4
    assert len(tuner._bm_optimiser.param_groups[0]["params"]) == 2
