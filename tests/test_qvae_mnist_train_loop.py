"""Training-loop orchestration contract for the QVAE-MNIST trainer."""

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [p for p in ORIGINAL_PATH if not p.endswith("qdiffusion") and not p.endswith("dplm")]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
for _name in [m for m in list(sys.modules) if m == "trainer" or m.startswith("trainer.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

trainer_module = importlib.import_module("trainer.trainer")
sys.path[:] = ORIGINAL_PATH
Trainer = trainer_module.Trainer


class FakeTuner:
    """Returns fixed losses and records the final checkpoint request."""

    def __init__(self):
        self.train_calls = 0
        self.test_calls = 0
        self.saved = []

    def train(self, epoch):
        self.train_calls += 1
        return torch.tensor(1.0)

    def test(self):
        self.test_calls += 1
        return (
            torch.tensor(2.0),
            torch.zeros(1, 4),
            torch.zeros(1, 4),
            [0],
        )

    def save_model(self, config_string=""):
        self.saved.append(config_string)


def build_trainer(num_epochs=4):
    """Builds a trainer whose heavy collaborators are stubbed."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(type="QVAE", num_epochs=num_epochs)
    trainer.train_losses = []
    trainer.test_losses = []
    trainer.output_dir = "unused"
    trainer.tuner = FakeTuner()
    trainer.model = object()
    trainer.test_loader = [(torch.zeros(2, 4), torch.tensor([0, 1]))]
    trainer.events = []
    trainer._setup_data = lambda: None
    trainer._create_model = lambda: trainer.model
    trainer._setup_tuner = lambda: trainer.tuner
    trainer._save_reconstruction = lambda *a: trainer.events.append("recon")
    trainer._plot_training_curve = lambda: trainer.events.append("curve")
    trainer._save_tsne_frame = lambda epoch: trainer.events.append(("frame", epoch))
    trainer._visualize_tsne = lambda: trainer.events.append("tsne")
    trainer.compute_energy = lambda model, loader: [1.0, 2.0]
    trainer._save_energy_plot = lambda energies, labels, output_path: trainer.events.append(
        ("energy", list(energies), list(labels))
    )
    trainer._save_reconstruction = lambda *a: trainer.events.append("recon")
    return trainer


def test_short_runs_do_not_divide_by_zero_and_save_the_final_model():
    """A 4-epoch run records losses, saves the final checkpoint and plots."""
    trainer = build_trainer(num_epochs=4)
    model, train_losses, test_losses = trainer.train()
    assert model is trainer.model
    assert train_losses == [1.0, 1.0, 1.0, 1.0]
    assert test_losses == [2.0, 2.0, 2.0, 2.0]
    assert trainer.tuner.train_calls == 4
    assert trainer.tuner.test_calls == 4
    assert trainer.tuner.saved == ["final_QVAE"]
    assert trainer.events.count("curve") == 1
    assert trainer.events.count("recon") == 4


def test_optional_stages_run_only_when_requested(monkeypatch, tmp_path):
    """t-SNE/energy/animation stages run behind their flags."""
    monkeypatch.setattr(trainer_module, "create_tsne_animation", lambda frames, output_path: None)
    trainer = build_trainer(num_epochs=4)
    trainer.output_dir = str(tmp_path)
    trainer.train(
        run_tsne=True,
        compute_energy=True,
        tsne_interval=2,
        generate_animation=True,
    )
    assert ("frame", 2) in trainer.events and ("frame", 4) in trainer.events
    assert "curve" in trainer.events
    assert "tsne" in trainer.events
    energy_events = [event for event in trainer.events if isinstance(event, tuple) and event[0] == "energy"]
    assert energy_events == [("energy", [1.0, 2.0], [0, 1])]
