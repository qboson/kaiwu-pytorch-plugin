"""Results-artifact contract for the QVAE-MNIST trainer."""

import importlib
import json
import os
import sys
from types import SimpleNamespace

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


def test_results_json_holds_losses_and_public_config(tmp_path):
    """Losses and non-private config values are written to results.json."""
    trainer = Trainer.__new__(Trainer)
    trainer.output_dir = str(tmp_path)
    trainer.train_losses = [1.0, 0.5]
    trainer.test_losses = [2.0, 1.0]
    trainer.config = SimpleNamespace(
        type="QVAE",
        num_epochs=2,
        batch_size=4,
        activation_fct=torch.nn.ReLU(),
        _hidden=7,
    )
    trainer.save_results()
    payload = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert payload["train_losses"] == [1.0, 0.5]
    assert payload["test_losses"] == [2.0, 1.0]
    assert payload["config"]["type"] == "QVAE"
    assert payload["config"]["num_epochs"] == 2
    assert payload["config"]["activation_fct"] == "ReLU"
    assert "_hidden" not in payload["config"]
