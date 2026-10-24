"""Checkpoint-load contract for the QVAE-MNIST model tuner."""

import importlib
import importlib.util
import os
import sys

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
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

_spec = importlib.util.spec_from_file_location(
    "qvae_model_tuner_load",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "trainer", "model_tuner.py"),
)
model_tuner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_tuner)
ModelTuner = model_tuner.ModelTuner
sys.path[:] = ORIGINAL_PATH


def prepare(tmp_path):
    """Saves one state dict and returns a tuner plus the checkpoint path."""
    source = torch.nn.Linear(2, 3)
    checkpoint = tmp_path / "weights.pt"
    torch.save(source.state_dict(), checkpoint)
    tuner = ModelTuner()
    tuner._model = torch.nn.Linear(2, 3)
    tuner.infile = str(checkpoint)
    return tuner, source


def test_load_restores_weights_and_defaults_to_eval(tmp_path):
    """Loading copies tensors and switches the model to eval."""
    tuner, source = prepare(tmp_path)
    tuner._model.train()
    tuner.load_model()
    assert tuner._model.training is False
    assert torch.equal(tuner._model.weight, source.weight)
    assert torch.equal(tuner._model.bias, source.bias)


def test_load_can_keep_training_mode(tmp_path):
    """``set_eval=False`` leaves the model in training mode."""
    tuner, _source = prepare(tmp_path)
    tuner._model.train()
    tuner.load_model(set_eval=False)
    assert tuner._model.training is True
