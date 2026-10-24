"""Checkpoint-save contract for the QVAE-MNIST model tuner."""

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
    "qvae_model_tuner_save",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "trainer", "model_tuner.py"),
)
model_tuner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_tuner)
ModelTuner = model_tuner.ModelTuner
sys.path[:] = ORIGINAL_PATH


def test_save_model_writes_the_state_dict(tmp_path):
    """The checkpoint stores the state dict under ``model_<suffix>.pt``."""
    tuner = ModelTuner()
    tuner._model = torch.nn.Linear(2, 3)
    tuner.outpath = str(tmp_path)
    tuner.save_model("demo")
    path = tmp_path / "model_demo.pt"
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    assert "weight" in payload and "bias" in payload
    assert torch.equal(payload["weight"], tuner._model.state_dict()["weight"])
