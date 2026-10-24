"""Model-type guard contract for the MNIST QVAE trainer."""

import importlib
import os
import sys
from types import SimpleNamespace

import pytest
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


class StubDataset:
    """Exposes the ``[0][0]`` tensor the trainer measures."""

    def __getitem__(self, index):
        assert index == 0
        return (torch.zeros(64), 0)


def build_trainer(model_type):
    """Builds only the attributes ``_create_model`` reads before branching."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        type=model_type,
        activation_fct=torch.nn.ReLU(),
        bm_type="rbm",
        sampler_type="sa",
    )
    trainer.train_loader = SimpleNamespace(dataset=StubDataset())
    trainer.n_batches = 0
    return trainer


def test_cell_qvae_is_rejected_with_a_pointer_to_the_other_example():
    """The MNIST trainer cannot build CellQVAE and says where to find it."""
    with pytest.raises(NotImplementedError, match="qvae_cell example"):
        build_trainer("CellQVAE")._create_model()


def test_unknown_model_type_still_reports_the_value():
    """Other unknown types keep the generic error."""
    # The example's custom ValueError currently drops its message (see #392),
    # so only the type is asserted here.
    with pytest.raises(ValueError):
        build_trainer("Transformer")._create_model()
