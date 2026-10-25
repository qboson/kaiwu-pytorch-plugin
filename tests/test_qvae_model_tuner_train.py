"""ModelTuner.train contract for the QVAE-MNIST example."""

import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINAL_PATH = list(sys.path)
sys.path[:] = [p for p in ORIGINAL_PATH if not p.endswith("qdiffusion") and not p.endswith("dplm")]
for _name in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(REPO_ROOT, "example", "qvae_mnist"))

_spec = importlib.util.spec_from_file_location(
    "qvae_model_tuner_train",
    os.path.join(REPO_ROOT, "example", "qvae_mnist", "trainer", "model_tuner.py"),
)
model_tuner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_tuner)
ModelTuner = model_tuner.ModelTuner
sys.path[:] = ORIGINAL_PATH


class FakeLoader:
    """Two batches with a dataset of four samples."""

    def __init__(self):
        self.dataset = list(range(4))
        self.batches = [
            (torch.zeros(2, 4), torch.tensor([0, 1])),
            (torch.ones(2, 4), torch.tensor([1, 0])),
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class StubModel(torch.nn.Module):
    """Exposes forward/loss/bm_loss over a trainable scalar."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.mode = None

    def train(self, mode=True):
        self.mode = "train"
        return self

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 64)
        q = torch.zeros(x.shape[0], 8)
        return logits, "posterior", q, "zeta"

    def loss(self, x, output_logits, posterior):
        return self.scale * 1.0

    def bm_loss(self, q, weight_decay):
        return self.scale * 2.0


class RecordingOptimizer:
    """Counts zero_grad/step calls."""

    def __init__(self):
        self.zeroed = 0
        self.steps = 0

    def zero_grad(self, *args, **kwargs):
        self.zeroed += 1

    def step(self, *args, **kwargs):
        self.steps += 1


def build_tuner():
    """Builds a two-stage tuner over the stubs."""
    tuner = ModelTuner()
    tuner._config = SimpleNamespace(type="QVAE", weight_decay=0.01)
    tuner._model = StubModel()
    tuner.train_loader = FakeLoader()
    tuner._use_two_optimisers = True
    tuner._vae_optimiser = RecordingOptimizer()
    tuner._bm_optimiser = RecordingOptimizer()
    return tuner


def test_two_stage_training_updates_both_optimizers_and_averages_per_sample():
    """Both optimizers step once per batch and the loss is per sample."""
    tuner = build_tuner()
    train_loss = tuner.train(epoch=1)
    assert tuner._model.mode == "train"
    assert tuner._vae_optimiser.zeroed == 2 and tuner._vae_optimiser.steps == 2
    assert tuner._bm_optimiser.zeroed == 2 and tuner._bm_optimiser.steps == 2
    # (1 + 2) per batch over two batches, divided by four samples.
    assert train_loss == 1.5
