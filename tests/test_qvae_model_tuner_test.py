"""ModelTuner.test contract for the QVAE-MNIST example."""

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
    "qvae_model_tuner_test",
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


class StubModel:
    """Returns fixed losses and logits, recording eval mode."""

    def __init__(self):
        self.mode = None
        self.losses = [torch.tensor(3.0), torch.tensor(5.0)]
        self.call = 0

    def train(self, mode=True):
        self.mode = "train"
        return self

    def eval(self):
        self.mode = "eval"
        return self

    def __call__(self, x):
        self.call += 1
        logits = torch.zeros(x.shape[0], 64)
        return logits, "posterior", "q", "zeta"

    def loss(self, x, output_logits, posterior):
        return self.losses[self.call - 1]


def test_test_loss_is_normalised_by_dataset_size_and_returns_last_batch(tmp_path):
    """The epoch loss divides by samples (not batches) and exposes the last batch."""
    tuner = ModelTuner()
    tuner._config = SimpleNamespace(type="QVAE")
    tuner._model = StubModel()
    tuner.test_loader = FakeLoader()
    test_loss, input_data, output_data, label_list = tuner.test()
    assert float(test_loss) == 2.0
    assert tuner._model.mode == "eval"
    assert torch.equal(input_data, torch.ones(2, 4))
    assert torch.allclose(output_data, torch.full((2, 64), 0.5))
    assert label_list is None
